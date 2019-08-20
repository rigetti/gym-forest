# Copyright 2019, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import os
from abc import ABCMeta, abstractmethod
from itertools import product
from typing import Callable

import gym
import gym.spaces
import networkx as nx
import numpy as np
from pyquil import get_qc
from pyquil.api import QuantumComputer
from pyquil.api._qac import AbstractCompiler
from pyquil.device import NxDevice
from pyquil.gates import I, RX, RY, RZ, CNOT, MEASURE, RESET
from pyquil.pyqvm import PyQVM
from pyquil.quil import Program, Pragma
from pyquil.quilbase import Gate
from pyquil.unitary_tools import all_bitstrings

NUM_ANGLES = 8            # discrete actions involve rotation by multiples of 2*pi/NUM_ANGLES
NUM_SHOTS = 10            # how many measurement shots to use? for a N qubit problem this produces N*NUM_SHOTS bits of data
MAX_PROGRAM_LENGTH = 25   # limit to the number of actions taken in a given episode

QPU_NAME = 'Aspen-4-16Q-A'

class MinimalPyQVMCompiler(AbstractCompiler):
    def quil_to_native_quil(self, program):
        return program

    def native_quil_to_executable(self, nq_program):
        return nq_program


def bitstring_index(bitstring):
    "Recover an integer from its bitstring representation."
    return int("".join(map(str, bitstring)), 2)


def lift_bitstring_function(n, f):
    """Lifts a function defined on single bitstrings to arrays of bitstrings.

    Args:
        n: The number of bits in the bitsring.
        f: The bitstring function, which produces a float value.
    Returns:
        A function which, given a K x n array of 0/1 values, returns the
        mean of f applied across the K rows.
    """
    bss = all_bitstrings(n).astype(np.float64)
    vals = np.apply_along_axis(f, 1, bss)
    # normalize to be between 0 and 1
    m, M = np.min(vals), np.max(vals)
    if np.isclose(m, M):
        vals[:] = 0.5
    else:
        vals -= m
        vals *= 1 / (M - m)

    def _fn(bitstrings):
        indices = np.apply_along_axis(bitstring_index, 1, bitstrings)
        return np.mean(vals[indices])

    return _fn


class ProblemSet(metaclass=ABCMeta):
    """Base class representing an abstract problem set."""
    @property
    @abstractmethod
    def num_problems(self) -> int:
        "The number of problems in the problem set."
        pass

    @property
    @abstractmethod
    def num_variables(self) -> int:
        "The number of variables in any problem."
        pass

    @abstractmethod
    def problem(self, i: int) -> np.ndarray:
        "An array representing the ith problem."
        pass

    @abstractmethod
    def bitstrings_score(self, i: int) -> Callable[[np.ndarray], float]:
        "The scoring function associated with problem i."
        pass


class AllProblems(ProblemSet):
    """A problem set of combinatorial optimization problems.

    Args:
        weights: A numpy array of weight matrices. weights[k,i,j] is the
                 coupling between vertex i and j in the kth problem.
        labels: A list of string labels, either 'maxcut', 'maxqp', or 'qubo'.
    """
    def __init__(self, weights, labels):
        assert len(weights.shape) == 3
        assert weights.shape[1] == weights.shape[2]
        assert len(weights) == len(labels)

        self._weights = weights
        self._labels = labels

    def num_problems(self):
        return self._weights.shape[0]

    def num_variables(self):
        return self._weights.shape[1]

    def problem(self, i):
        # due to the symmetry of these problems, we only need to observe the upper triangular entries
        upper = np.triu_indices(self._weights.shape[1])
        return self._weights[i, :, :][upper]

    def bitstrings_score(self, i):
        W = self._weights[i, :, :]
        n = W.shape[0]

        if self._labels[i] == 'maxcut':
            def cutweight(bitstring):
                return sum((W[i, j] for i in range(n) for j in range(n)
                            if bitstring[i] != bitstring[j]), 0.0)

            return lift_bitstring_function(n, cutweight)
        elif self._labels[i] == 'maxqp':
            def quadratic(x):
                z = 2 * x - 1
                return np.dot(z.T, np.dot(W, z))

            return lift_bitstring_function(n, quadratic)
        elif self._labels[i] == 'qubo':
            def quadratic(x):
                return np.dot(x.T, np.dot(-W, x))

            return lift_bitstring_function(n, quadratic)


class ForestDiscreteEnv(gym.Env):
    """The Rigetti Forest environment.

    This implements a Gym environment for gate-based quantum computing with
    problem-specific rewards on the Rigetti hardware.

    Attributes:
        observation: A np.array, formed by concatenating observed bitstring values
                     with a vector containing the problem weights.
        observation_space: The (continuous) set of possible observations.
        action space: The space of discrete actions.
        instrs: A table mapping action IDs to PyQuil gates.

    Args:
        data: A path to a numpy dataset.
        label: Either a path to a dataset of labels, or a single label value.
        shuffle: A flag indicating whether the data should be randomly shuffled.
        qpu: A flag indicating whether to run on the qpu given by QPU_NAME.

    """
    def __init__(self, data, label, shuffle=False, qpu=False):
        weights = np.load(data)
        n_graphs = len(weights)

        # read labels from file, or as single label
        if os.path.exists(label):
            labels = np.load(label)
        else:
            labels = [label for _ in range(n_graphs)]

        if shuffle:
            self._shuffled_order = np.random.permutation(n_graphs)
            weights = weights[self._shuffled_order]
            labels = labels[self._shuffled_order]

        self.pset = AllProblems(weights, labels)
        self.num_qubits = self.pset.num_variables()

        qubits = list(range(self.num_qubits))
        angles = np.linspace(0, 2 * np.pi, NUM_ANGLES, endpoint=False)
        self.instrs = [CNOT(q0, q1) for q0, q1 in product(qubits, qubits) if q0 != q1]
        self.instrs += [op(theta, q) for q, op, theta in product(qubits, [RX, RY, RZ], angles)]
        self.action_space = gym.spaces.Discrete(len(self.instrs))

        obs_len = NUM_SHOTS * self.num_qubits + len(self.pset.problem(0))
        self.observation_space = gym.spaces.Box(np.full(obs_len, -1.0), np.full(obs_len, 1.0), dtype=np.float32)

        self.reward_threshold = .8

        self.qpu = qpu
        if qpu:
            self._qc = get_qc(QPU_NAME)
        else:
            self._qc = QuantumComputer(name='qvm',
                                       qam=PyQVM(n_qubits=self.num_qubits),
                                       device=NxDevice(nx.complete_graph(self.num_qubits)),
                                       compiler=MinimalPyQVMCompiler())

        self.reset()

    def reset(self, problem_id=None):
        """Reset the state of the environment.

        This clears out whatever program you may have assembled so far, and
        updates the active problem.

        Args:
            problem_id: The numeric index of the problem (relative to the problem set).
                        If None, a random problem will be chosen.
        """
        if problem_id is None:
            problem_id = np.random.randint(0, self.pset.num_problems())

        self.problem_id = problem_id
        self._prob_vec = self.pset.problem(self.problem_id)
        # the scoring function (for reward computation)
        self._prob_score = self.pset.bitstrings_score(self.problem_id)

        # we put some trivial gates on each relevant qubit, so that we can
        # always recover the problem variables from the program itself
        self.program = Program([I(q) for q in range(self.num_qubits)])
        self.current_step = 0
        self.running_episode_reward = 0

        self.bitstrings, info = self._run_program(self.program)
        return self.observation

    @property
    def observation(self):
        """Get the current observed quantum + problem state."""
        return np.concatenate([self.bitstrings.flatten(), self._prob_vec])

    def step(self, action):
        """Advance the environment by performing the specified action."""
        # get the instruction indicated by the action
        instr = self.instrs[action]
        # extend the program
        self.program.inst(instr)
        # run and get some measured bitstrings
        self.bitstrings, info = self._run_program(self.program)
        # compute the avg score of the bitstrings
        reward = self._prob_score(self.bitstrings)
        self.running_episode_reward += reward

        info['instr'] = instr
        info['reward-nb'] = reward
        self.current_step += 1

        # are we done yet?
        done = False
        if self.current_step >= MAX_PROGRAM_LENGTH:
            done = True
        if reward >= self.reward_threshold:
            reward += (MAX_PROGRAM_LENGTH - self.current_step)
            done = True

        return self.observation, reward, done, info

    def _wrap_program(self, program):
        # the actions select gates. but a pyquil program needs a bit more
        # namely, declaration of classical memory for readout, and suitable
        # measurement instructions
        ro = program.declare('ro', 'BIT', self.num_qubits)
        for q in range(self.num_qubits):
            program.inst(MEASURE(q, ro[q]))
        program.wrap_in_numshots_loop(NUM_SHOTS)
        return program

    def _run_program(self, program):
        program = program.copy()

        if self.qpu:
            # time to go through the compiler. whee!
            pragma = Program([Pragma('INITIAL_REWIRING', ['"PARTIAL"']), RESET()])
            program = pragma + program
            program = self._wrap_program(program)
            nq_program = self._qc.compiler.quil_to_native_quil(program)
            gate_count = sum(1 for instr in nq_program if isinstance(instr, Gate))
            executable = self._qc.compiler.native_quil_to_executable(nq_program)
            results = self._qc.run(executable=executable)
        else:
            program = self._wrap_program(program)
            gate_count = len(program)
            results = self._qc.run(program)

        info = {
            'gate_count': gate_count # compiled length for qpu, uncompiled for qvm
        }
        return results, info

    def render(self, mode='human'):
        raise NotImplementedError("Rendering of this environment not currently supported.")

    def seed(self, seed):
        np.random.seed(seed)
