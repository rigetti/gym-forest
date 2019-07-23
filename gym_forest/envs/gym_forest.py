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

NUM_ANGLES = 8
NUM_SHOTS = 10
MAX_PROGRAM_LENGTH = 25


class MinimalPyQVMCompiler(AbstractCompiler):
    def quil_to_native_quil(self, program):
        return program

    def native_quil_to_executable(self, nq_program):
        return nq_program


def bitstring_index(bitstring):
    return int("".join(map(str, bitstring)), 2)


def lift_bitstring_function(n, f):
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
    @property
    @abstractmethod
    def num_problems(self) -> int:
        pass

    @property
    @abstractmethod
    def num_variables(self) -> int:
        pass

    @abstractmethod
    def problem(self, i: int) -> np.ndarray:
        pass

    @abstractmethod
    def bitstrings_score(self, i: int) -> Callable[[np.ndarray], float]:
        pass


class AllProblems(ProblemSet):
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
        self._instrs = [CNOT(q0, q1) for q0, q1 in product(qubits, qubits) if q0 != q1]
        self._instrs += [op(theta, q) for q, op, theta in product(qubits, [RX, RY, RZ], angles)]
        self.action_space = gym.spaces.Discrete(len(self._instrs))

        obs_len = NUM_SHOTS * self.num_qubits + len(self.pset.problem(0))
        self.observation_space = gym.spaces.Box(np.full(obs_len, -1.0), np.full(obs_len, 1.0), dtype=np.float32)

        self.reward_threshold = .8

        self.qpu = qpu
        if qpu:
            self._qc = get_qc('Aspen-4-16Q-A')
        else:
            self._qc = QuantumComputer(name='qvm',
                                       qam=PyQVM(n_qubits=self.num_qubits),
                                       device=NxDevice(nx.complete_graph(self.num_qubits)),
                                       compiler=MinimalPyQVMCompiler())

        self.reset()

    def reset(self, problem_id=None):
        if problem_id is None:
            problem_id = np.random.randint(0, self.pset.num_problems())

        self.problem_id = problem_id
        self._prob_vec = self.pset.problem(self.problem_id)
        self._prob_score = self.pset.bitstrings_score(self.problem_id)

        self.program = Program([I(q) for q in range(self.num_qubits)])
        self.current_step = 0
        self.running_episode_reward = 0

        self.bitstrings, info = self._run_program(self.program)
        return self.observation

    @property
    def observation(self):
        return np.concatenate([self.bitstrings.flatten(), self._prob_vec])

    def step(self, action):
        instr = self._instrs[action]
        self.program.inst(instr)
        self.bitstrings, info = self._run_program(self.program)
        reward = self._prob_score(self.bitstrings)
        self.running_episode_reward += reward

        info['instr'] = instr
        info['reward-nb'] = reward
        self.current_step += 1

        done = False
        if self.current_step >= MAX_PROGRAM_LENGTH:
            done = True
        if reward >= self.reward_threshold:
            reward += (MAX_PROGRAM_LENGTH - self.current_step)
            done = True

        return self.observation, reward, done, info

    def _wrap_program(self, program):
        ro = program.declare('ro', 'BIT', self.num_qubits)
        for q in range(self.num_qubits):
            program.inst(MEASURE(q, ro[q]))
        program.wrap_in_numshots_loop(NUM_SHOTS)
        return program

    def _run_program(self, program):
        program = program.copy()

        if self.qpu:
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
            'gate_count': gate_count
        }
        return results, info

    def render(self, mode='human'):
        raise NotImplementedError("Rendering of this environment not currently supported.")

    def seed(self, seed):
        np.random.seed(seed)
