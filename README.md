>Notice: This is research code that will not necessarily be maintained to
>support further releases of Forest and other Rigetti Software. We welcome bug
>reports and PRs but make no guarantee about fixes or responses.

# gym-forest

Gym environment for classical synthesis of quantum programs. For more
information about this, see our paper "Automated Quantum Programming via
Reinforcement Learning for Combinatorial Optimization".


## Installation

In addition to cloning this repository, you will need to download the data files
associated with the latest GitHub release. They should be unzipped at the
toplevel of the `gym-forest` code directory.

```
git clone https://github.com/rigetti/gym-forest.git
cd gym-forest
tar -xzf data.tar.bz2
pip install .
```

## Usage

This library provides several OpenAI gym environments available for
reinforcement learning tasks. For a very simple example, try the following at a
python repl:

```
>>> import gym
>>> import gym_forest

>>> env = gym.make('forest-train-qvm-v0')
>>> obs = env.reset()
>>> print(obs)
...
```

This environment contains problem instances from the combined MaxCut, MaxQP, and
QUBO training sets. Resetting the environment selects a random problem instance.

Actions are represented numerically, but encode a discrete set of gates:
```
>>> action = env.action_space.sample()
>>> env.instrs[action]
<Gate RX(3*pi/2) 9>
>>> obs, reward, done, info = env.step(action)
...
```

For more information, please take a look at `gym_forest/envs/gym_forest.py` and
comments within.

## Available Environments

We provide sets of randomly generated combinatorial optimization problems, and
Gym environments for solving these using a quantum resource, either simulated
(QVM) or real (QPU). The datasets are described in more detail in our paper, but
we summarize the environments below:


| Environment Name           | Resource | Problem Type        | Split      |
|----------------------------|----------|---------------------|------------|
| `'forest-train-qvm-v0'`    | QVM      | Maxcut, MaxQP, QUBO | Training   |
| `'forest-train-qpu-v0'`    | QPU      | Maxcut, MaxQP, QUBO | Training   |
| `'forest-maxcut-valid-v0'` | QVM      | Maxcut              | Validation |
| `'forest-maxqp-valid-v0'`  | QVM      | MaxQP               | Validation |
| `'forest-qubo-valid-v0'`   | QVM      | QUBO                | Validation |
| `'forest-maxcut-test-v0'`  | QVM      | Maxcut              | Testing    |
| `'forest-maxqp-test-v0'`   | QVM      | MaxQP               | Testing    |
| `'forest-qubo-test-v0'`    | QVM      | QUBO                | Testing    |

## Examples

The models in our paper were developed using the [stable
baselines](https://github.com/hill-a/stable-baselines) library. To use this, you
may `pip install -r examples/requirements.txt`.

### Example: Single-episode Rollout

See the code in `examples/rollout_episode.py` for an example showing the
behavior of at "QVM-trained" model on a a random maxcut test problem.

Note: This example uses the saved model weights, included in `data.tar.bz2`.

### Example: Training PPO agents.

See the code in `examples/train.py` for an example of how training can be
performed.

## Citation

If you use this code, please cite:

```
@article{mckiernan2019automated,
  title={Automated Quantum Programming via Reinforcement Learning for Combinatorial Optimization},
  author={McKiernan, K. and Davis, E. and Alam, M. S. and Rigetti, C.},
  note={TODO},
  year={2019}
}
```
