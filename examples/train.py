# Copyright 2019, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


# This contains an example of training a `stable_baselines` agent
# against a gym_forest environment.

import os
import datetime

from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

import gym
import gym_forest

MAX_N_MODELS = 4
SAVE_FREQUENCY = 10000

env_name = 'forest-train-qvm-v0'
policy = MlpPolicy
n_steps = 512
lam = .999
seed = 1337
time_stamp = datetime.datetime.now().timestamp()
parameters = [env_name, policy.__name__, n_steps, lam, seed, time_stamp]

label = '-'.join([str(i) for i in parameters])
log_dir = os.path.join(os.path.dirname(__file__), '..', 'models', label)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
print(log_dir)

env = gym.make(env_name)
env.seed(seed)
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

model = PPO2(policy, env, n_steps=n_steps, lam=lam)
for iteration in range(MAX_N_MODELS):
    model.learn(total_timesteps=SAVE_FREQUENCY)
    model.save('./{}/model_{}.p'.format(log_dir, iteration))

env.close()
del model
