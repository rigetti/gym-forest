# Copyright 2019, Rigetti Computing
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


# This is an example of a single-episode rollout, using the QVM trained agent
# on the MaxCut test set.

import os

import gym
import gym_forest
from stable_baselines import PPO2

MODEL_FILE = os.path.join(os.path.dirname(__file__), '..', 'models', 'qvm.p')
ENV_NAME = 'forest-maxcut-test-v0'
MAX_STEPS = 25

env = gym.make(ENV_NAME)
agent = PPO2.load(MODEL_FILE)

obs = env.reset()
best_reward = 0
eps_reward = 0
for i in range(MAX_STEPS):
    action, _ = agent.predict(obs)
    obs, reward, done, info = env.step(action)
    eps_reward += reward
    if done:
        # early termination returns the remaining episode reward,
        # assuming that we do just as well on the remaining steps
        # here we get the corresponding single-step reward
        single_step_reward = reward/(MAX_STEPS-i)
        print('[{}]\t {}\t reward {:.3f}'.format(i, info['instr'], single_step_reward))
        best_reward = max(best_reward, single_step_reward)
        if i + 1 < MAX_STEPS:
            print("Terminated early!")
        break
    else:
        print('[{}]\t {}\t reward {:.3f}'.format(i, info['instr'], reward))
        best_reward = max(reward, best_reward)

print("\n\nCompleted Episode\n")
print("Total Episode Reward: {0:.3f}".format(eps_reward))
print("Max Single-Step Reward: {0:.3f}".format(best_reward))
