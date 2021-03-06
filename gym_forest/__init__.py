import os
import logging

from gym.envs.registration import register

logger = logging.getLogger(__name__)

def datapath(name):
    return os.path.join(os.path.dirname(__file__), 'datasets', name)

register(
    id='forest-train-qvm-v0',
    entry_point='gym_forest.envs:ForestDiscreteEnv',
    kwargs={'data': datapath('all_data_train.npy'),
            'label': datapath('all_data_labels.npy'),
            'shuffle': True})

register(
    id='forest-train-qpu-v0',
    entry_point='gym_forest.envs:ForestDiscreteEnv',
    kwargs={'data': datapath('all_data_train.npy'),
            'label': datapath('all_data_labels.npy'),
            'shuffle': True,
            'qpu': True})

register(
    id='forest-maxcut-valid-v0',
    entry_point='gym_forest.envs:ForestDiscreteEnv',
    kwargs={'data': datapath('maxcut_n10_e5_valid.npy'),
            'label': 'maxcut'})

register(
    id='forest-maxqp-valid-v0',
    entry_point='gym_forest.envs:ForestDiscreteEnv',
    kwargs={'data': datapath('maxqp_n10_valid.npy'),
            'label': 'maxqp'})

register(
    id='forest-qubo-valid-v0',
    entry_point='gym_forest.envs:ForestDiscreteEnv',
    kwargs={'data': datapath('qubo_n10_valid.npy'),
            'label': 'qubo'})

register(
    id='forest-maxcut-test-v0',
    entry_point='gym_forest.envs:ForestDiscreteEnv',
    kwargs={'data': datapath('maxcut_n10_e5_test.npy'),
            'label': 'maxcut'})

register(
    id='forest-maxqp-test-v0',
    entry_point='gym_forest.envs:ForestDiscreteEnv',
    kwargs={'data': datapath('maxqp_n10_test.npy'),
            'label': 'maxqp'})

register(
    id='forest-qubo-test-v0',
    entry_point='gym_forest.envs:ForestDiscreteEnv',
    kwargs={'data': datapath('qubo_n10_test.npy'),
            'label': 'qubo'})
