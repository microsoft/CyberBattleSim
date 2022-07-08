# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test the CyberBattle Gym environment"""

import pytest
import gym
import numpy as np

from .cyberbattle_env import AttackerGoal


def test_few_gym_iterations() -> None:
    """Run a few iterations of the gym environment"""
    env = gym.make('CyberBattleToyCtf-v0')

    for _ in range(2):
        env.reset()
        action_mask = env.compute_action_mask()
        assert action_mask
        for t in range(12):
            # env.render()

            # sample a valid action
            action = env.sample_valid_action()

            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

    env.close()
    pass


def test_step_after_done() -> None:
    actions = [
        {'local_vulnerability': np.array([0, 1])},  # done=False r=9.0
        {'remote_vulnerability': np.array([0, 1, 0])},  # done=False r=4.0
        {'connect': np.array([0, 1, 2, 0])},  # done=False r=100.0
        {'local_vulnerability': np.array([1, 3])},  # done=False r=9.0
        {'connect': np.array([0, 2, 3, 1])},  # done=False r=100.0
        {'remote_vulnerability': np.array([1, 2, 1])},  # done=False r=6.0
        {'remote_vulnerability': np.array([1, 2, 0])},  # done=False r=6.0
        {'remote_vulnerability': np.array([2, 1, 1])},  # done=False r=2.0
        {'local_vulnerability': np.array([1, 0])},  # done=False r=6.0
        {'local_vulnerability': np.array([1, 1])},  # done=False r=0.0
        {'local_vulnerability': np.array([2, 1])},  # done=False r=6.0
        {'remote_vulnerability': np.array([2, 3, 0])},  # done=False r=4.0
        {'local_vulnerability': np.array([2, 4])},  # done=False r=9.0
        {'connect': np.array([0, 3, 2, 2])},  # done=False r=100.0
        {'local_vulnerability': np.array([3, 3])},  # done=False r=9.0
        {'local_vulnerability': np.array([3, 0])},  # done=False r=6.0
        {'remote_vulnerability': np.array([0, 4, 1])},  # done=False r=8.0
        {'local_vulnerability': np.array([3, 1])},  # done=False r=0.0
        {'connect': np.array([2, 4, 3, 3])},  # done=False r=100.0
        {'remote_vulnerability': np.array([1, 3, 1])},  # done=False r=2.0
        {'remote_vulnerability': np.array([1, 4, 0])},  # done=False r=6.0
        {'local_vulnerability': np.array([4, 1])},  # done=False r=6.0
        {'remote_vulnerability': np.array([0, 5, 0])},  # done=False r=4.0
        {'local_vulnerability': np.array([4, 4])},  # done=False r=9.0
        {'connect': np.array([3, 5, 2, 4])},  # done=False r=100.0
        {'remote_vulnerability': np.array([2, 5, 1])},  # done=False r=2.0
        {'local_vulnerability': np.array([5, 3])},  # done=False r=9.0
        {'connect': np.array([2, 6, 3, 5])},  # done=False r=100.0
        {'remote_vulnerability': np.array([4, 6, 1])},  # done=False r=6.0
        {'local_vulnerability': np.array([5, 0])},  # done=False r=6.0
        {'remote_vulnerability': np.array([4, 6, 0])},  # done=False r=6.0
        {'local_vulnerability': np.array([5, 1])},  # done=False r=0.0
        {'local_vulnerability': np.array([6, 1])},  # done=False r=6.0
        {'remote_vulnerability': np.array([6, 7, 0])},  # done=False r=4.0
        {'remote_vulnerability': np.array([0, 7, 1])},  # done=False r=2.0
        {'local_vulnerability': np.array([6, 4])},  # done=False r=9.0
        {'connect': np.array([4, 7, 2, 6])},  # done=False r=100.0
        {'local_vulnerability': np.array([7, 3])},  # done=False r=9.0
        {'connect': np.array([0, 8, 3, 7])},  # done=False r=100.0
        {'remote_vulnerability': np.array([0, 8, 0])},  # done=False r=6.0
        {'local_vulnerability': np.array([7, 0])},  # done=False r=6.0
        {'local_vulnerability': np.array([8, 4])},  # done=False r=9.0
        {'remote_vulnerability': np.array([3, 9, 1])},  # done=False r=2.0
        {'connect': np.array([3, 9, 2, 8])},  # done=False r=100.0
        {'remote_vulnerability': np.array([4, 9, 0])},  # done=False r=2.0
        {'local_vulnerability': np.array([9, 0])},  # done=False r=6.0
        {'remote_vulnerability': np.array([3, 8, 1])},  # done=False r=6.0
        {'remote_vulnerability': np.array([6, 10, 0])},  # done=False r=6.0
        {'local_vulnerability': np.array([9, 1])},  # done=False r=0.0
        {'local_vulnerability': np.array([9, 3])},  # done=False r=9.0
        {'remote_vulnerability': np.array([8, 10, 1])},  # done=False r=8.0
        {'local_vulnerability': np.array([7, 1])},  # done=False r=0.0
        {'connect': np.array([8, 10, 3, 9])},  # done=False r=100.0
        {'local_vulnerability': np.array([10, 4])},  # done=False r=9.0
        {'local_vulnerability': np.array([8, 1])},  # done=False r=6.0
        {'connect': np.array([7, 11, 2, 10])},  # done=True r=5000.0

        # this is one too many (after done)
        {'connect': np.array([10, 5, 2, 4])},
    ]

    env = gym.make('CyberBattleChain-v0', size=10, attacker_goal=AttackerGoal(own_atleast_percent=1.0))
    env.reset()
    for a in actions[:-1]:
        observation, reward, done, info = env.step(a)
        print(f"{a}, # done={done} r={reward}")

    with pytest.raises(RuntimeError, match=r'new episode must be started with env\.reset\(\)'):
        env.step(actions[-1])


@pytest.mark.parametrize('env_name', ['CyberBattleToyCtf-v0', 'CyberBattleRandom-v0', 'CyberBattleChain-v0'])
def test_wrap_spec(env_name) -> None:
    env = gym.make(env_name)

    class DummyWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            assert hasattr(self, 'spec')
            self.spec.dummy = 7

    assert hasattr(env.spec, 'properties')
    assert hasattr(env.spec, 'ports')
    assert hasattr(env.spec, 'local_vulnerabilities')
    assert hasattr(env.spec, 'remote_vulnerabilities')

    env = DummyWrapper(env)

    assert hasattr(env.spec, 'properties')
    assert hasattr(env.spec, 'ports')
    assert hasattr(env.spec, 'local_vulnerabilities')
    assert hasattr(env.spec, 'remote_vulnerabilities')
    assert hasattr(env.spec, 'dummy')
