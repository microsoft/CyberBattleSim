# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helper to run the random agent from a jupyter notebook"""

import cyberbattle._env.cyberbattle_env as cyberbattle_env
import logging

LOGGER = logging.getLogger(__name__)


def run_random_agent(episode_count: int, iteration_count: int, gym_env: cyberbattle_env.CyberBattleEnv):
    """Run a simple random agent on the specified gym environment and
    plot exploration graph and reward function
    """

    for i_episode in range(episode_count):
        observation = gym_env.reset()

        total_reward = 0.0

        for t in range(iteration_count):
            action = gym_env.sample_valid_action()

            LOGGER.debug(f"action={action}")
            observation, reward, done, info = gym_env.step(action)

            total_reward += reward

            if reward > 0:
                prettry_printed = gym_env.pretty_print_internal_action(action)
                print(f'+ rewarded action: {action} total_reward={total_reward} reward={reward} @t={t}\n  {prettry_printed}')
                gym_env.render()

            if done:
                print(f"Episode finished after {t+1} timesteps")
                break

        gym_env.render()

    gym_env.close()
    print("simulation ended")
