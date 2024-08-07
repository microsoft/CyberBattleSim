# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A sample run of the CyberBattle simulation"""

from typing import cast
import gymnasium as gym
import logging
import sys
from cyberbattle._env.cyberbattle_env import CyberBattleEnv


def main() -> int:
    """Entry point if called as an executable"""

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)

    env = cast(CyberBattleEnv, gym.make("CyberBattleToyCtf-v0"))

    # logging.info(env.action_space.sample())
    # logging.info(env.observation_space.sample())

    for i_episode in range(1):
        observation, _ = env.reset()
        action_mask = env.compute_action_mask()
        total_reward = 0
        for t in range(500):
            # env.render()

            # sample a valid action
            action = env.sample_valid_action()
            while not env.apply_mask(action, action_mask):
                action = env.sample_valid_action()

            print("action: " + str(action))
            observation, reward, done, truncated, info = env.step(action)
            action_mask = observation["action_mask"]
            total_reward = total_reward + reward
            # print(observation)
            print("total_reward=" + str(total_reward))
            if truncated:
                print("Episode truncated after {} timesteps".format(t + 1))
                break
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

    env.close()
    return 0


if __name__ == "__main__":
    main()
