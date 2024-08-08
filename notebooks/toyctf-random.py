# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %% [markdown] magic_args="[markdown]"
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
#
# # Random agent playing the Capture The Flag toy environment

# %%
import sys
import logging
import gymnasium as gym

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s: %(message)s")
# %matplotlib inline

# %% [markdown]

# ### CyberBattle simulation
# - **Environment**: a network of nodes with assigned vulnerabilities/functionalities, value, and firewall configuration
# - **Action space**: local attack | remote attack | authenticated connection
# - **Observation**: effects of action on environment

# %%
from typing import cast
from cyberbattle._env.cyberbattle_env import CyberBattleEnv

_gym_env = gym.make("CyberBattleToyCtf-v0")

gym_env = cast(CyberBattleEnv, _gym_env)

# %%
gym_env.environment

# %%
gym_env.action_space

# %%
gym_env.action_space.sample()

# %% [markdown]
# ## A random agent

# %%
for i_episode in range(1):
    observation, _ = gym_env.reset()

    total_reward = 0

    for t in range(5600):
        action = gym_env.sample_valid_action()

        observation, reward, done, _, info = gym_env.step(action)

        total_reward += reward

        if reward > 0:
            print("####### rewarded action: {action}")
            print(f"total_reward={total_reward} reward={reward}")
            gym_env.render()

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

    gym_env.render()

gym_env.close()
print("simulation ended")

# %% [markdown]
# ### End of simulation

# %%
