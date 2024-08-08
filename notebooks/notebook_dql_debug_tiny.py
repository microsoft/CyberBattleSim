# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,title,-all
#     cell_metadata_json: true
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

# %% {"tags": []}
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Notebook used for debugging purpose to train the
the DQL agent and then run it one step at a time.
"""

# pylint: disable=invalid-name
# %matplotlib inline

# %% {"tags": []}
import sys
import logging
import gymnasium as gym
import cyberbattle.agents.baseline.learner as learner
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.agent_dql as dqla
from cyberbattle.agents.baseline.agent_wrapper import ActionTrackingStateAugmentation, AgentWrapper, Verbosity

logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")

# %% {"tags": ["parameters"]}
gymid = "CyberBattleTiny-v0"
iteration_count = 150
training_episode_count = 10


# %% {"tags": []}
# Load the gym environment

_gym_env = gym.make(gymid)

from typing import cast
from cyberbattle._env.cyberbattle_env import CyberBattleEnv

ctf_env = cast(CyberBattleEnv, _gym_env)

ep = w.EnvironmentBounds.of_identifiers(maximum_node_count=12, maximum_total_credentials=10, identifiers=ctf_env.identifiers)

# %% {"tags": []}
# Evaluate the Deep Q-learning agent
dqn_learning_run = learner.epsilon_greedy_search(
    cyberbattle_gym_env=ctf_env,
    environment_properties=ep,
    learner=dqla.DeepQLearnerPolicy(
        ep=ep,
        gamma=0.015,
        replay_memory_size=10000,
        target_update=5,
        batch_size=512,
        learning_rate=0.01,  # torch default learning rate is 1e-2
    ),
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.90,
    epsilon_exponential_decay=5000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    render=False,
    plot_episodes_length=False,
    title="DQL",
)

# %% {"tags": []}
# initialize the environment

current_o, _ = ctf_env.reset()
wrapped_env = AgentWrapper(ctf_env, ActionTrackingStateAugmentation(ep, current_o))
_l = dqn_learning_run["learner"]

# %% {"tags": []}
# Use the trained agent to run the steps one by one

max_steps = 10

# next action suggested by DQL agent
h = []
for i in range(max_steps):
    # run the suggested action
    _, next_action, _ = _l.exploit(wrapped_env, current_o)
    h.append((ctf_env.get_explored_network_node_properties_bitmap_as_numpy(current_o), next_action))
    print(h[-1])
    if next_action is None:
        break
    current_o, _, _, _, _ = wrapped_env.step(next_action)

print(f"len: {len(h)}")

# %% {"tags": []}
ctf_env.render()
