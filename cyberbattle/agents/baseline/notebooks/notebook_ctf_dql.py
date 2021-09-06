# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,title,-all
#     cell_metadata_json: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 2.7.12 64-bit
#     language: python
#     name: python271264bit467ff8c3aa8a4177808b84ed66cfa565
# ---

# %%
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Notebook used to benchmark all the baseline agents
on a given CyberBattleSim environment and compare
them to the dumb 'random agent' baseline.

NOTE: You can run this `.py`-notebook directly from VSCode.
You can also generate a traditional Jupyter Notebook
using the VSCode command `Export Currenty Python File As Jupyter Notebook`.
"""

# pylint: disable=invalid-name

# %%
import sys
import logging
import gym

import cyberbattle.agents.baseline.learner as learner
import cyberbattle.agents.baseline.plotting as p
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.agent_dql as dqla
import cyberbattle.agents.baseline.agent_tabularqlearning as a
from cyberbattle.agents.baseline.agent_wrapper import Verbosity

# import importlib
# importlib.reload(learner)

logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")

# %% {"tags": ["parameters"]}
# gymid = 'CyberBattleTiny-v0'
gymid = "CyberBattleToyCtf-v0"

iteration_count = 1500  # 2000
training_episode_count = 20  # 50
eval_episode_count = 10

# %%
# Load the Gym environment

gym_env = gym.make(gymid)


ep = w.EnvironmentBounds.of_identifiers(
    maximum_node_count=12,
    maximum_total_credentials=10,
    identifiers=gym_env.identifiers
)

# %%
# Evaluate the Deep Q-learning agent
dqn_learning_run = learner.epsilon_greedy_search(
    cyberbattle_gym_env=gym_env,
    environment_properties=ep,
    learner=dqla.DeepQLearnerPolicy(
        ep=ep,
        # gamma=1,  # 0.015,
        gamma=0.015,
        replay_memory_size=10000,
        target_update=5,
        batch_size=512,
        # torch default learning rate is 1e-2
        # a large value helps converge in less episodes
        learning_rate=0.01
    ),
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.90,
    # epsilon_multdecay=0.75,  # 0.999,
    epsilon_exponential_decay=5000,  # 10000
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    render=False,
    plot_episodes_length=False,
    title="DQL"
)

# %%
tql = learner.epsilon_greedy_search(
    gym_env,
    ep,
    a.QTabularLearner(ep, gamma=0.015, learning_rate=0.01, exploit_percentile=100),
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.90,
    render=False,
    # epsilon_multdecay=0.75,  # 0.999,
    epsilon_exponential_decay=5000,  # 10000
    epsilon_minimum=0.01,
    verbosity=Verbosity.Quiet,
    plot_episodes_length=False,
    title="Q-learning"
)

# %%
# Evaluate the pre-trained DQL agent
dql_exploit_run = learner.epsilon_greedy_search(
    gym_env,
    ep,
    learner=dqn_learning_run['learner'],
    episode_count=eval_episode_count,
    iteration_count=iteration_count,
    epsilon=0.0,  # 0.35,
    render=False,
    title="Exploiting DQL",
    plot_episodes_length=False,
    verbosity=Verbosity.Quiet
)


# %%
# Evaluate the random agent
random_run = learner.epsilon_greedy_search(
    gym_env,
    ep,
    learner=learner.RandomPolicy(),
    episode_count=eval_episode_count,
    iteration_count=iteration_count,
    epsilon=1.0,  # purely random
    render=False,
    verbosity=Verbosity.Quiet,
    plot_episodes_length=False,
    title="Random search"
)

# %%
# Plot averaged cumulative rewards for DQL vs Random vs DQL-Exploit
themodel = dqla.CyberBattleStateActionModel(ep)
p.plot_averaged_cummulative_rewards(
    all_runs=[
        dqn_learning_run,
        random_run,
        tql,
        dql_exploit_run
    ],
    title=f'Benchmark -- max_nodes={ep.maximum_node_count}, episodes={eval_episode_count},\n'
    f'State: {[f.name() for f in themodel.state_space.feature_selection]} '
    f'({len(themodel.state_space.feature_selection)}\n'
    f"Action: abstract_action ({themodel.action_space.flat_size()})")


# %%
# Plot cumulative rewards for all episodes
p.plot_all_episodes(dqn_learning_run)

# %% Plot episode length
# p.plot_episodes_length([
#     dqn_learning_run,
#     random_run,
#     dql_exploit_run
# ])

# %%
