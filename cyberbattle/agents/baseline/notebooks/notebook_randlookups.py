# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Random exploration with credential lookup exploitation (notebook)

This notebooks can be run directly from VSCode, to generate a
traditional Jupyter Notebook to open in your browser
 you can run the VSCode command `Export Currenty Python File As Jupyter Notebook`.
"""

# pylint: disable=invalid-name

# %%
from cyberbattle._env.cyberbattle_env import AttackerGoal
from cyberbattle.agents.baseline.agent_randomcredlookup import CredentialCacheExploiter
import cyberbattle.agents.baseline.learner as learner
import gym
import logging
import sys
import cyberbattle.agents.baseline.plotting as p
import cyberbattle.agents.baseline.agent_wrapper as w
from cyberbattle.agents.baseline.agent_wrapper import Verbosity

# %%
logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")


# %%
cyberbattlechain_10 = gym.make('CyberBattleChain-v0', size=10,
                               attacker_goal=AttackerGoal(own_atleast_percent=1.0))


# %%
ep = w.EnvironmentBounds.of_identifiers(
    maximum_total_credentials=12,
    maximum_node_count=12,
    identifiers=cyberbattlechain_10.identifiers
)

iteration_count = 9000
training_episode_count = 50
eval_episode_count = 5

# %%
credexplot = learner.epsilon_greedy_search(
    cyberbattlechain_10,
    learner=CredentialCacheExploiter(),
    environment_properties=ep,
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.90,
    render=False,
    epsilon_multdecay=0.75,  # 0.999,
    epsilon_minimum=0.01,
    verbosity=Verbosity.Quiet,
    title="Random+CredLookup"
)

# %%
randomlearning_results = learner.epsilon_greedy_search(
    cyberbattlechain_10,
    environment_properties=ep,
    learner=CredentialCacheExploiter(),
    episode_count=eval_episode_count,
    iteration_count=iteration_count,
    epsilon=1.0,  # purely random
    render=False,
    verbosity=Verbosity.Quiet,
    title="Random search"
)


# %%
p.plot_episodes_length([credexplot])

p.plot_all_episodes(credexplot)

all_runs = [credexplot,
            randomlearning_results
            ]
p.plot_averaged_cummulative_rewards(
    title=f'Benchmark -- max_nodes={ep.maximum_node_count}, episodes={eval_episode_count},\n',
    all_runs=all_runs)

# %%
