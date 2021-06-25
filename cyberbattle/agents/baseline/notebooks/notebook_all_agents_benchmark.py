# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Random agent with credential lookup (notebook)

This notebooks can be run directly from VSCode, to generate a
traditional Jupyter Notebook to open in your browser
 you can run the VSCode command `Export Currenty Python File As Jupyter Notebook`.
"""

# pylint: disable=invalid-name

# %% [markdown]
# # Chain network CyberBattle Gym played by a random agent with credential cache lookup

# %%
from cyberbattle._env import cyberbattle_env
import gym
import logging
import sys
import cyberbattle.agents.baseline.plotting as p
import cyberbattle.agents.baseline.agent_wrapper as w
from cyberbattle.agents.baseline.agent_wrapper import Verbosity
import cyberbattle.agents.baseline.learner as learner
import cyberbattle.agents.baseline.agent_randomcredlookup as rca
import cyberbattle.agents.baseline.agent_dql as dqla
import cyberbattle.agents.baseline.agent_tabularqlearning as tqa

import importlib
importlib.reload(tqa)
importlib.reload(dqla)
importlib.reload(learner)
importlib.reload(cyberbattle_env)


logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")


# %% [markdown]
# # Gym environment: chain-like network
# See Jupyer notebook `chainenetwork-random` for an introduction to this network environment.
cyberbattlechain_10 = gym.make(
    'CyberBattleChain-v0',
    size=10,
    attacker_goal=cyberbattle_env.AttackerGoal(own_atleast_percent=1.0)
)

cyberbattlechain_10.environment
# training_env.environment.plot_environment_graph()
cyberbattlechain_10.environment.network.nodes
cyberbattlechain_10.action_space
cyberbattlechain_10.action_space.sample()
cyberbattlechain_10.observation_space.sample()
o0 = cyberbattlechain_10.reset()
o_test, r, d, i = cyberbattlechain_10.step(cyberbattlechain_10.sample_valid_action())
o0 = cyberbattlechain_10.reset()

o0.keys()

# %%
ep = w.EnvironmentBounds.of_identifiers(
    maximum_node_count=22,
    maximum_total_credentials=22,
    identifiers=cyberbattlechain_10.identifiers
)

print(f"port_count = {ep.port_count}, property_count = {ep.property_count}")

fe_example = w.RavelEncoding(ep, [w.Feature_active_node_properties(ep), w.Feature_discovered_node_count(ep)])
a = w.StateAugmentation(o0)
w.Feature_discovered_ports(ep).get(a, None)
fe_example.encode_at(a, 0)


iteration_count = 9000
training_episode_count = 50
eval_episode_count = 5


# %%
random_run = learner.epsilon_greedy_search(
    cyberbattlechain_10,
    ep,
    learner=learner.RandomPolicy(),
    episode_count=10,  # training_episode_count,
    iteration_count=iteration_count,
    epsilon=1.0,
    render=False,
    verbosity=Verbosity.Quiet,
    title="Random"
)


# %%
credlookup_run = learner.epsilon_greedy_search(
    cyberbattlechain_10,
    ep,
    learner=rca.CredentialCacheExploiter(),
    episode_count=10,
    iteration_count=iteration_count,
    epsilon=0.90,
    render=False,
    epsilon_exponential_decay=10000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    title="Credential lookups (ϵ-greedy)"
)

# %%

tabularq_run = learner.epsilon_greedy_search(
    cyberbattlechain_10,
    ep,
    learner=tqa.QTabularLearner(
        ep,
        gamma=0.10, learning_rate=0.90, exploit_percentile=100),
    render=False,
    episode_count=training_episode_count,
    iteration_count=iteration_count,
    epsilon=0.90,
    epsilon_exponential_decay=10000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    title="Tabular Q-learning"
)

# %%
tabularq_exploit_run = learner.epsilon_greedy_search(
    cyberbattlechain_10,
    ep,
    learner=tqa.QTabularLearner(
        ep,
        trained=tabularq_run['learner'],
        gamma=0.0,
        learning_rate=0.0,
        exploit_percentile=90),
    episode_count=eval_episode_count,
    iteration_count=iteration_count,
    epsilon=0.0,
    render=False,
    verbosity=Verbosity.Quiet,
    title="Exploiting Q-matrix"
)

# %%
dql_run = learner.epsilon_greedy_search(
    cyberbattle_gym_env=cyberbattlechain_10,
    environment_properties=ep,
    learner=dqla.DeepQLearnerPolicy(
        ep=ep,
        gamma=0.015,
        replay_memory_size=10000,
        target_update=10,
        batch_size=512,
        learning_rate=0.01
    ),
    episode_count=15,
    iteration_count=iteration_count,
    epsilon=0.90,
    render=False,
    epsilon_exponential_decay=5000,
    epsilon_minimum=0.10,
    verbosity=Verbosity.Quiet,
    title="DQL"
)

# %%
dql_exploit_run = learner.epsilon_greedy_search(
    cyberbattlechain_10,
    ep,
    learner=dql_run['learner'],
    episode_count=50,
    iteration_count=iteration_count,
    epsilon=0.00,
    epsilon_minimum=0.00,
    render=False,
    verbosity=Verbosity.Quiet,
    title="Exploiting DQL"
)

# %%
all_runs = [
    random_run,
    credlookup_run,
    tabularq_run,
    tabularq_exploit_run,
    dql_run,
    dql_exploit_run
]

p.plot_episodes_length(all_runs)
p.plot_averaged_cummulative_rewards(
    title=f'Agent Benchmark\n'
    f'max_nodes:{ep.maximum_node_count}\n',
    all_runs=all_runs)

# %%
contenders = [
    credlookup_run,
    tabularq_run,
    dql_run,
    dql_exploit_run
]
p.plot_episodes_length(contenders)
p.plot_averaged_cummulative_rewards(
    title=f'Agent Benchmark top contenders\n'
    f'max_nodes:{ep.maximum_node_count}\n',
    all_runs=contenders)


# %%
for r in contenders:
    p.plot_all_episodes(r)

# %%
