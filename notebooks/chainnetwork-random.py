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
#     display_name: cybersim
#     language: python
#     name: cybersim
# ---

# %% [markdown]
# pyright: reportUnusedExpression=false

# %% [markdown]
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
#
# # Chain network CyberBattle Gym played by a random agent

# %% [markdown]
# # Gym random agent attacking a chain-like network
#
# ## Chain network
# We consider a computer network of Windows and Linux machines where each machine has vulnerability
# granting access to another machine as per the following pattern:
#
#     Start ---> (Linux ---> Windows --->  ... Linux ---> Windows)*  ---> Linux[Flag]
#
# The network is parameterized by the length of the central Linux-Windows chain.
# The start node leaks the credentials to connect to all other nodes:
#
# For each `XXX ---> Windows` section, the XXX node has:
#     -  a local vulnerability exposing the RDP password to the Windows machine
#     -  a bunch of other trap vulnerabilities (high cost with no outcome)
# For each `XXX ---> Linux` section,
#     - the Windows node has a local vulnerability exposing the SSH password to the Linux machine
#     - a bunch of other trap vulnerabilities (high cost with no outcome)
#
# The chain is terminated by one node with a flag (reward).

# %% [markdown]
# ## Benchmark
# The following plot shows the average and one standard deviation cumulative reward over time as a random agent attacks the network.

# %%
# %%HTML
# <img src="random_plot.png" width="300">

# %%
import sys
import logging
import gymnasium as gym
import cyberbattle.simulation.actions as actions
import cyberbattle._env.cyberbattle_env as cyberbattle_env
import cyberbattle.agents.random_agent as random_agent
import cyberbattle.samples.chainpattern.chainpattern as chainpattern
import importlib

importlib.reload(actions)
importlib.reload(cyberbattle_env)
importlib.reload(chainpattern)

logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")

# %%
# chainpattern.create_network_chain_link(2)

# %%
gym_env = gym.make("CyberBattleChain-v0", size=10, attacker_goal=None).unwrapped
assert isinstance(gym_env, cyberbattle_env.CyberBattleEnv)

# %%
gym_env.environment

# %%
gym_env.environment.network.nodes

# %%
gym_env.action_space

# %%
gym_env.action_space.sample()

# %%
gym_env.observation_space.sample()

# %%
for i in range(100):
    gym_env.sample_valid_action()

# %% tags=["parameters"]
iterations = 10000

# %%
random_agent.run_random_agent(1, iterations, gym_env)

# %%
o, r, d, _, i = gym_env.step(gym_env.sample_valid_action())

# %%
o

# %%
