# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Command and Control interface
# This notebooks shows how to interact with the command&control server to observe the environment and initiate actions on the nodes where the attacker client is installed.

# %%
import networkx as nx
from tabulate import tabulate
import cyberbattle.simulation.model as model
import cyberbattle.simulation.actions as actions
import cyberbattle.simulation.commandcontrol as commandcontrol
import importlib

importlib.reload(model)
importlib.reload(actions)
importlib.reload(commandcontrol)
import plotly.offline as plo

plo.init_notebook_mode(connected=True)  # type: ignore
# %matplotlib inline

# %% [markdown]
# We first create a simulation environment from a randomly generated network graph.

# %%
g = nx.erdos_renyi_graph(35, 0.05, directed=True)
g = model.assign_random_labels(g)
env = model.Environment(network=g, vulnerability_library=dict([]), identifiers=model.SAMPLE_IDENTIFIERS)

# %% [markdown]
# We create the `CommandControl` object used to the environment and execute actions, and plot the graph explored so far.
#

# %%
c = commandcontrol.CommandControl(env)

# %%
c.plot_nodes()
print("Nodes disovered so far: " + str(c.list_nodes()))
starting_node = c.list_nodes()[0]["id"]

# %% [markdown]
# For debugging purpose it's also convient to view the internals of the environment via the `EnvironmentDebugging` object. For instance we can use it to plot the entire graph, including nodes that were not discovered yet by the attacker.

# %%
dbg = commandcontrol.EnvironmentDebugging(c)

# %%
env.plot_environment_graph()
print(nx.info(env.network))  # type: ignore

# %%
print(tabulate(c.list_all_attacks(), {}))

# %%
outcome = c.run_attack(starting_node, "RecentlyAccessedMachines")
outcome

# %%
c.plot_nodes()

# %%
print(tabulate(c.list_nodes(), {}))

# %%
print(tabulate(c.list_all_attacks(), {}))
