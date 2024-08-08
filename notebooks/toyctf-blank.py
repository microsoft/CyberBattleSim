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

# %% [markdown]
# pyright:  reportUnusedExpression=false

# %% [markdown]
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
#
# # Capture the Flag Toy Example - Interactive (Human player)

# %% [markdown]
# This is a blank instantiaion of the Capture The Flag network to be played interactively by a human player (not via the gym envrionment).
# The interface exposed to the attacker is given by the following commands:
#     - c2.print_all_attacks()
#     - c2.run_attack(node, attack_id)
#     - c2.run_remote_attack(source_node, target_node, attack_id)
#     - c2.connect_and_infect(source_node, target_node, port_name, credential_id)

# %%
import sys, logging
import cyberbattle.simulation.model as model
import cyberbattle.simulation.commandcontrol as commandcontrol
import cyberbattle.samples.toyctf.toy_ctf as ctf
import plotly.offline as plo

plo.init_notebook_mode(connected=True) # type: ignore
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s: %(message)s")
# %matplotlib inline

# %%
network = model.create_network(ctf.nodes)
env = model.Environment(network=network, vulnerability_library=dict([]), identifiers=ctf.ENV_IDENTIFIERS)
env

# %%
env.plot_environment_graph()

# %%
c2 = commandcontrol.CommandControl(env)
dbg = commandcontrol.EnvironmentDebugging(c2)


def plot():
    dbg.plot_discovered_network()
    c2.print_all_attacks()


plot()
