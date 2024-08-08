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
# # Randomly generated CyberBattle network environment (experimental)

# %%
import numpy as np
import cyberbattle.simulation.model as model
import cyberbattle.agents.random_agent as random_agent
from typing import List
import cyberbattle.simulation.generate_network as g

# %matplotlib inline

# %%
traffic = g.generate_random_traffic_network(
    seed=1,
    n_clients=50,
    n_servers={
        "SMB": 15,
        "HTTP": 15,
        "RDP": 15,
    },
    alpha=np.array([(1, 1), (0.2, 0.5)], dtype=float),
    beta=np.array([(1000, 10), (10, 100)], dtype=float),
)

# %%
network = g.cyberbattle_model_from_traffic_graph(traffic)

# %%
env = model.Environment(network=network, vulnerability_library=dict([]), identifiers=g.ENV_IDENTIFIERS)
env.plot_environment_graph()

# %%
network.nodes


# %%
def ports_from_vuln(vuln: model.VulnerabilityInfo) -> List[model.PortName]:
    if isinstance(vuln.outcome, model.LeakedCredentials):
        return [c.port for c in vuln.outcome.credentials]
    else:
        return []


# %%
all_existing_ports = (
    set({port for _, v in env.vulnerability_library.items() for port in ports_from_vuln(v)})
    .union({port for _, node_info in env.nodes() for _, v in node_info.vulnerabilities.items() for port in ports_from_vuln(v)})
    .union({service.name for _, node_info in env.nodes() for service in node_info.services})
)

# %%
all_existing_ports

# %%
[node_info.services for _, node_info in env.nodes()]

# %%
# import sys, logging
# logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s: %(message)s")
import gymnasium as gym
import cyberbattle._env.cyberbattle_env as cyberbattle_env

# %%
gym_env = gym.make("CyberBattleRandom-v0").unwrapped

# %%
assert isinstance(gym_env, cyberbattle_env.CyberBattleEnv)

random_agent.run_random_agent(1, 5600, gym_env)

# %%
