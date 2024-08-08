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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
#
# # Randomly generated CyberBattle network environment for Active Directory

# %%
import cyberbattle.samples.active_directory.generate_ad as ad
import cyberbattle.simulation.commandcontrol as commandcontrol
import logging, sys, random

random.seed(1)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s: %(message)s")
# %matplotlib inline

# %%
env = ad.new_random_environment(42)
env.plot_environment_graph()

# %%
c2 = commandcontrol.CommandControl(env)
dbg = commandcontrol.EnvironmentDebugging(c2)

# 1 - Start from client
dbg.plot_discovered_network()
c2.print_all_attacks()

# %%
outcome = c2.run_attack("workstation_0", "FindDomainControllers")
dbg.plot_discovered_network()
c2.print_all_attacks()

# %%
outcome = c2.run_attack("workstation_0", "EnumerateFileShares")
dbg.plot_discovered_network()
c2.print_all_attacks()

# %%
outcome = c2.run_attack("workstation_0", "AuthorizationSpoofAndCrack")
dbg.plot_discovered_network()
c2.print_all_attacks()

# %%
c2.connect_and_infect("workstation_0", "workstation_4", "SHELL", "user_28")
dbg.plot_discovered_network()
c2.print_all_attacks()

# %%
c2.run_attack("workstation_4", "ScanForCreds")
dbg.plot_discovered_network()
c2.print_all_attacks()

# %%
c2.connect_and_infect("workstation_0", "domain_controller_1", "AD", "dc_1")
dbg.plot_discovered_network()
c2.print_all_attacks()

# %%
c2.run_attack("domain_controller_1", "DumpNTDS")
dbg.plot_discovered_network()
c2.print_all_attacks()
