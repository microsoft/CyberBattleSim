# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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
# pyright:  reportUnusedExpression=false

# %% [markdown]
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
#
# # Capture The Flag Toy Example - Solved manually

# %% [markdown]
# This notebook demonstrates how to model a toy `Capture The Flag` security game as a CyberBattle environment.

# %%
import sys, logging
import cyberbattle.simulation.model as model
import cyberbattle.simulation.commandcontrol as commandcontrol
import cyberbattle.samples.toyctf.toy_ctf as ctf

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s: %(message)s")

import plotly.offline as plo

plo.init_notebook_mode(connected=True) # type: ignore
# %matplotlib inline

# %%
network = model.create_network(ctf.nodes)
env = model.Environment(network=network, vulnerability_library=dict([]), identifiers=ctf.ENV_IDENTIFIERS)
env.plot_environment_graph()

# %% [markdown]
# ### Solution to the CTF
#
# This is the list of actions taken to capture 7 of the 8 flags from the CTF game.
#
# | Source      | Action | Result |
# |------------ | ------ | ------ |
# | WEBSITE     | page content has a link to github       | Discover Github project |
# | GITHUB      | navigate github history                 | **FLAG** Some secure access token (SAS) leaked in a reverted git commit (`CredScan`) |
# | AZURESTORAGE| access blob using SAS token             | |
# | WEBSITE     | view source HTML                        | Find URL to hidden .txt file on the website, extract directory path from it |
# |             | navigate to parent URL and find 3 files | **FLAG** Discover browseable web directory |
# |             | - readme.txt file                       | Discover secret data (the flag) |
# |             | - getting-started.txt                   | Discover MYSQL credentials |
# |             | - deprecation-checklist.txt             | Discover URL to external sharepoint website |
# | SHAREPOINT  | Navigate to sharepoint site             | **FLAG** Finding AD Service Principal Credentials on Sharepoint |
# | CLIENT-AZURE| `az resource` with creds from sharepoint| Obtain secrets hidden in azure managed resources |
# |             |                                         | Get AzureVM info, including public IP address |
# | CLIENT      | `ssh IP`                                | Failed attempt: internet incoming traffic blocked on the VM by NSG |
# | CLIENT      | SSH into WEBSITE with  mysql creds      | **FLAG** Shared credentials with database user|
# |             |                                         |**FLAG** Login using insecure SSH user/password|
# | WEBSITE/SSH | `history`                               |**FLAG**  Stealing credentials for the monitoring user|
# |             | `sudo -u monitor`                        | Failed! monitor not sudoable. message about being reported!
# | CLIENT      | SSH into WEBSITE with 'monitor creds     | Failed! password authentication disabled! looking for private key|
# | CLIENT      | SSH into WEBSITE as 'web'               | |
# |             | `su -u monitor` using password           |**FLAG**  User escalation by stealing credentials from bash history|
# |             | `cat ~/azurecreds.txt`                  | Get user credentials to Azure
# | CLIENT      | `az resource` with monitor's creds       | Steal more secrets
#

# %%
c2 = commandcontrol.CommandControl(env)
dbg = commandcontrol.EnvironmentDebugging(c2)

# 1 - Start from client
dbg.plot_discovered_network()

# %%
c2.print_all_attacks()

# %%
outcome = c2.run_attack("client", "SearchEdgeHistory")
dbg.plot_discovered_network()

# %%
c2.print_all_attacks()

# %%
# 2
github = c2.run_remote_attack("client", "Website", "ScanPageContent")
dbg.plot_discovered_network()

# %%
# 3
leakedSasUrl = c2.run_remote_attack("client", "GitHubProject", "CredScanGitHistory")
dbg.plot_discovered_network()

# %%
# 4
blobwithflag = c2.connect_and_infect("client", "AzureStorage", "HTTPS", "SASTOKEN1")
dbg.plot_discovered_network()
blobwithflag

# %%
# 5
browsableDirectory = c2.run_remote_attack("client", "Website", "ScanPageSource")
dbg.plot_discovered_network()

# %%
# 6
outcome_mysqlleak = c2.run_remote_attack("client", "Website.Directory", "NavigateWebDirectoryFurther")
sharepoint_url = c2.run_remote_attack("client", "Website.Directory", "NavigateWebDirectory")
dbg.plot_discovered_network()

# %%
# 7
outcome_azure_ad = c2.run_remote_attack("client", "Sharepoint", "ScanSharepointParentDirectory")
dbg.plot_discovered_network()

# %%
# 8
azureVmInfo = c2.connect_and_infect("client", "AzureResourceManager", "HTTPS", "ADPrincipalCreds")
dbg.plot_discovered_network()

# %%
c2.run_remote_attack("client", "AzureResourceManager", "ListAzureResources")
dbg.plot_discovered_network()

# %%
# 9 - CLIENT: Attempt to SSH into AzureVM from IP retrieved from Azure Resource Manager
should_fail = c2.connect_and_infect("client", "AzureVM", "SSH", "ReusedMySqlCred-web")
print("Success=" + str(should_fail))
dbg.plot_discovered_network()

# %%
# 10
owned = c2.connect_and_infect("client", "Website", "SSH", "ReusedMySqlCred-web")
dbg.plot_discovered_network()

# %%
# 11
outcome = c2.run_attack("Website", "CredScanBashHistory")
dbg.plot_discovered_network()

# %%
c2.print_all_attacks()

# %%
# 12
should_fail = c2.connect_and_infect("Website", "Website[user=monitor]", "sudo", "monitorBashCreds")
dbg.plot_discovered_network()

# %%
# 13
should_fail = c2.connect_and_infect("client", "Website[user=monitor]", "SSH", "monitorBashCreds")
dbg.plot_discovered_network()
should_fail

# %%
# 14
flag = c2.connect_and_infect("Website", "Website[user=monitor]", "su", "monitorBashCreds")
dbg.plot_discovered_network()

# %%
# 15
outcome = c2.run_attack("Website[user=monitor]", "CredScan-HomeDirectory")
dbg.plot_discovered_network()

# %%
# 16
secrets = c2.connect_and_infect("client", "AzureResourceManager[user=monitor]", "HTTPS", "azuread_user_credentials")
dbg.plot_discovered_network()

# %%
c2.print_all_attacks()
