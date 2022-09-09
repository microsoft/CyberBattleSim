# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Unit tests for commandcontrol.py.

"""
# pylint: disable=missing-function-docstring

from . import model, commandcontrol
from ..samples.toyctf import toy_ctf as ctf


def test_toyctf() -> None:
    # Use the C&C to exploit remote and local vulnerabilities in the toy CTF game
    network = model.create_network(ctf.nodes)
    env = model.Environment(network=network,
                            vulnerability_library=dict([]),
                            identifiers=ctf.ENV_IDENTIFIERS)
    command = commandcontrol.CommandControl(env)
    leak_website = command.run_attack('client', 'SearchEdgeHistory')
    assert leak_website
    github = command.run_remote_attack('client', 'Website', 'ScanPageContent')
    leaked_sas_url_outcome = command.run_remote_attack('client', 'GitHubProject', 'CredScanGitHistory')
    leaked_sas_url = commandcontrol.get_outcome_first_credential(leaked_sas_url_outcome)

    blobwithflag = command.connect_and_infect('client', 'AzureStorage', 'HTTPS', leaked_sas_url)
    assert (blobwithflag is not False)

    browsable_directory = command.run_remote_attack('client', 'Website', 'ScanPageSource')
    assert browsable_directory

    outcome_mysqlleak = command.run_remote_attack('client', 'Website.Directory', 'NavigateWebDirectoryFurther')
    mysql_credential = commandcontrol.get_outcome_first_credential(outcome_mysqlleak)
    sharepoint_url = command.run_remote_attack('client', 'Website.Directory', 'NavigateWebDirectory')
    assert sharepoint_url

    outcome_azure_ad = command.run_remote_attack('client', 'Sharepoint', 'ScanSharepointParentDirectory')
    azure_ad_credentials = commandcontrol.get_outcome_first_credential(outcome_azure_ad)

    azure_vm_info = command.connect_and_infect('client', 'AzureResourceManager', 'HTTPS', azure_ad_credentials)
    assert (azure_vm_info is not False)

    azure_resources = command.run_remote_attack('client', 'AzureResourceManager', 'ListAzureResources')
    assert azure_resources

    directly_ssh_connected = command.connect_and_infect('client', 'AzureVM', 'SSH', mysql_credential)
    assert not directly_ssh_connected

    sshd = command.connect_and_infect('client', 'Website', 'SSH', mysql_credential)
    assert sshd is not False

    outcome = command.run_attack('Website', 'CredScanBashHistory')
    monitor_bash_breds = commandcontrol.get_outcome_first_credential(outcome)

    connected_as_monitor = command.connect_and_infect('Website', 'Website[user=monitor]', 'sudo', monitor_bash_breds)
    assert not connected_as_monitor

    connected_as_monitor_from_client = command.connect_and_infect(
        'client', 'Website[user=monitor]', 'SSH', monitor_bash_breds)
    assert not connected_as_monitor_from_client

    flag = command.connect_and_infect('Website', 'Website[user=monitor]', 'su', monitor_bash_breds)
    assert flag is not False

    outcome_azuread = command.run_attack('Website[user=monitor]', 'CredScan-HomeDirectory')
    azure_ad_user_credential = commandcontrol.get_outcome_first_credential(outcome_azuread)

    secrets = command.connect_and_infect('client', 'AzureResourceManager', 'HTTPS',
                                         azure_ad_user_credential)
    assert secrets is not False

    reward = command.total_reward()
    print('Total reward ' + str(reward))
    assert reward == 389.0
    assert github is not None
    pass
