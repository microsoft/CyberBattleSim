# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Defines a set of networks following a speficic pattern
learnable from the properties associated with the nodes.

The network pattern is:
         Start ---> (Linux ---> Windows --->  ... Linux ---> Windows)*  ---> Linux[Flag]

The network is parameterized by the length of the central Linux-Windows chain.
The start node leaks the credentials to connect to all other nodes:

For each `XXX ---> Windows` section, the XXX node has:
    -  a local vulnerability exposing the RDP password to the Windows machine
    -  a bunch of other trap vulnerabilities (high cost with no outcome)
For each `XXX ---> Linux` section,
    - the Windows node has a local vulnerability exposing the SSH password to the Linux machine
    - a bunch of other trap vulnerabilities (high cost with no outcome)

The chain is terminated by one node with a flag (reward).

A Node-Property matrix would be three-valued (0,1,?) and look like this:

===== Initial state
        Properties
Nodes   L  W  SQL
1       1  0  0
2       ?  ?  ?
3       ?  ?  ?
...
10
======= After discovering node 2
        Properties
Nodes   L  W  SQL
1       1  0  0
2       0  1  1
3       ?  ?  ?
...
10
===========================

"""
from cyberbattle.simulation.model import Identifiers, NodeID, NodeInfo
from ...simulation import model as m
from typing import Dict

DEFAULT_ALLOW_RULES = [
    m.FirewallRule("RDP", m.RulePermission.ALLOW),
    m.FirewallRule("SSH", m.RulePermission.ALLOW),
    m.FirewallRule("HTTPS", m.RulePermission.ALLOW),
    m.FirewallRule("HTTP", m.RulePermission.ALLOW)]

# Environment constants used for all instances of the chain network
ENV_IDENTIFIERS = Identifiers(
    properties=[
        'Windows',
        'Linux',
        'ApacheWebSite',
        'IIS_2019',
        'IIS_2020_patched',
        'MySql',
        'Ubuntu',
        'nginx/1.10.3',
        'SMB_vuln',
        'SMB_vuln_patched',
        'SQLServer',
        'Win10',
        'Win10Patched',
        'FLAG:Linux'
    ],
    ports=[
        'HTTPS',
        'GIT',
        'SSH',
        'RDP',
        'PING',
        'MySQL',
        'SSH-key',
        'su'
    ],
    local_vulnerabilities=[
        'ScanBashHistory',
        'ScanExplorerRecentFiles',
        'SudoAttempt',
        'CrackKeepPassX',
        'CrackKeepPass'
    ],
    remote_vulnerabilities=[
        'ProbeLinux',
        'ProbeWindows'
    ]
)


def prefix(x: int, name: str):
    """Prefix node name with an instance"""
    return f"{x}_{name}"


def rdp_password(index):
    """Generate RDP password for the specified chain link"""
    return f"WindowsPassword!{index}"


def ssh_password(index):
    """Generate SSH password for the specified chain link"""
    return f"LinuxPassword!{index}"


def create_network_chain_link(n: int) -> Dict[NodeID, NodeInfo]:
    """Instantiate one link of the network chain with associated index n"""

    def current(name):
        return prefix(n, name)

    def next(name):
        return prefix(n + 1, name)

    return {
        current("LinuxNode"): m.NodeInfo(
            services=[m.ListeningService("HTTPS"),
                      m.ListeningService("SSH", allowedCredentials=[ssh_password(n)])],
            firewall=m.FirewallConfiguration(incoming=DEFAULT_ALLOW_RULES,
                                             outgoing=DEFAULT_ALLOW_RULES),
            value=100,
            properties=["MySql", "Ubuntu", "nginx/1.10.3"],
            owned_string="Intermediate chain node owned, no intinsic value",
            vulnerabilities=dict(
                ProbeLinux=m.VulnerabilityInfo(
                    description="Probe to check if the node runs Linux",
                    type=m.VulnerabilityType.REMOTE,
                    outcome=m.ProbeSucceeded(["Ubuntu"]),
                    reward_string="Remote machine is running Linux",
                    cost=5.0
                ),
                ProbeWindows=m.VulnerabilityInfo(
                    description="Probe to check if the node runs Windows",
                    type=m.VulnerabilityType.REMOTE,
                    outcome=m.ProbeFailed(),
                    reward_string="Remote machine is not running Windows",
                    cost=5.0
                ),
                ScanBashHistory=m.VulnerabilityInfo(
                    description="Scan bash history for possible references to other machines",
                    type=m.VulnerabilityType.LOCAL,
                    outcome=m.LeakedNodesId([next("WindowsNode")]),
                    reward_string="Found a reference to a remote Windows node in bash history",
                    cost=1.0
                ),
                ScanExplorerRecentFiles=m.VulnerabilityInfo(
                    description="Scan Windows Explorer recent files for possible references to other machines",
                    type=m.VulnerabilityType.LOCAL,
                    outcome=m.ExploitFailed(),
                    reward_string="Trap: feature not supported on Linux",
                    cost=10.0
                ),
                SudoAttempt=m.VulnerabilityInfo(
                    description="Attempt to sudo into admin user",
                    type=m.VulnerabilityType.LOCAL,
                    outcome=m.ExploitFailed(),
                    reward_string="Trap: suspicious attempt to run sudo",
                    cost=100.0
                ),
                CrackKeepPassX=m.VulnerabilityInfo(
                    description="Attempt to crack KeepPassX and look for credentials",
                    type=m.VulnerabilityType.LOCAL,
                    outcome=m.LeakedCredentials(credentials=[
                        m.CachedCredential(node=next("WindowsNode"), port="RDP",
                                           credential=rdp_password(n + 1))]),
                    reward_string=f"Discovered password to Windows machine {n+1}",
                    cost=1.0
                ))),

        next("WindowsNode"): m.NodeInfo(
            services=[m.ListeningService("HTTPS"),
                      m.ListeningService("RDP", allowedCredentials=[rdp_password(n + 1)])],
            value=100,
            properties=["Windows", "Win10", "Win10Patched"],
            vulnerabilities=dict(
                ProbeLinux=m.VulnerabilityInfo(
                    description="Probe to check if the node runs Linux",
                    type=m.VulnerabilityType.REMOTE,
                    outcome=m.ProbeFailed(),
                    reward_string="Remote machine is not running Linux",
                    cost=1.0
                ),
                ProbeWindows=m.VulnerabilityInfo(
                    description="Probe to check if the node runs Windows",
                    type=m.VulnerabilityType.REMOTE,
                    outcome=m.ProbeSucceeded(["Windows"]),
                    reward_string="Remote machine is running Windows",
                    cost=1.0
                ),
                ScanBashHistory=m.VulnerabilityInfo(
                    description="Scan bash history for possible references to other machines",
                    type=m.VulnerabilityType.LOCAL,
                    outcome=m.ExploitFailed(),
                    reward_string="Trap: feature not supported on Windows!",
                    cost=100.0
                ),
                ScanExplorerRecentFiles=m.VulnerabilityInfo(
                    description="Scan Windows Explorer recent files for possible references to other machines",
                    type=m.VulnerabilityType.LOCAL,
                    outcome=m.LeakedNodesId([prefix(n + 2, "LinuxNode")]),
                    reward_string="Found a reference to a remote Linux node in bash history",
                    cost=1.0
                ),
                SudoAttempt=m.VulnerabilityInfo(
                    description="Attempt to sudo into admin user",
                    type=m.VulnerabilityType.LOCAL,
                    outcome=m.ExploitFailed(),
                    reward_string="Trap: feature not supported on Windows!",
                    cost=100.0
                ),
                CrackKeepPassX=m.VulnerabilityInfo(
                    description="Attempt to crack KeepPassX and look for credentials",
                    type=m.VulnerabilityType.LOCAL,
                    outcome=m.ExploitFailed(),
                    reward_string="Trap: feature not supported on Windows!",
                    cost=100.0
                ),
                CrackKeepPass=m.VulnerabilityInfo(
                    description="Attempt to crack KeepPass and look for credentials",
                    type=m.VulnerabilityType.LOCAL,
                    outcome=m.LeakedCredentials(credentials=[
                        m.CachedCredential(node=prefix(n + 2, "LinuxNode"), port="SSH",
                                           credential=ssh_password(n + 2))]),
                    reward_string=f"Discovered password to Linux machine {n+2}",
                    cost=1.0
                )
            ))
    }


def create_chain_network(size: int) -> Dict[NodeID, NodeInfo]:
    """Create a chain network with the chain section of specified size.
    Size must be an even number
    The number of nodes in the network is `size + 2` to account for the start node (0)
    and final node (size + 1).
    """

    if size % 2 == 1:
        raise ValueError(f"Chain size must be even: {size}")

    final_node_index = size + 1

    nodes = {
        'start': m.NodeInfo(
            services=[],
            value=0,
            vulnerabilities=dict(
                ScanExplorerRecentFiles=m.VulnerabilityInfo(
                    description="Scan Windows Explorer recent files for possible references to other machines",
                    type=m.VulnerabilityType.LOCAL,
                    outcome=m.LeakedCredentials(credentials=[
                        m.CachedCredential(node=prefix(1, "LinuxNode"), port="SSH",
                                           credential=ssh_password(1))]),
                    reward_string="Found a reference to a remote Linux node in bash history",
                    cost=1.0
                )),
            agent_installed=True,
            reimagable=False),

        prefix(final_node_index, "LinuxNode"): m.NodeInfo(
            services=[m.ListeningService("HTTPS"),
                      m.ListeningService("SSH", allowedCredentials=[ssh_password(final_node_index)])],
            value=1000,
            owned_string="FLAG: flag discovered!",
            properties=["MySql", "Ubuntu", "nginx/1.10.3", "FLAG:Linux"],
            vulnerabilities=dict()
        )
    }

    # Add chain links
    for i in range(1, size, 2):
        nodes.update(create_network_chain_link(i))

    return nodes


def new_environment(size) -> m.Environment:
    return m.Environment(
        network=m.create_network(create_chain_network(size)),
        vulnerability_library=dict([]),
        identifiers=ENV_IDENTIFIERS
    )
