# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
    This is the set of tests for actions.py which implements the actions an agent can take
    in this simulation.
"""
import random
from datetime import datetime
from typing import Dict, List

import pytest
import networkx as nx

from . import model, actions

ADMINTAG = model.AdminEscalation().tag
SYSTEMTAG = model.SystemEscalation().tag

# pylint: disable=redefined-outer-name, protected-access
Fixture = actions.AgentActions

empty_vuln_dict: Dict[model.VulnerabilityID, model.VulnerabilityInfo] = {}
SINGLE_VULNERABILITIES = {
    "UACME61":
    model.VulnerabilityInfo(
        description="UACME UAC bypass #61",
        type=model.VulnerabilityType.LOCAL,
        URL="https://github.com/hfiref0x/UACME",
        precondition=model.Precondition(f"Windows&Win10&(~({ADMINTAG}|{SYSTEMTAG}))"),
        outcome=model.AdminEscalation(),
        rates=model.Rates(0, 0.2, 1.0))}

# temporary vuln dictionary for development purposes only.
# Remove once the full list of vulnerabilities is put together
# here we'll have 1 UAC bypass, 1 credential dump, and 1 remote infection vulnerability
SAMPLE_VULNERABILITIES = {
    "UACME61":
    model.VulnerabilityInfo(
        description="UACME UAC bypass #61",
        type=model.VulnerabilityType.LOCAL,
        URL="https://github.com/hfiref0x/UACME",
        precondition=model.Precondition(f"Windows&Win10&(~({ADMINTAG}|{SYSTEMTAG}))"),
        outcome=model.AdminEscalation(),
        rates=model.Rates(0, 0.2, 1.0)),
    "UACME67":
    model.VulnerabilityInfo(
        description="UACME UAC bypass #67 (fake system escalation) ",
        type=model.VulnerabilityType.LOCAL,
        URL="https://github.com/hfiref0x/UACME",
        precondition=model.Precondition(f"Windows&Win10&(~({ADMINTAG}|{SYSTEMTAG}))"),
        outcome=model.SystemEscalation(),
        rates=model.Rates(0, 0.2, 1.0)),
    "MimikatzLogonpasswords":
    model.VulnerabilityInfo(
        description="Mimikatz sekurlsa::logonpasswords.",
        type=model.VulnerabilityType.LOCAL,
        URL="https://github.com/gentilkiwi/mimikatz",
        precondition=model.Precondition(f"Windows&({ADMINTAG}|{SYSTEMTAG})"),
        outcome=model.LeakedCredentials([]),
        rates=model.Rates(0, 1.0, 1.0)),
    "RDPBF":
    model.VulnerabilityInfo(
        description="RDP Brute Force",
        type=model.VulnerabilityType.REMOTE,
        URL="https://attack.mitre.org/techniques/T1110/",
        precondition=model.Precondition("Windows&PortRDPOpen"),
        outcome=model.LateralMove(),
        rates=model.Rates(0, 0.2, 1.0),
        cost=1.0)
}

ENV_IDENTIFIERS = model.Identifiers(
    local_vulnerabilities=['UACME61', 'UACME67', 'MimikatzLogonpasswords', 'UACME61'],
    remote_vulnerabilities=['RDPBF'],
    ports=['RDP', 'HTTP', 'HTTPS', 'SSH'],
    properties=[
        "Linux", "PortSSHOpen", "PortSQLOpen",
        "Windows", "Win10", "PortRDPOpen",
        "PortHTTPOpen", "PortHTTPsOpen",
        "SharepointLeakingPassword"]
)


def sample_random_firwall_configuration() -> model.FirewallConfiguration:
    """Sample a random firewall set of rules"""
    return model.FirewallConfiguration(
        outgoing=[model.FirewallRule(p, permission=model.RulePermission.ALLOW)
                  for p in random.choices(ENV_IDENTIFIERS.properties,
                                          k=random.randint(0, len(ENV_IDENTIFIERS.properties)))],
        incoming=[model.FirewallRule(p, permission=model.RulePermission.ALLOW)
                  for p in random.choices(ENV_IDENTIFIERS.properties,
                                          k=random.randint(0, len(ENV_IDENTIFIERS.properties)))])


# temporary info for a single node network
SINGLE_NODE = {
    'a': model.NodeInfo(
        services=[model.ListeningService("RDP"),
                  model.ListeningService("HTTP"),
                  model.ListeningService("HTTPS")],
        value=70,
        properties=list(["Windows", "Win10", "PortRDPOpen", "PortHTTPOpen", "PortHTTPsOpen"]),
        firewall=sample_random_firwall_configuration(),
        agent_installed=False)}

# temporary info for 4 nodes
# a is a windows web server, b is linux SQL server, c is a windows workstation,
# and dc is a domain controller
NODES = {
    'a': model.NodeInfo(
        services=[model.ListeningService("RDP"),
                  model.ListeningService("HTTP"),
                  model.ListeningService("HTTPS")],
        value=70,
        properties=list(["Windows", "Win10", "PortRDPOpen", "PortHTTPOpen", "PortHTTPsOpen"]),
        vulnerabilities=dict(
            ListNeighbors=model.VulnerabilityInfo(
                description="reveal other nodes",
                type=model.VulnerabilityType.LOCAL,
                outcome=model.LeakedNodesId(nodes=['b', 'c', 'dc'])),

            DumpCreds=model.VulnerabilityInfo(
                description="leaking some creds",
                type=model.VulnerabilityType.LOCAL,
                outcome=model.LeakedCredentials([model.CachedCredential('Sharepoint', "HTTPS", "ADPrincipalCreds"),
                                                 model.CachedCredential('Sharepoint', "HTTPS", "cred")])
            )
        ),
        agent_installed=True),
    'b': model.NodeInfo(
        services=[model.ListeningService("SSH"),
                  model.ListeningService("SQL")],
        value=80,
        properties=list(["Linux", "PortSSHOpen", "PortSQLOpen"]),
        agent_installed=False),
    'c': model.NodeInfo(
        services=[model.ListeningService("RDP"),
                  model.ListeningService("HTTP"),
                  model.ListeningService("HTTPS")],
        value=40,
        properties=list(["Windows", "Win10", "PortRDPOpen", "PortHTTPOpen", "PortHTTPsOpen"]),
        agent_installed=True),
    'dc': model.NodeInfo(
        services=[model.ListeningService("RDP"),
                  model.ListeningService("WMI")],
        value=100, properties=list(["Windows", "Win10", "PortRDPOpen", "PortWMIOpen"]),
        agent_installed=False),
    'Sharepoint': model.NodeInfo(
        services=[model.ListeningService("HTTPS", allowedCredentials=["ADPrincipalCreds"])], value=100,
        properties=["SharepointLeakingPassword"],
        firewall=model.FirewallConfiguration(
            incoming=[model.FirewallRule(port="SSH", permission=model.RulePermission.ALLOW),
                      model.FirewallRule(port="HTTPS", permission=model.RulePermission.ALLOW),
                      model.FirewallRule(port="HTTP", permission=model.RulePermission.ALLOW),
                      model.FirewallRule(port="RDP", permission=model.RulePermission.BLOCK)
                      ],
            outgoing=[]),
        vulnerabilities=dict(
            ScanSharepointParentDirectory=model.VulnerabilityInfo(
                description="Navigate to SharePoint site, browse parent "
                            "directory",
                type=model.VulnerabilityType.REMOTE,
                outcome=model.LeakedCredentials(credentials=[
                    model.CachedCredential(node="AzureResourceManager",
                                           port="HTTPS",
                                           credential="ADPrincipalCreds")]),
                rates=model.Rates(successRate=1.0),
                cost=1.0)
        )),

}


# Define an environment from this graph
ENV = model.Environment(
    network=model.create_network(NODES),
    vulnerability_library=dict([]),
    identifiers=ENV_IDENTIFIERS,
    creationTime=datetime.utcnow(),
    lastModified=datetime.utcnow(),
)


@ pytest.fixture
def actions_on_empty_environment() -> actions.AgentActions:
    """
        the test fixtures to reduce the amount of overhead
        This fixture will provide us with an empty environment.
    """
    egraph = nx.empty_graph(0, create_using=nx.DiGraph())
    env = model.Environment(network=egraph,
                            version=model.VERSION_TAG,
                            vulnerability_library=SAMPLE_VULNERABILITIES,
                            identifiers=ENV_IDENTIFIERS,
                            creationTime=datetime.utcnow(),
                            lastModified=datetime.utcnow())
    return actions.AgentActions(env)


@ pytest.fixture
def actions_on_single_node_environment() -> actions.AgentActions:
    """
        This fixture will provide us with a single node environment
    """
    env = model.Environment(network=model.create_network(SINGLE_NODE),
                            version=model.VERSION_TAG,
                            vulnerability_library=SAMPLE_VULNERABILITIES,
                            identifiers=ENV_IDENTIFIERS,
                            creationTime=datetime.utcnow(),
                            lastModified=datetime.utcnow())
    return actions.AgentActions(env)


@ pytest.fixture
def actions_on_simple_environment() -> actions.AgentActions:
    """
     This fixture will provide us with a 4 node environment environment.
     simulating three workstations connected to a single server
    """
    env = model.Environment(network=model.create_network(NODES),
                            version=model.VERSION_TAG,
                            vulnerability_library=SAMPLE_VULNERABILITIES,
                            identifiers=ENV_IDENTIFIERS,
                            creationTime=datetime.utcnow(),
                            lastModified=datetime.utcnow())
    return actions.AgentActions(env)


def test_list_vulnerabilities_function(actions_on_single_node_environment: Fixture,
                                       actions_on_simple_environment: Fixture) -> None:
    """
        This function will test the list_vulnerabilities function from the
        AgentActions class in actions.py
    """
    # test on an environment with a single node
    single_node_results: List[model.VulnerabilityID] = []
    single_node_results = actions_on_single_node_environment.list_vulnerabilities_in_target('a')
    assert len(single_node_results) == 3

    simple_graph_results: List[model.VulnerabilityID] = []
    simple_graph_results = actions_on_simple_environment.list_vulnerabilities_in_target('dc')
    assert len(simple_graph_results) == 3


def test_exploit_remote_vulnerability(actions_on_simple_environment: Fixture) -> None:
    """
        This function will test the exploit_remote_vulnerability function from the
        AgentActions class in actions.py
    """

    actions_on_simple_environment.exploit_local_vulnerability('a', "ListNeighbors")

    # test with invalid source node
    with pytest.raises(ValueError, match=r"invalid node id '.*'"):
        actions_on_simple_environment.exploit_remote_vulnerability('z', 'b', "RDPBF")

    # test with invalid destination node
    with pytest.raises(ValueError, match=r"invalid target node id '.*'"):
        actions_on_simple_environment.exploit_remote_vulnerability('a', 'z', "RDPBF")

    # test with a local vulnerability
    with pytest.raises(ValueError, match=r"vulnerability id '.*' is for an attack of type .*"):
        actions_on_simple_environment.exploit_remote_vulnerability('a', 'c', "MimikatzLogonpasswords")

    # test with an invalid vulnerability (one not there)
    result = actions_on_simple_environment.exploit_remote_vulnerability('a', 'c', "HackTheGibson")
    assert result.outcome is None and result.reward <= 0

    # add RDP brute force to the target node
    # very hacky not to be used normally.
    graph: nx.graph.Graph = actions_on_simple_environment._environment.network
    node: model.NodeInfo = graph.nodes['c']['data']
    node.vulnerabilities = SAMPLE_VULNERABILITIES

    # test a valid and functional one.
    result = actions_on_simple_environment.exploit_remote_vulnerability('a', 'c', "RDPBF")
    assert isinstance(result.outcome, model.LateralMove)
    assert result.reward < node.value


def test_exploit_local_vulnerability(actions_on_simple_environment: Fixture) -> None:
    """
        This function will test the exploit_local_vulnerability function from the
        AgentActions class in actions.py
    """

    # check one with invalid prerequisites
    result: actions.ActionResult = actions_on_simple_environment.\
        exploit_local_vulnerability('a', "MimikatzLogonpasswords")
    assert isinstance(result.outcome, model.ExploitFailed)

    # test admin privilege escalation
    # exploit_local_vulnerability(node_id, vulnerability_id)
    result = actions_on_simple_environment.exploit_local_vulnerability('a', "UACME61")
    assert isinstance(result.outcome, model.AdminEscalation)
    node: model.NodeInfo = actions_on_simple_environment._environment.network.nodes['a']['data']
    assert model.AdminEscalation().tag in node.properties

    # test system privilege escalation
    result = actions_on_simple_environment.exploit_local_vulnerability('c', "UACME67")
    assert isinstance(result.outcome, model.SystemEscalation)
    node = actions_on_simple_environment._environment.network.nodes['c']['data']
    assert model.SystemEscalation().tag in node.properties

    # test dump credentials
    result = actions_on_simple_environment.\
        exploit_local_vulnerability('a', "MimikatzLogonpasswords")
    assert isinstance(result.outcome, model.LeakedCredentials)


def test_connect_to_remote_machine(actions_on_empty_environment: Fixture,
                                   actions_on_single_node_environment: Fixture,
                                   actions_on_simple_environment: Fixture) -> None:
    """
        This function will test the connect_to_remote_machine function from the
        AgentActions class in actions.py
    """
    actions_on_simple_environment.exploit_local_vulnerability('a', "ListNeighbors")
    actions_on_simple_environment.exploit_local_vulnerability('a', "DumpCreds")

    # test connect to remote machine on an empty environment
    with pytest.raises(ValueError, match=r"invalid node id '.*'"):
        actions_on_empty_environment.connect_to_remote_machine("a", "b", "RDP", "cred")

    # test connect to remote machine on an environment with 1 node
    with pytest.raises(ValueError, match=r"invalid node id '.*'"):
        actions_on_single_node_environment.connect_to_remote_machine("a", "b", "RDP", "cred")

    graph: nx.graph.Graph = actions_on_simple_environment._environment.network

    # test connect to remote machine on an environment with multiple nodes
    # test with valid source node and invalid destination node
    with pytest.raises(ValueError, match=r"invalid node id '.*'"):
        actions_on_simple_environment.\
            connect_to_remote_machine("a", "f", "RDP", "cred")

    # test with an invalid source node and valid destination node
    with pytest.raises(ValueError, match=r"invalid node id '.*'"):
        actions_on_simple_environment.connect_to_remote_machine("f", "dc", "RDP", "cred")

    # test with both nodes invalid
    with pytest.raises(ValueError, match=r"invalid node id '.*'"):
        actions_on_simple_environment.connect_to_remote_machine("f", "z", "RDP", "cred")

    # test with invalid protocol
    result = actions_on_simple_environment.connect_to_remote_machine("a", "dc", "TCPIP", "cred")
    assert result.reward <= 0 and result.outcome is None

    # test with invalid credentials
    result2 = actions_on_simple_environment.connect_to_remote_machine("a", "dc", "RDP", "cred")
    assert result2.outcome is None and result2.reward <= 0

    # test blocking firewall rule
    ret_val = actions_on_simple_environment.connect_to_remote_machine("a", 'Sharepoint', "RDP", "ADPrincipalCreds")
    assert ret_val.reward < 0

    # test with valid nodes
    ret_val = actions_on_simple_environment.connect_to_remote_machine("a", 'Sharepoint', "HTTPS", "ADPrincipalCreds")

    assert ret_val.reward == 100

    assert graph.has_edge("a", "dc")


def test_check_prerequisites(actions_on_simple_environment: Fixture) -> None:
    """
        This function will test the _checkPrerequisites function
        It's marked as a private function but still needs to be tested before use

    """
    # testing on a node/vuln combo  which should give us a negative result
    result = actions_on_simple_environment._check_prerequisites('dc', SAMPLE_VULNERABILITIES["MimikatzLogonpasswords"])
    assert not result

    # testing on a node/vuln combo which should give us a positive reuslt.
    result = actions_on_simple_environment._check_prerequisites('dc', SAMPLE_VULNERABILITIES["UACME61"])
    assert result
