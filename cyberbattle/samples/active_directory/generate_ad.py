""" Generating random active directory networks"""
import random
from typing import Any
from cyberbattle.simulation.model import FirewallConfiguration, FirewallRule, Identifiers, RulePermission
from cyberbattle.simulation import model as m
import networkx as nx

ENV_IDENTIFIERS = Identifiers(
    properties=[
        'breach_node',
        'domain_controller',
        "admin"  # whether or not the users of this machine are admins
    ],
    ports=['SMB', 'AD', 'SHELL'],
    local_vulnerabilities=[
        'FindDomainControllers',
        'EnumerateFileShares',
        'AuthorizationSpoofAndCrack',
        'ScanForCreds',
        'DumpNTDS',
        'ProbeAdmin'
    ],
    remote_vulnerabilities=[
        'PasswordSpray'
    ]
)


def create_network_from_smb_traffic(
    n_clients: int,
    n_servers: int,
    n_users: int
) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from([f"workstation_{i}" for i in range(0, n_clients)])
    graph.add_nodes_from([f"share_{i}" for i in range(0, n_servers)])
    graph.add_node("domain_controller_1")
    # NOTE: The following is not needed. The AgentActions code will annotate our network based on dynamic discovery
    # graph.add_edges_from([(f"share_{i}", f"workstation_{j}") for i in range(0, n_servers) for j in range(0, n_clients)], protocol="SMB")
    # graph.add_edges_from([("domain_controller_1", f"workstation_{i}") for i in range(0, n_clients)], protocol="AD")
    # graph.add_edges_from([("domain_controller_1", f"share_{i}") for i in range(0, n_servers)], protocol="AD")

    firewall_conf = FirewallConfiguration(
        [FirewallRule("SMB", RulePermission.ALLOW), FirewallRule("AD", RulePermission.ALLOW), FirewallRule("SHELL", RulePermission.ALLOW)],
        [FirewallRule("SMB", RulePermission.ALLOW), FirewallRule("AD", RulePermission.ALLOW), FirewallRule("SHELL", RulePermission.ALLOW)])

    def default_vulnerabilities() -> m.VulnerabilityLibrary:
        lib = {}
        lib['FindDomainControllers'] = m.VulnerabilityInfo(
            description="Search for valid domain controllers in the current machines environment.",
            type=m.VulnerabilityType.LOCAL,
            outcome=m.LeakedNodesId(nodes=["domain_controller_1"]),
            reward_string="Found domain controllers"
        )
        lib['EnumerateFileShares'] = m.VulnerabilityInfo(
            description="Find all SMB shares this machine knows about.",
            type=m.VulnerabilityType.LOCAL,
            outcome=m.LeakedNodesId(nodes=[f"share_{i}" for i in range(0, n_servers)]),
            reward_string="Found shares"
        )
        lib["ProbeAdmin"] = m.VulnerabilityInfo(
            description="Probe a workstation to see if you have admin creds on it",
            type=m.VulnerabilityType.LOCAL,
            outcome=m.ProbeFailed(),
            reward_string="No admin creds."
        )
        lib['ScanForCreds'] = m.VulnerabilityInfo(
            description="Scan the local security managers for credentials. Need to be admin on the box.",
            type=m.VulnerabilityType.LOCAL,
            outcome=m.LeakedCredentials(credentials=[m.CachedCredential(node="domain_controller_1", port="AD", credential="dc_1")]),
            precondition=m.Precondition("admin"),
            rates=m.Rates(successRate=0.9),
            reward_string="DA credentials found"
        )
        return lib

    def breach_vulnerabilities(lib: m.VulnerabilityLibrary) -> m.VulnerabilityLibrary:
        # NOTE: The randomization here does not support random credential leakage... This is kind of a major limitation for tooling like responder
        credrandset = set([random.randrange(0, n_users) for i in range(0, random.randrange(3, n_clients))])
        lib['AuthorizationSpoofAndCrack'] = m.VulnerabilityInfo(
            description="Spoof an authoritative source on the network to get a crackable hash, then try to crack it",
            type=m.VulnerabilityType.LOCAL,
            outcome=m.LeakedCredentials(credentials=[m.CachedCredential(node=f"share_{shareid}", port="SMB", credential=f"user_{credind}") for credind in credrandset for shareid in range(0, n_servers)]
                                        + [m.CachedCredential(node=f"workstation_{credind % n_clients}", port="SHELL", credential=f"user_{credind}") for credind in credrandset])
        )
        return lib

    def admin_vulnerabilities(lib: m.VulnerabilityLibrary) -> m.VulnerabilityLibrary:
        lib["ProbeAdmin"] = m.VulnerabilityInfo(
            description="Probe a workstation to see if you have admin creds on it",
            type=m.VulnerabilityType.LOCAL,
            outcome=m.ProbeSucceeded(discovered_properties=["admin"]),
            reward_string="Admin creds verified."
        )
        return lib

    def dc_vulnerabilities(lib: m.VulnerabilityLibrary) -> m.VulnerabilityLibrary:
        lib['DumpNTDS'] = m.VulnerabilityInfo(
            description="Dump the NTDS file from AD",
            type=m.VulnerabilityType.LOCAL,
            outcome=m.LeakedCredentials([m.CachedCredential(node=f"share_{shareind}", port="SMB", credential=f"user_{credind}") for credind in range(0, n_users) for shareind in range(0, n_servers)]
                                        + [m.CachedCredential(node=f"workstation_{wkid}", port="SHELL", credential=f"user_{uid}") for wkid in range(0, n_clients) for uid in range(0, n_users)]),
            precondition=m.Precondition("domain_controller"),
            reward_string="Dumped all user hashes. Get crackin'"
        )
        return lib

    # Workstation 1 is our entry node
    entry_node_id = "workstation_0"
    graph.nodes[entry_node_id].clear()
    graph.nodes[entry_node_id].update(
        {'data': m.NodeInfo(services=[],
                            value=0,
                            properties=["breach_node"],
                            vulnerabilities=breach_vulnerabilities(default_vulnerabilities()),
                            agent_installed=True,
                            firewall=firewall_conf,
                            reimagable=False)})

    for i in range(1, n_clients):
        nodeid = f"workstation_{i}"
        graph.nodes[nodeid].clear()
        props = []
        vulns = default_vulnerabilities()
        if random.random() > 0.2:  # TODO: make this value rarer as network size increases? Kinda wonky considering we only have one exploit path really
            props = ["admin"]
            vulns = admin_vulnerabilities(vulns)
        graph.nodes[nodeid].update({'data': m.NodeInfo(
            services=[m.ListeningService(name="SHELL", allowedCredentials=[f"user_{uid}" for uid in range(0, n_users) if uid % n_clients == i])],
            properties=props,
            value=1,
            firewall=firewall_conf,
            vulnerabilities=vulns
        )})

    for i in range(0, n_servers):
        nodeid = f"share_{i}"
        graph.nodes[nodeid].clear()
        graph.nodes[nodeid].update({'data': m.NodeInfo(
            services=[m.ListeningService(name="SMB", allowedCredentials=[f"user_{sid}" for sid in range(0, n_users) if sid % n_servers == i])],
            properties=[],
            value=5,
            firewall=firewall_conf,
            vulnerabilities=default_vulnerabilities()
        )})

    nodeid = "domain_controller_1"
    graph.nodes[nodeid].clear()
    graph.nodes[nodeid].update({'data': m.NodeInfo(
        services=[m.ListeningService(name="AD", allowedCredentials=["dc_1"])],
        properties=["domain_controller"],
        value=1000,
        firewall=firewall_conf,
        vulnerabilities=dc_vulnerabilities(default_vulnerabilities())
    )})

    return graph


def new_random_environment(seed: Any) -> m.Environment:
    """Create a new simulation environment based on
    a randomly generated network topology for SMB shares.
    """
    random.seed(seed)
    clients = random.randrange(5, 10)
    servers = random.randrange(1, 2)
    users = random.randrange(20, 100)
    network = create_network_from_smb_traffic(clients, servers, users)

    return m.Environment(network=network,
                         vulnerability_library=dict([]),
                         identifiers=ENV_IDENTIFIERS)
