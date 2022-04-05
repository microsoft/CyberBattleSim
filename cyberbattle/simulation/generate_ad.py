""" Generating random active directory networks"""
import random
from cyberbattle.simulation.model import FirewallConfiguration, FirewallRule, Identifiers, RulePermission
from cyberbattle.simulation import model as m
import networkx as nx

ENV_IDENTIFIERS = Identifiers(
    properties=[
        'breach_node',
        'domain_controller'
    ],
    ports=['SMB', 'AD'],
    local_vulnerabilities=[
        'FindDomainControllers',
        'EnumerateFileShares',
        'AuthorizationSpoofAndCrack',
        'FindUserSPNs'
    ],
    remote_vulnerabilities=[
        'DomainPasswordSpray',
        'ScanWindowsLSA',
        'ScanWindowsSAM',
        'PassTheHash',
        'DumpNTDS'
    ]
)


def create_network_from_smb_traffic(
    n_clients: int,
    n_servers: int,
    n_users: int,
    n_domain_admins: int,
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
        [FirewallRule("SMB", RulePermission.ALLOW), FirewallRule("AD", RulePermission.ALLOW)],
        [FirewallRule("SMB", RulePermission.ALLOW), FirewallRule("AD", RulePermission.ALLOW)])

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
        # NOTE: The randomization here does not support random credential leakage... This is kind of a major limitation for tooling like responder
        credrandset = set([random.randrange(0, n_users) for i in range(0, random.randrange(1, 3))])
        lib['AuthorizationSpoofAndCrack'] = m.VulnerabilityInfo(
            description="Spoof an authoritative source on the network to get a crackable hash, then try to crack it",
            type=m.VulnerabilityType.LOCAL,
            rates=m.Rates(successRate=0.2),
            outcome=m.LeakedCredentials(credentials=[m.CachedCredential(node=f"share_{shareid}", port="SMB", credential=f"user_{credind}") for credind in credrandset for shareid in range(0, n_servers)])
        )
        return lib

    # Workstation 1 is our entry node
    entry_node_id = "workstation_0"
    graph.nodes[entry_node_id].clear()
    graph.nodes[entry_node_id].update(
        {'data': m.NodeInfo(services=[],
                            value=0,
                            properties=["breach_node"],
                            vulnerabilities=default_vulnerabilities(),
                            agent_installed=True,
                            firewall=firewall_conf,
                            reimagable=False)})

    for i in range(1, n_clients):
        nodeid = f"workstation_{i}"
        graph.nodes[nodeid].clear()
        graph.nodes[nodeid].update({'data': m.NodeInfo(
            services=[],
            properties=[],
            value=0,
            vulnerabilities=default_vulnerabilities()
        )})

    for i in range(0, n_servers):
        nodeid = f"share_{i}"
        graph.nodes[nodeid].clear()
        graph.nodes[nodeid].update({'data': m.NodeInfo(
            services=[],
            properties=[],
            value=50,
            vulnerabilities=default_vulnerabilities()
        )})

    nodeid = "domain_controller_1"
    graph.nodes[nodeid].clear()
    graph.nodes[nodeid].update({'data': m.NodeInfo(
        services=[],
        properties=["domain_controller"],
        value=100,
        vulnerabilities=default_vulnerabilities()
    )})

    return graph


def new_random_environment():
    """Create a new simulation environment based on
    a randomly generated network topology for SMB shares.
    """
    clients = random.randrange(5, 10)
    servers = random.randrange(5, 10)
    users = random.randrange(50, 5000)
    admins = random.randrange(1, 15)
    network = create_network_from_smb_traffic(clients, servers, users, admins)

    return m.Environment(network=network,
                         vulnerability_library=dict([]),
                         identifiers=ENV_IDENTIFIERS)
