""" Generating random active directory networks"""
import random
from cyberbattle.simulation.model import CredentialID, FirewallConfiguration, FirewallRule, Identifiers, RulePermission
from cyberbattle.simulation import model as m
import networkx as nx

ENV_IDENTIFIERS = Identifiers(
    properties=[
        'breach_node',
        'domain_controller'
    ],
    ports=['SMB'],
    local_vulnerabilities=[
        'FindDomainControllers',
        # 'EnumerateFileShares',
        'LLMNR/NBT-NSpoison',
        'DomainPasswordSpray',
        'FindUserSPNs',
        'DumpNTDS'
    ],
    remote_vulnerabilities=[
        'ScanWindowsLSA',
        'ScanWindowsSAM',
        'PassTheHash'
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
    graph.add_node("domain_controller")
    graph.add_edges_from([(f"share_{i}", f"workstation_{j}") for i in range(0, n_servers) for j in range(0, n_clients)])
    graph.add_edges_from([("domain_controller", f"workstation_{i}") for i in range(0, n_clients)])
    graph.add_edges_from([("domain_controller", f"share_{i}") for i in range(0, n_servers)])

    password_counter = 0

    def generate_password() -> CredentialID:
        nonlocal password_counter
        password_counter = password_counter + 1
        return f'unique_pwd{password_counter}'

    # domain_admin_creds = [generate_password() for _ in range(0, n_domain_admins)]  # domain admin credentials in AD
    # all_creds = domain_admin_creds + [generate_password() for _ in range(0, n_users)]  # all credentials in AD

    firewall_conf = FirewallConfiguration(
        [FirewallRule("SMB", RulePermission.ALLOW)],
        [FirewallRule("SMB", RulePermission.ALLOW)])

    # Pick a random workstation node as the agent entry node
    entry_node_index = random.randrange(n_clients)
    entry_node_id = f"workstation_{entry_node_index}"
    graph.nodes[entry_node_id].clear()
    graph.nodes[entry_node_id].update(
        {'data': m.NodeInfo(services=[],
                            value=0,
                            properties=["breach_node"],
                            vulnerabilities={},
                            agent_installed=True,
                            firewall=firewall_conf,
                            reimagable=False)})

    def create_node_data(node_id: m.NodeID):
        return m.NodeInfo(
            services=[],
            value=random.randint(0, 100),
            vulnerabilities={},
            agent_installed=False,
            firewall=firewall_conf
        )

    for node in list(graph.nodes):
        if node != entry_node_id:
            graph.nodes[node].clear()
            graph.nodes[node].update({'data': create_node_data(node)})

    return graph


def new_environment():
    """Create a new simulation environment based on
    a randomly generated network topology for SMB shares.
    """
    network = create_network_from_smb_traffic(2, 2, 2, 2)

    return m.Environment(network=network,
                         vulnerability_library=dict([]),
                         identifiers=ENV_IDENTIFIERS)
