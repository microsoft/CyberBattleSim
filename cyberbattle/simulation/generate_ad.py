""" Generating random active directory networks"""
from cyberbattle.simulation.model import Identifiers
import numpy as np
from cyberbattle.simulation import model as m
from cyberbattle.simulation.generate_network import generate_random_traffic_network
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
    traffic: nx.DiGraph,
) -> nx.DiGraph:
    graph = traffic
    # TODO: add edge for DC to all nodes
    # TODO: fill in data for SMB shares (probably pull from generate_network.py)
    return graph


def new_environment(n_shares: int):
    """Create a new simulation environment based on
    a randomly generated network topology for SMB shares.
    """
    traffic = generate_random_traffic_network(seed=None,
                                              n_clients=50,
                                              n_servers={
                                                  "SMB": n_shares,
                                              },
                                              alpha=np.array([(1, 1), (0.2, 0.5)]),
                                              beta=np.array([(1000, 10), (10, 100)]))

    network = create_network_from_smb_traffic(traffic)

    return m.Environment(network=network,
                         vulnerability_library=dict([]),
                         identifiers=ENV_IDENTIFIERS)
