# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Union, Tuple

import gym
import numpy as onp
import networkx as nx

from .graph_spaces import DiGraph


Action = Union[Tuple[int, int], Tuple[int, int, int], Tuple[int, int, int, int]]


class CyberBattleGraph(gym.Wrapper):
    """

    A wrapper for CyberBattleSim that maintains the agent's
    knowledge graph containing information about the subset
     of the network that was explored so far.

    Currently the nodes of this graph are a subset of the environment nodes.
    Eventually we will add new node types to represent various entities
    like credentials and users. Edges will represent relationships between those entities
    (e.g. user X is authenticated with machine Y using credential Z).

    Actions
    -------

    Actions are of the form:

    .. code:: python

        (kind, *indicators)


    The ``kind`` which is one of

    .. code:: python

        # kind
        0: Local Vulnerability
        1: Remote Vulnerability
        2: Connect

    The indicators vary in meaning and length, depending on the ``kind``:

    .. code:: python

        # kind=0 (Local Vulnerability)
        indicators = (node_id, local_vulnerability_id)

        # kind=1 (Remote Vulnerability)
        indicators = (from_node_id, to_node_id, remote_vulnerability_id)

        # kind=2 (Connect)
        indicators = (from_node_id, to_node_id, port_id, credential_id)

    The node ids can be obtained from the graph, e.g.

    .. code:: python

        node_ids = observation['graph'].keys()

    The other indicators are listed below.

    .. code:: python

        # local_vulnerability_ids
        0: ScanBashHistory
        1: ScanExplorerRecentFiles
        2: SudoAttempt
        3: CrackKeepPassX
        4: CrackKeepPass

        # remote_vulnerability_ids
        0: ProbeLinux
        1: ProbeWindows

        # port_ids
        0: HTTPS
        1: GIT
        2: SSH
        3: RDP
        4: PING
        5: MySQL
        6: SSH-key
        7: su

    Examples
    ~~~~~~~~
    Here are some example actions:

    .. code:: python

        a = (0, 5, 3)        # try local vulnerability "CrackKeepPassX" on node 5
        a = (1, 5, 7, 1)     # try remote vulnerability "ProbeWindows" from node 5 to node 7
        a = (2, 5, 7, 3, 2)  # try to connect from node 5 to node 7 using credential 2 over RDP port


    Observations
    ------------

    Observations are graphs of the nodes that have been discovered so far. Each node is annotated
    with a dict of properties of the form:

    .. code:: python

        node_properties = {
            'name': 'FooServer',                         # human-readable identifier
            'privilege_level': 1,                        # 0: not owned, 1: admin, 2: system
            'flags': array(-1, 0,  1, 0,  0, ...,  0]),  # 1: set, -1: unset, 0: unknown
            'credentials': array([-1, 5, -1, ..., -1]),  # array of ports (-1 means no cred)
            'has_leaked_creds': True,                    # whether node has leaked any credentials so far
        }

        # flag_ids
        0: Windows
        1: Linux
        2: ApacheWebSite
        3: IIS_2019
        4: IIS_2020_patched
        5: MySql
        6: Ubuntu
        7: nginx/1.10.3
        8: SMB_vuln
        9: SMB_vuln_patched
        10: SQLServer
        11: Win10
        12: Win10Patched
        13: FLAG:Linux

    Note that the **position** of a non-trivial port number in ``'credentials'`` corresponds to the
    credential id. Therefore, for the node in the example above, we have a known credential on
    :code:`port_id=5` with :code:`credential_id=1` (the position in the array).


    """
    __kinds = ('local_vulnerability', 'remote_vulnerability', 'connect')

    def __init__(self, env, maximum_total_credentials=22, maximum_node_count=22):
        super().__init__(env)
        self._bounds = self.env.bounds
        self.__graph = nx.DiGraph()
        self.observation_space = DiGraph(self.bounds.maximum_node_count)

    def reset(self):
        observation = self.env.reset()
        self.__graph = nx.DiGraph()
        self.__add_node(observation)
        self.__update_nodes(observation)
        return self.__graph

    def step(self, action: Action):
        """

        Take a step in the MDP.

        Args:
            action: An *abstract* action.

        Returns:
            observation: The next-step observation.
            reward: The reward associated with the given action (and previous observation).
            done: Whether the next-step observation is a terminal state.
            info: Some additional info.

        """
        kind_id, *indicators = action
        observation, reward, done, info = self.env.step({self.__kinds[kind_id]: indicators})
        for _ in range(observation['newly_discovered_nodes_count']):
            self.__add_node(observation)
        if True:  # TODO: do we need to update edges and nodes every time?
            self.__update_edges(observation)
            self.__update_nodes(observation)
        return self.__graph, reward, done, info

    def __add_node(self, observation):
        while self.__graph.number_of_nodes() < observation['discovered_node_count']:
            node_index = self.__graph.number_of_nodes()
            creds = onp.full(self._bounds.maximum_total_credentials, -1, dtype=onp.int8)
            self.__graph.add_node(
                node_index,
                name=observation['_discovered_nodes'][node_index],
                privilege_level=None, flags=None,  # these are set by __update_nodes()
                credentials=creds,
                has_leaked_creds=False,
            )

    def __update_edges(self, observation):
        g_orig = observation['_explored_network']
        node_ids = {n: i for i, n in enumerate(observation['_discovered_nodes'])}
        for (from_name, to_name), edge_properties in g_orig.edges.items():
            self.__graph.add_edge(node_ids[from_name], node_ids[to_name], **edge_properties)

    def __update_nodes(self, observation):
        node_properties = zip(
            observation['nodes_privilegelevel'],
            observation['discovered_nodes_properties'],
        )
        for node_id, (privilege_level, flags) in enumerate(node_properties):
            # This value is already provided in self.__graph.nodes[node_id]['data'].privilege_level
            self.__graph.nodes[node_id]['privilege_level'] = privilege_level
            # This value is already provided in self.__graph.nodes[node_id]['data'].properties
            self.__graph.nodes[node_id]['flags'] = flags

        for cred_id, (node_id, port_id) in enumerate(observation['credential_cache_matrix']):
            node_id, port_id = int(node_id), int(port_id)
            # NOTE: this code ignores situations where the same cred_id is
            # used for two different ports (This can be the case, even on the same node for two different ports.)
            self.__graph.nodes[node_id]['credentials'][cred_id] = port_id
            # Mark the node has leaking credentials
            self.__graph.nodes[node_id]['has_leaked_creds'] = True
