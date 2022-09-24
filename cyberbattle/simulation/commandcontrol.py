# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A 'Command & control'-like interface exposing to a human player
 the attacker view and actions of the game.
This includes commands to visualize the part of the environment
that were explored so far, and for each node where the attacker client
is installed, execute actions on the machine.
"""
import networkx as nx
from typing import List, Optional, Dict, Union, Tuple, Set
import plotly.graph_objects as go

from . import model, actions


class CommandControl:
    """ The Command and Control interface to the simulation.

    This represents a server that centralize information and secrets
    retrieved from the individual clients running on the network nodes.
    """

    # Global list aggregating all credentials gathered so far, from any node in the network
    __gathered_credentials: Set[model.CachedCredential]
    _actuator: actions.AgentActions
    __environment: model.Environment
    __total_reward: float

    def __init__(self, environment_or_actuator: Union[model.Environment, actions.AgentActions]):
        if isinstance(environment_or_actuator, model.Environment):
            self.__environment = environment_or_actuator
            self._actuator = actions.AgentActions(self.__environment, throws_on_invalid_actions=True)
        elif isinstance(environment_or_actuator, actions.AgentActions):
            self.__environment = environment_or_actuator._environment
            self._actuator = environment_or_actuator
        else:
            raise ValueError(
                "Invalid type: expecting Union[model.Environment, actions.AgentActions])")

        self.__gathered_credentials = set()
        self.__total_reward = 0

    def __save_credentials(self, outcome: model.VulnerabilityOutcome) -> None:
        """Save credentials obtained from exploiting a vulnerability"""
        if isinstance(outcome, model.LeakedCredentials):
            self.__gathered_credentials.update(outcome.credentials)
        return

    def __accumulate_reward(self, reward: actions.Reward) -> None:
        """Accumulate new reward"""
        self.__total_reward += reward

    def total_reward(self) -> actions.Reward:
        """Return the current accumulated reward"""
        return self.__total_reward

    def list_nodes(self) -> List[actions.DiscoveredNodeInfo]:
        """Returns the list of nodes ID that were discovered or owned by the attacker."""
        return self._actuator.list_nodes()

    def get_node_color(self, node_info: model.NodeInfo) -> str:
        if node_info.agent_installed:
            return 'red'
        else:
            return 'green'

    def plot_nodes(self) -> None:
        """Plot the sub-graph of nodes either so far
        discovered  (their ID is knowned by the agent)
        or owned (i.e. where the attacker client is installed)."""
        discovered_nodes = [node_id for node_id, _ in self._actuator.discovered_nodes()]
        sub_graph = self.__environment.network.subgraph(discovered_nodes)
        nx.draw(sub_graph,
                with_labels=True,
                node_color=[self.get_node_color(self.__environment.get_node(i)) for i in sub_graph.nodes])

    def known_vulnerabilities(self) -> model.VulnerabilityLibrary:
        """Return the global list of known vulnerability."""
        return self.__environment.vulnerability_library

    def list_remote_attacks(self, node_id: model.NodeID) -> List[model.VulnerabilityID]:
        """Return list of all remote attacks that the Command&Control may
        execute onto the specified node."""
        return self._actuator.list_remote_attacks(node_id)

    def list_local_attacks(self, node_id: model.NodeID) -> List[model.VulnerabilityID]:
        """Return list of all local attacks that the Command&Control may
        execute onto the specified node."""
        return self._actuator.list_local_attacks(node_id)

    def list_attacks(self, node_id: model.NodeID) -> List[model.VulnerabilityID]:
        """Return list of all attacks that the Command&Control may
        execute on the specified node."""
        return self._actuator.list_attacks(node_id)

    def list_all_attacks(self) -> List[Dict[str, object]]:
        """List all possible attacks from all the nodes currently owned by the attacker"""
        return self._actuator.list_all_attacks()

    def print_all_attacks(self) -> None:
        """Pretty print list of all possible attacks from all the nodes currently owned by the attacker"""
        return self._actuator.print_all_attacks()

    def run_attack(self,
                   node_id: model.NodeID,
                   vulnerability_id: model.VulnerabilityID
                   ) -> Optional[model.VulnerabilityOutcome]:
        """Run an attack and attempt to exploit a vulnerability on the specified node."""
        result = self._actuator.exploit_local_vulnerability(node_id, vulnerability_id)
        if result.outcome is not None:
            self.__save_credentials(result.outcome)
        self.__accumulate_reward(result.reward)
        return result.outcome

    def run_remote_attack(self, node_id: model.NodeID,
                          target_node_id: model.NodeID,
                          vulnerability_id: model.VulnerabilityID
                          ) -> Optional[model.VulnerabilityOutcome]:
        """Run a remote attack from the specified node to exploit a remote vulnerability
        in the specified target node"""

        result = self._actuator.exploit_remote_vulnerability(
            node_id, target_node_id, vulnerability_id)
        if result.outcome is not None:
            self.__save_credentials(result.outcome)
        self.__accumulate_reward(result.reward)
        return result.outcome

    def connect_and_infect(self, source_node_id: model.NodeID,
                           target_node_id: model.NodeID,
                           port_name: model.PortName,
                           credentials: model.CredentialID) -> bool:
        """Install the agent on a remote machine using the
         provided credentials"""
        result = self._actuator.connect_to_remote_machine(source_node_id, target_node_id, port_name,
                                                          credentials)
        self.__accumulate_reward(result.reward)
        return result.outcome is not None

    @property
    def credentials_gathered_so_far(self) -> Set[model.CachedCredential]:
        """Returns the list of credentials gathered so far by the
         attacker (from any node)"""
        return self.__gathered_credentials


def get_outcome_first_credential(outcome: Optional[model.VulnerabilityOutcome]) -> model.CredentialID:
    """Return the first credential found in a given vulnerability exploit outcome"""
    if outcome is not None and isinstance(outcome, model.LeakedCredentials):
        return outcome.credentials[0].credential
    else:
        raise ValueError('Vulnerability outcome does not contain any credential')


class EnvironmentDebugging:
    """Provides debugging feature exposing internals of the environment
     that are not normally revealed to an attacker agent according to
     the rules of the simulation.
    """
    __environment: model.Environment
    __actuator: actions.AgentActions

    def __init__(self, actuator_or_c2: Union[actions.AgentActions, CommandControl]):
        if isinstance(actuator_or_c2, actions.AgentActions):
            self.__actuator = actuator_or_c2
        elif isinstance(actuator_or_c2, CommandControl):
            self.__actuator = actuator_or_c2._actuator
        else:
            raise ValueError("Invalid type: expecting Union[actions.AgentActions, CommandControl])")

        self.__environment = self.__actuator._environment

    def network_as_plotly_traces(self, xref: str = "x", yref: str = "y") -> Tuple[List[go.Scatter], dict]:
        known_nodes = [node_id for node_id, _ in self.__actuator.discovered_nodes()]

        subgraph = self.__environment.network.subgraph(known_nodes)

        # pos = nx.fruchterman_reingold_layout(subgraph)
        pos = nx.shell_layout(subgraph, [[known_nodes[0]], known_nodes[1:]])

        def edge_text(source: model.NodeID, target: model.NodeID) -> str:
            data = self.__environment.network.get_edge_data(source, target)
            name: str = data['kind'].name
            return name

        color_map = {actions.EdgeAnnotation.LATERAL_MOVE: 'red',
                     actions.EdgeAnnotation.REMOTE_EXPLOIT: 'orange',
                     actions.EdgeAnnotation.KNOWS: 'gray'}

        def edge_color(source: model.NodeID, target: model.NodeID) -> str:
            data = self.__environment.network.get_edge_data(source, target)
            if 'kind' in data:
                return color_map[data['kind']]
            return 'black'

        layout: dict = dict(title="CyberBattle simulation", font=dict(size=10), showlegend=True,
                            autosize=False, width=800, height=400,
                            margin=go.layout.Margin(l=2, r=2, b=15, t=35),
                            hovermode='closest',
                            annotations=[dict(
                                ax=pos[source][0],
                                ay=pos[source][1], axref=xref, ayref=yref,
                                x=pos[target][0],
                                y=pos[target][1], xref=xref, yref=yref,
                                arrowcolor=edge_color(source, target),
                                hovertext=edge_text(source, target),
                                showarrow=True,
                                arrowhead=1,
                                arrowsize=1,
                                arrowwidth=1,
                                startstandoff=10,
                                standoff=10,
                                align='center',
                                opacity=1
                            ) for (source, target) in list(subgraph.edges)]
                            )

        owned_nodes_coordinates = [(i, c) for i, c in pos.items()
                                   if self.get_node_information(i).agent_installed]
        discovered_nodes_coordinates = [(i, c)
                                        for i, c in pos.items()
                                        if not self.get_node_information(i).agent_installed]

        trace_owned_nodes = go.Scatter(
            x=[c[0] for i, c in owned_nodes_coordinates],
            y=[c[1] for i, c in owned_nodes_coordinates],
            mode='markers+text',
            name='owned',
            marker=dict(symbol='circle-dot',
                        size=5,
                        # green #0e9d00
                        color='#D32F2E',  # red
                        line=dict(color='rgb(255,0,0)', width=8)
                        ),
            text=[i for i, c in owned_nodes_coordinates],
            hoverinfo='text',
            textposition="bottom center"
        )

        trace_discovered_nodes = go.Scatter(
            x=[c[0] for i, c in discovered_nodes_coordinates],
            y=[c[1] for i, c in discovered_nodes_coordinates],
            mode='markers+text',
            name='discovered',
            marker=dict(symbol='circle-dot',
                        size=5,
                        color='#0e9d00',  # green
                        line=dict(color='rgb(0,255,0)', width=8)
                        ),
            text=[i for i, c in discovered_nodes_coordinates],
            hoverinfo='text',
            textposition="bottom center"
        )

        dummy_scatter_for_edge_legend = [
            go.Scatter(
                x=[0], y=[0], mode="lines",
                line=dict(color=color_map[a]),
                name=a.name
            ) for a in actions.EdgeAnnotation]

        all_scatters = dummy_scatter_for_edge_legend + [trace_owned_nodes, trace_discovered_nodes]
        return (all_scatters, layout)

    def plot_discovered_network(self) -> None:
        """Plot the network graph with plotly"""
        fig = go.Figure()
        traces, layout = self.network_as_plotly_traces()
        for t in traces:
            fig.add_trace(t)
        fig.update_layout(layout)
        fig.show()

    def get_node_information(self, node_id: model.NodeID) -> model.NodeInfo:
        """Print node information"""
        return self.__environment.get_node(node_id)
