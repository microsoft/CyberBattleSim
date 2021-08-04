# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Data model for the simulation environment.

The simulation environment is given by the directed graph
formally defined by:

  Node := NodeID x ListeningService[] x Value x Vulnerability[] x FirewallConfig
  Edge := NodeID x NodeID x PortName

 where:
  - NodeID: string
  - ListeningService : Name x AllowedCredentials
  - AllowedCredentials : string[] # credential pair represented by just a
    string ID
  - Value : [0...100]     # Intrinsic value of reaching this node
  - Vulnerability : VulnerabilityID x Type x Precondition x Outcome x Rates
  - VulnerabilityID : string
  - Rates : ProbingDetectionRate x ExploitDetectionRate x SuccessRate
  - FirewallConfig: {
      outgoing :  FirwallRule[]
      incoming : FirwallRule [] }
  - FirewallRule: PortName x { ALLOW, BLOCK }
"""

from datetime import datetime, time
from typing import NamedTuple, List, Dict, Optional, Union, Tuple, Iterator
import dataclasses
from dataclasses import dataclass
import matplotlib.pyplot as plt  # type:ignore
from enum import Enum, IntEnum
import boolean
import networkx as nx
import yaml
import random

VERSION_TAG = "0.1.0"

ALGEBRA = boolean.BooleanAlgebra()

# Type alias for identifiers
NodeID = str

# A unique identifier
ID = str

# a (login,password/token) credential pair is abstracted as just a unique
# string identifier
CredentialID = str

# Intrinsic value of a reaching a given node in [0,100]
NodeValue = int


PortName = str


@dataclass
class ListeningService:
    """A service port on a given node accepting connection initiated
    with the specified allowed credentials """
    # Name of the port the service is listening to
    name: PortName
    # credential allowed to authenticate with the service
    allowedCredentials: List[CredentialID] = dataclasses.field(default_factory=list)
    # whether the service is running or stopped
    running: bool = True
    # Weight used to evaluate the cost of not running the service
    sla_weight = 1.0


x = ListeningService(name='d')
VulnerabilityID = str

# Probability rate
Probability = float

# The name of a node property indicating the presence of a
# service, component, feature or vulnerability on a given node.
PropertyName = str


class Rates(NamedTuple):
    """Probabilities associated with a given vulnerability"""
    probingDetectionRate: Probability = 0.0
    exploitDetectionRate: Probability = 0.0
    successRate: Probability = 1.0


class VulnerabilityType(Enum):
    """Is the vulnerability exploitable locally or remotely?"""
    LOCAL = 1
    REMOTE = 2


class PrivilegeLevel(IntEnum):
    """Access privilege level on a given node"""
    NoAccess = 0
    LocalUser = 1
    Admin = 2
    System = 3
    MAXIMUM = 3


def escalate(current_level, escalation_level: PrivilegeLevel) -> PrivilegeLevel:
    return PrivilegeLevel(max(int(current_level), int(escalation_level)))


class VulnerabilityOutcome:
    """Outcome of exploiting a given vulnerability"""


class LateralMove(VulnerabilityOutcome):
    """Lateral movement to the target node"""
    success: bool


class CustomerData(VulnerabilityOutcome):
    """Access customer data on target node"""


class PrivilegeEscalation(VulnerabilityOutcome):
    """Privilege escalation outcome"""

    def __init__(self, level: PrivilegeLevel):
        self.level = level

    @property
    def tag(self):
        """Escalation tag that gets added to node properties when
        the escalation level is reached for that node"""
        return f"privilege_{self.level}"


class SystemEscalation(PrivilegeEscalation):
    """Escalation to SYSTEM privileges"""

    def __init__(self):
        super().__init__(PrivilegeLevel.System)


class AdminEscalation(PrivilegeEscalation):
    """Escalation to local administrator privileges"""

    def __init__(self):
        super().__init__(PrivilegeLevel.Admin)


class ProbeSucceeded(VulnerabilityOutcome):
    """Probing succeeded"""

    def __init__(self, discovered_properties: List[PropertyName]):
        self.discovered_properties = discovered_properties


class ProbeFailed(VulnerabilityOutcome):
    """Probing failed"""


class ExploitFailed(VulnerabilityOutcome):
    """This is for situations where the exploit fails """


class CachedCredential(NamedTuple):
    """Encodes a machine-port-credential triplet"""
    node: NodeID
    port: PortName
    credential: CredentialID


class LeakedCredentials(VulnerabilityOutcome):
    """A set of credentials obtained by exploiting a vulnerability"""

    credentials: List[CachedCredential]

    def __init__(self, credentials: List[CachedCredential]):
        self.credentials = credentials


class LeakedNodesId(VulnerabilityOutcome):
    """A set of node IDs obtained by exploiting a vulnerability"""

    def __init__(self, nodes: List[NodeID]):
        self.nodes = nodes


VulnerabilityOutcomes = Union[
    LeakedCredentials, LeakedNodesId, PrivilegeEscalation, AdminEscalation,
    SystemEscalation, CustomerData, LateralMove, ExploitFailed]


class AttackResult():
    """The result of attempting a specific attack (either local or remote)"""
    success: bool
    expected_outcome: Union[VulnerabilityOutcomes, None]


class Precondition:
    """ A predicate logic expression defining the condition under which a given
    feature or vulnerability is present or not.
    The symbols used in the expression refer to properties associated with
    the corresponding node.
    E.g. 'Win7', 'Server', 'IISInstalled', 'SQLServerInstalled',
    'AntivirusInstalled' ...
    """

    expression: boolean.Expression

    def __init__(self, expression: Union[boolean.Expression, str]):
        if isinstance(expression, boolean.Expression):
            self.expression = expression
        else:
            self.expression = ALGEBRA.parse(expression)


class VulnerabilityInfo(NamedTuple):
    """Definition of a known vulnerability"""
    # an optional description of what the vulnerability is
    description: str
    # type of vulnerability
    type: VulnerabilityType
    # what happens when successfully exploiting the vulnerability
    outcome: VulnerabilityOutcome
    # a boolean expression over a node's properties determining if the
    # vulnerability is present or not
    precondition: Precondition = Precondition("true")
    # rates of success/failure associated with this vulnerability
    rates: Rates = Rates()
    # points to information about the vulnerability
    URL: str = ""
    # some cost associated with exploiting this vulnerability (e.g.
    # brute force more costly than dumping credentials)
    cost: float = 1.0
    # a string displayed when the vulnerability is successfully exploited
    reward_string: str = ""


# A dictionary storing information about all supported vulnerabilities
# or features supported by the simulation.
# This is to be used as a global dictionary pre-populated before
# starting the simulation and estimated from real-world data.
VulnerabilityLibrary = Dict[VulnerabilityID, VulnerabilityInfo]


class RulePermission(Enum):
    """Determine if a rule is blocks or allows traffic"""
    ALLOW = 0
    BLOCK = 1


class FirewallRule(NamedTuple):
    """A firewall rule"""
    # A port name
    port: PortName
    # permission on this port
    permission: RulePermission
    # An optional reason for the block/allow rule
    reason: str = ""


class FirewallConfiguration(NamedTuple):
    """Firewall configuration on a given node.
    Determine if traffic should be allowed or specifically blocked
    on a given port for outgoing and incoming traffic.
    The rules are process in order: the first rule matching a given
    port is applied and the rest are ignored.

    Port that are not listed in the configuration
    are assumed to be blocked. (Adding an explicit block rule
    can still be useful to give a reason for the block.)
    """
    outgoing: List[FirewallRule] = [
        FirewallRule("RDP", RulePermission.ALLOW),
        FirewallRule("SSH", RulePermission.ALLOW),
        FirewallRule("HTTPS", RulePermission.ALLOW),
        FirewallRule("HTTP", RulePermission.ALLOW)]
    incoming: List[FirewallRule] = [
        FirewallRule("RDP", RulePermission.ALLOW),
        FirewallRule("SSH", RulePermission.ALLOW),
        FirewallRule("HTTPS", RulePermission.ALLOW),
        FirewallRule("HTTP", RulePermission.ALLOW)]


class MachineStatus(Enum):
    """Machine running status"""
    Stopped = 0
    Running = 1
    Imaging = 2


@dataclass
class NodeInfo:
    """A computer node in the enterprise network"""
    # List of port/protocol the node is listening to
    services: List[ListeningService]
    # List of known vulnerabilities for the node
    vulnerabilities: VulnerabilityLibrary = dataclasses.field(default_factory=dict)
    # Intrinsic value of the node (translates into a reward if the node gets owned)
    value: NodeValue = 0
    # Properties of the nodes, some of which can imply further vulnerabilities
    properties: List[PropertyName] = dataclasses.field(default_factory=list)
    # Fireall configuration of the node
    firewall: FirewallConfiguration = FirewallConfiguration()
    # Attacker agent installed on the node? (aka the node is 'pwned')
    agent_installed: bool = False
    # Esclation level
    privilege_level: PrivilegeLevel = PrivilegeLevel.NoAccess
    # Can the node be re-imaged by a defender agent?
    reimagable: bool = True
    # Last time the node was reimaged
    last_reimaging: Optional[time] = None
    # String displayed when the node gets owned
    owned_string: str = ""
    # Machine status: running or stopped
    status = MachineStatus.Running
    # Relative node weight used to calculate the cost of stopping this machine
    # or its services
    sla_weight: float = 1.0


class Identifiers(NamedTuple):
    """Define the global set of identifiers used
    in the definition of a given environment.
    Such set defines a common vocabulary possibly
    shared across multiple environments, thus
    ensuring a consistent numbering convention
    that a machine learniong model can learn from."""
    # Array of all possible node property identifiers
    properties: List[PropertyName] = []
    # Array of all possible port names
    ports: List[PortName] = []
    # Array of all possible local vulnerabilities names
    local_vulnerabilities: List[VulnerabilityID] = []
    # Array of all possible remote vulnerabilities names
    remote_vulnerabilities: List[VulnerabilityID] = []


def iterate_network_nodes(network: nx.graph.Graph) -> Iterator[Tuple[NodeID, NodeInfo]]:
    """Iterates over the nodes in the network"""
    for nodeid, nodevalue in network.nodes.items():
        node_data: NodeInfo = nodevalue['data']
        yield nodeid, node_data


class Environment(NamedTuple):
    """ The static graph defining the network of computers """
    network: nx.DiGraph
    vulnerability_library: VulnerabilityLibrary
    identifiers: Identifiers
    creationTime: datetime = datetime.utcnow()
    lastModified: datetime = datetime.utcnow()
    # a version tag indicating the environment schema version
    version: str = VERSION_TAG

    def nodes(self) -> Iterator[Tuple[NodeID, NodeInfo]]:
        """Iterates over the nodes in the network"""
        return iterate_network_nodes(self.network)

    def get_node(self, node_id: NodeID) -> NodeInfo:
        """Retrieve info for the node with the specified ID"""
        node_info: NodeInfo = self.network.nodes[node_id]['data']
        return node_info

    def plot_environment_graph(self) -> None:
        """Plot the full environment graph"""
        nx.draw(self.network,
                with_labels=True,
                node_color=[n['data'].value
                            for i, n in
                            self.network.nodes.items()],
                cmap=plt.cm.Oranges)  # type:ignore


def create_network(nodes: Dict[NodeID, NodeInfo]) -> nx.DiGraph:
    """Create a network with a set of nodes and no edges"""
    graph = nx.DiGraph()
    graph.add_nodes_from([(k, {'data': v}) for (k, v) in list(nodes.items())])
    return graph

# Helpers to infer constants from an environment


def collect_ports_from_vuln(vuln: VulnerabilityInfo) -> List[PortName]:
    """Returns all the port named referenced in a given vulnerability"""
    if isinstance(vuln.outcome, LeakedCredentials):
        return [c.port for c in vuln.outcome.credentials]
    else:
        return []


def collect_vulnerability_ids_from_nodes_bytype(
        nodes: Iterator[Tuple[NodeID, NodeInfo]],
        global_vulnerabilities: VulnerabilityLibrary,
        type: VulnerabilityType) -> List[VulnerabilityID]:
    """Collect and return all IDs of all vulnerability of the specified type
    that are referenced in a given set of nodes and vulnerability library
    """
    return sorted(list({
        id
        for _, node_info in nodes
        for id, v in node_info.vulnerabilities.items()
        if v.type == type
    }.union(
        id
        for id, v in global_vulnerabilities.items()
        if v.type == type
    )))


def collect_properties_from_nodes(nodes: Iterator[Tuple[NodeID, NodeInfo]]) -> List[PropertyName]:
    """Collect and return sorted list of all property names used in a given set of nodes"""
    return sorted({
        p
        for _, node_info in nodes
        for p in node_info.properties
    })


def collect_ports_from_nodes(
        nodes: Iterator[Tuple[NodeID, NodeInfo]],
        vulnerability_library: VulnerabilityLibrary) -> List[PortName]:
    """Collect and return all port names used in a given set of nodes
    and global vulnerability library"""
    return sorted(list({
        port
        for _, v in vulnerability_library.items()
        for port in collect_ports_from_vuln(v)
    }.union({
        port
        for _, node_info in nodes
        for _, v in node_info.vulnerabilities.items()
        for port in collect_ports_from_vuln(v)
    }.union(
        {service.name
         for _, node_info in nodes
         for service in node_info.services}))))


def collect_ports_from_environment(environment: Environment) -> List[PortName]:
    """Collect and return all port names used in a given environment"""
    return collect_ports_from_nodes(environment.nodes(), environment.vulnerability_library)


def infer_constants_from_nodes(
        nodes: Iterator[Tuple[NodeID, NodeInfo]],
        vulnerabilities: Dict[VulnerabilityID, VulnerabilityInfo]) -> Identifiers:
    """Infer global environment constants from a given network"""
    return Identifiers(
        properties=collect_properties_from_nodes(nodes),
        ports=collect_ports_from_nodes(nodes, vulnerabilities),
        local_vulnerabilities=collect_vulnerability_ids_from_nodes_bytype(
            nodes, vulnerabilities, VulnerabilityType.LOCAL),
        remote_vulnerabilities=collect_vulnerability_ids_from_nodes_bytype(
            nodes, vulnerabilities, VulnerabilityType.REMOTE)
    )


def infer_constants_from_network(
        network: nx.Graph,
        vulnerabilities: Dict[VulnerabilityID, VulnerabilityInfo]) -> Identifiers:
    """Infer global environment constants from a given network"""
    return infer_constants_from_nodes(iterate_network_nodes(network), vulnerabilities)


# Network creation

# A sample set of envrionment constants
SAMPLE_IDENTIFIERS = Identifiers(
    ports=['RDP', 'SSH', 'SMB', 'HTTP', 'HTTPS', 'WMI', 'SQL'],
    properties=[
        'Windows', 'Linux', 'HyperV-VM', 'Azure-VM', 'Win7', 'Win10',
        'PortRDPOpen', 'GuestAccountEnabled']
)


def assign_random_labels(
        graph: nx.DiGraph,
        vulnerabilities: VulnerabilityLibrary = dict([]),
        identifiers: Identifiers = SAMPLE_IDENTIFIERS) -> nx.DiGraph:
    """Create an envrionment network by randomly assigning node information
    (properties, firewall configuration, vulnerabilities)
    to the nodes of a given graph structure"""

    # convert node IDs to string
    graph = nx.relabel_nodes(graph, {i: str(i) for i in graph.nodes})

    def create_random_firewall_configuration() -> FirewallConfiguration:
        return FirewallConfiguration(
            outgoing=[
                FirewallRule(port=p, permission=RulePermission.ALLOW)
                for p in
                random.sample(
                    identifiers.ports,
                    k=random.randint(0, len(identifiers.ports)))],
            incoming=[
                FirewallRule(port=p, permission=RulePermission.ALLOW)
                for p in random.sample(
                    identifiers.ports,
                    k=random.randint(0, len(identifiers.ports)))])

    def create_random_properties() -> List[PropertyName]:
        return list(random.sample(
            identifiers.properties,
            k=random.randint(0, len(identifiers.properties))))

    def pick_random_global_vulnerabilities() -> VulnerabilityLibrary:
        count = random.random()
        return {k: v for (k, v) in vulnerabilities.items() if random.random() > count}

    def add_leak_neighbors_vulnerability(library: VulnerabilityLibrary, node_id: NodeID) -> None:
        """Create a vulnerability for each node that reveals its immediate neighbors"""
        neighbors = {t for (s, t) in graph.edges() if s == node_id}
        if len(neighbors) > 0:
            library['RecentlyAccessedMachines'] = VulnerabilityInfo(
                description="AzureVM info, including public IP address",
                type=VulnerabilityType.LOCAL,
                outcome=LeakedNodesId(list(neighbors)))

    def create_random_vulnerabilities(node_id: NodeID) -> VulnerabilityLibrary:
        library = pick_random_global_vulnerabilities()
        add_leak_neighbors_vulnerability(library, node_id)
        return library

    # Pick a random node as the agent entry node
    entry_node_index = random.randrange(len(graph.nodes))
    entry_node_id, entry_node_data = list(graph.nodes(data=True))[entry_node_index]
    graph.nodes[entry_node_id].clear()
    node_data = NodeInfo(services=[],
                         value=0,
                         properties=create_random_properties(),
                         vulnerabilities=create_random_vulnerabilities(entry_node_id),
                         firewall=create_random_firewall_configuration(),
                         agent_installed=True,
                         reimagable=False,
                         privilege_level=PrivilegeLevel.Admin)
    graph.nodes[entry_node_id].update({'data': node_data})

    def create_random_node_data(node_id: NodeID) -> NodeInfo:
        return NodeInfo(
            services=[],
            value=random.randint(0, 100),
            properties=create_random_properties(),
            vulnerabilities=create_random_vulnerabilities(node_id),
            firewall=create_random_firewall_configuration(),
            agent_installed=False,
            privilege_level=PrivilegeLevel.NoAccess)

    for node in list(graph.nodes):
        if node != entry_node_id:
            graph.nodes[node].clear()
            graph.nodes[node].update({'data': create_random_node_data(node)})

    return graph


# Serialization

def setup_yaml_serializer() -> None:
    """Setup a clean YAML formatter for object of type Environment.
    """

    yaml.add_representer(Precondition,
                         lambda dumper, data: dumper.represent_scalar('!BooleanExpression',
                                                                      str(data.expression)))  # type: ignore
    yaml.SafeLoader.add_constructor('!BooleanExpression',
                                    lambda loader, expression: Precondition(
                                        loader.construct_scalar(expression)))  # type: ignore
    yaml.add_constructor('!BooleanExpression',
                         lambda loader, expression:
                         Precondition(loader.construct_scalar(expression)))  # type: ignore

    yaml.add_representer(VulnerabilityType,
                         lambda dumper, data: dumper.represent_scalar('!VulnerabilityType',
                                                                      str(data.name)))  # type: ignore

    yaml.SafeLoader.add_constructor('!VulnerabilityType',
                                    lambda loader, expression: VulnerabilityType[
                                        loader.construct_scalar(expression)])  # type: ignore
    yaml.add_constructor('!VulnerabilityType',
                         lambda loader, expression: VulnerabilityType[
                             loader.construct_scalar(expression)])  # type: ignore
