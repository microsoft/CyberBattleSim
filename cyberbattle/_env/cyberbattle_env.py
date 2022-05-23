# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Anatares OpenGym Environment"""


import time
import copy
import logging
import networkx
from networkx import convert_matrix
from typing import NamedTuple, Optional, Tuple, List, Dict, TypeVar, TypedDict, cast

import numpy
import gym
from gym import spaces
from gym.utils import seeding

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cyberbattle._env.defender import DefenderAgent
from cyberbattle.simulation.model import PortName, PrivilegeLevel
from ..simulation import commandcontrol, model, actions
from .discriminatedunion import DiscriminatedUnion


LOGGER = logging.getLogger(__name__)

# Used to allocate a discrete space value representing a field that
# is 'Not Applicable' (of value 0 by convention)
NA = 1

# Value defining an unused space slot
UNUSED_SLOT = numpy.int32(0)
# Value defining a used space slot
USED_SLOT = numpy.int32(1)


# Action Space dictionary type
# The actual action space is later defined as a 'spaces.Dict(x)' where x:ActionSpaceDict
ActionSpaceDict = TypedDict(
    'ActionSpaceDict', {'local_vulnerability': spaces.Space,  # type: ignore
                        'remote_vulnerability': spaces.Space,  # type: ignore
                        'connect': spaces.Space  # type: ignore
                        })

# The type of a sample from the Action space
Action = TypedDict(
    'Action', {'local_vulnerability': numpy.ndarray,
               # adding the generic type causes runtime
               # TypeError `'type' object is not subscriptable'`
               'remote_vulnerability': numpy.ndarray,
               'connect': numpy.ndarray
               }, total=False)

# Type of a sample from the ActionMask space
ActionMask = TypedDict(
    'ActionMask', {'local_vulnerability': numpy.ndarray,
                   'remote_vulnerability': numpy.ndarray,
                   'connect': numpy.ndarray
                   })

# Type of a sample from the Observation space
Observation = TypedDict(
    'Observation', {

        # ---------------------------------------------------------
        # Outcome of the action just executed
        # ---------------------------------------------------------

        # number of new nodes discovered
        'newly_discovered_nodes_count': numpy.int32,

        # whether a lateral move was just performed
        'lateral_move': numpy.int32,

        # whether customer data were just discovered
        'customer_data_found': numpy.int32,

        # 0 if there were no probing attempt
        # 1 if an attempted probing failed
        # 2 if an attempted probing succeeded
        'probe_result': numpy.int32,

        # whether an escalation was completed and to which level
        'escalation': numpy.int32,

        # credentials that were just discovered after executing an action
        'leaked_credentials': Tuple[numpy.ndarray, ...],  # type: ignore

        # bitmask indicating which action are valid in the current state
        'action_mask': ActionMask,

        # ---------------------------------------------------------
        # State information aggregated over all actions executed so far
        # ---------------------------------------------------------

        # size of the credential stack
        'credential_cache_length': int,

        # total nodes discovered so far
        'discovered_node_count': int,

        # Matrix of properties for all the discovered nodes
        'discovered_nodes_properties': numpy.ndarray,

        # Node privilege level on every discovered node (e.g., 0 if not owned, 1 owned, 2  admin, 3 for system)
        'nodes_privilegelevel': numpy.ndarray,

        # Encoding of the credential cache of shape: (credential_cache_length, 2)
        #
        # Each row represent a discovered credential, the row index is the
        # the credential index is given by the row index (i.e. order of discovery)
        # A row is of the form: (target_node_discover_index, port_index)
        'credential_cache_matrix': numpy.ndarray,

        # ---------------------------------------------------------
        # Raw information fields coming from the simulation environment
        # that are not encoded as gym spaces (were previously in the 'info' field)
        # ---------------------------------------------------------

        # internal IDs of the credentials in the cache
        'credential_cache': List[model.CachedCredential],

        # Mapping node index to internal IDs of all nodes discovered so far.
        # The external node index used by the agent to refer to a node
        # is defined as the index of the node in this array
        'discovered_nodes': List[model.NodeID],

        # The subgraph of nodes discovered so far with annotated edges
        # representing interactions that took place during the simulation. (See
        # actions.EdgeAnnotation)
        'explored_network': networkx.DiGraph

    })


# Information returned to gym by the step function
StepInfo = TypedDict(
    'StepInfo', {
        'description': str,
        'duration_in_ms': float,
        'step_count': int,
        'network_availability': float
    })


class OutOfBoundIndexError(Exception):
    """The agent attempted to reference an entity (node or a vulnerability) with an invalid index"""


Key = TypeVar('Key')
Value = TypeVar('Value')


def inverse_dict(self: Dict[Key, Value]) -> Dict[Value, Key]:
    """Inverse a dictionary"""
    return {v: k for k, v in self.items()}


class DummySpace(spaces.Space):
    """This class ensures that the values in the gym.spaces.Dict space are derived from gym.Space"""

    def __init__(self, sample: object):
        self._sample = sample

    def contains(self, obj: object) -> bool:
        return True

    def sample(self) -> object:
        return self._sample


def sourcenode_of_action(x: Action) -> int:
    """Return the source node of a given action"""
    if 'local_vulnerability' in x:
        return x['local_vulnerability'][0]
    elif 'remote_vulnerability' in x:
        return x['remote_vulnerability'][0]

    assert 'connect' in x
    return x['connect'][0]


class EnvironmentBounds(NamedTuple):
    """Define global bounds posisibly shared by a set of CyberBattle gym environments

    maximum_node_count            - Maximum number of nodes in a given network
    maximum_total_credentials     - Maximum number of credentials in a given network
    maximum_discoverable_credentials_per_action - Maximum number of credentials
                                                    that can be returned at a time by any action

    port_count            - Unique protocol ports
    property_count        - Unique node property names
    local_attacks_count   - Unique local vulnerabilities
    remote_attacks_count  - Unique remote vulnerabilities
    """
    maximum_total_credentials: int
    maximum_node_count: int
    maximum_discoverable_credentials_per_action: int

    port_count: int
    property_count: int
    local_attacks_count: int
    remote_attacks_count: int

    @classmethod
    def of_identifiers(cls,
                       identifiers: model.Identifiers,
                       maximum_total_credentials: int,
                       maximum_node_count: int,
                       maximum_discoverable_credentials_per_action: Optional[int] = None
                       ):
        if not maximum_discoverable_credentials_per_action:
            maximum_discoverable_credentials_per_action = maximum_total_credentials
        return EnvironmentBounds(
            maximum_total_credentials=maximum_total_credentials,
            maximum_node_count=maximum_node_count,
            maximum_discoverable_credentials_per_action=maximum_discoverable_credentials_per_action,
            port_count=len(identifiers.ports),
            property_count=len(identifiers.properties),
            local_attacks_count=len(identifiers.local_vulnerabilities),
            remote_attacks_count=len(identifiers.remote_vulnerabilities)
        )


class AttackerGoal(NamedTuple):
    """Define conditions to be simultanesouly met for the attacker to win.
    If field values are not specified the default is to target full ownership
    of the network nodes.
    """
    # Include goal to reach at least the specifed cumulative total reward after
    reward: float = 0.0
    # Include goal to bring the availability to lower that the specified SLA value
    low_availability: float = 1.0
    # Include goal to own at least the specified number of nodes.
    own_atleast: int = 0
    # Include goal to own at least the specified percentage of the network nodes.
    # Set to 1.0 to define goal as the ownership of all network nodes.
    own_atleast_percent: float = 1.0


class DefenderGoal(NamedTuple):
    """Define conditions to be simultanesouly met for the defender to win."""
    # Met if attacker is evicted from all the network nodes
    eviction: bool


class DefenderConstraint(NamedTuple):
    """Define constraints to be maintained by the defender at all time."""
    maintain_sla: float


class CyberBattleEnv(gym.Env):
    """OpenAI Gym environment interface to the CyberBattle simulation.

    # Actions

        Run a local attack:            `(source_node x local_vulnerability_to_exploit)`
        Run a remote attack command:   `(source_node x target_node x remote_vulnerability_to_exploit)`
        Connect to a remote node:      `(source_node x target_node x target_port x credential_index_from_cache)`

    # Observation

       See type `Observation` for a full description of the observation space.
       It includes:
       - How many new nodes were discovered
       - Whether lateral move succeeded
       - Whether customer data were found
       - Whehter escalation attempt succeeded
       - Matrix of all node properties discovered so far
       - List of leaked credentials

    # Information
       - Action mask indicating the subset of valid actions at the current state

    # Termination

    The simulation ends if either the attacker reaches its goal (e.g. full network ownership),
    the defender reaches its goal (e.g. full eviction of the attacker)
    or if one of the defender's constraints is not met (e.g. SLA).
    """

    metadata = {'render.modes': ['human']}

    @property
    def environment(self) -> model.Environment:
        return self.__environment

    def __reset_environment(self) -> None:
        self.__environment: model.Environment = copy.deepcopy(self.__initial_environment)
        self.__discovered_nodes: List[model.NodeID] = []
        self.__owned_nodes_indices_cache: Optional[List[int]] = None
        self.__credential_cache: List[model.CachedCredential] = []
        self.__episode_rewards: List[float] = []
        # The actuator used to execute actions in the simulation environment
        self._actuator = actions.AgentActions(self.__environment)
        self._defender_actuator = actions.DefenderAgentActions(self.__environment)

        self.__stepcount = 0
        self.__start_time = time.time()
        self.__done = False

        for node_id, node_data in self.__environment.nodes():
            if node_data.agent_installed:
                self.__discovered_nodes.append(node_id)

    @property
    def name(self) -> str:
        return "CyberBattleEnv"

    @property
    def identifiers(self) -> model.Identifiers:
        return self.__environment.identifiers

    @property
    def _bounds(self) -> EnvironmentBounds:
        return self.__bounds

    def validate_environment(self, environment: model.Environment):
        """Validate that the size of the network and associated constants fits within
        the dimensions bounds set for the CyberBattle gym environment"""
        assert environment.identifiers.ports
        assert environment.identifiers.properties
        assert environment.identifiers.local_vulnerabilities
        assert environment.identifiers.remote_vulnerabilities

        node_count = len(environment.network.nodes.items())
        if node_count > self.__bounds.maximum_node_count:
            raise ValueError(f"Network node count ({node_count}) exceeds "
                             f"the specified limit of {self.__bounds.maximum_node_count}.")

        # Maximum number of credentials that can possibly be returned by any action
        effective_maximum_credentials_per_action = max([
            len(vulnerability.outcome.credentials)
            for _, node_info in environment.nodes()
            for _, vulnerability in node_info.vulnerabilities.items()
            if isinstance(vulnerability.outcome, model.LeakedCredentials)])

        if effective_maximum_credentials_per_action > self.__bounds.maximum_discoverable_credentials_per_action:
            raise ValueError(
                f"Some action in the environment returns {effective_maximum_credentials_per_action} "
                f"credentials which exceeds the maximum number of discoverable credentials "
                f"of {self.__bounds.maximum_discoverable_credentials_per_action}")

        refeerenced_ports = model.collect_ports_from_environment(environment)
        undefined_ports = set(refeerenced_ports).difference(environment.identifiers.ports)
        if undefined_ports:
            raise ValueError(f"The network has references to undefined port names: {undefined_ports}")

        referenced_properties = model.collect_properties_from_nodes(model.iterate_network_nodes(environment.network))
        undefined_properties = set(referenced_properties).difference(environment.identifiers.properties)
        if undefined_properties:
            raise ValueError(f"The network has references to undefined property names: {undefined_properties}")

        local_vulnerabilities = \
            model.collect_vulnerability_ids_from_nodes_bytype(
                environment.nodes(),
                environment.vulnerability_library,
                model.VulnerabilityType.LOCAL
            )

        undefined_local_vuln = set(local_vulnerabilities).difference(environment.identifiers.local_vulnerabilities)
        if undefined_local_vuln:
            raise ValueError(f"The network has references to undefined local"
                             f" vulnerability names: {undefined_local_vuln}")

        remote_vulnerabilities = \
            model.collect_vulnerability_ids_from_nodes_bytype(
                environment.nodes(),
                environment.vulnerability_library,
                model.VulnerabilityType.REMOTE
            )

        undefined_remote_vuln = set(remote_vulnerabilities).difference(environment.identifiers.remote_vulnerabilities)
        if undefined_remote_vuln:
            raise ValueError(f"The network has references to undefined remote"
                             f" vulnerability names: {undefined_remote_vuln}")

    # number of distinct privilege levels
    privilege_levels = model.PrivilegeLevel.MAXIMUM + 1

    def __init__(self,
                 initial_environment: model.Environment,
                 maximum_total_credentials: int = 1000,
                 maximum_node_count: int = 100,
                 maximum_discoverable_credentials_per_action: int = 5,
                 defender_agent: Optional[DefenderAgent] = None,
                 attacker_goal: Optional[AttackerGoal] = AttackerGoal(own_atleast_percent=1.0),
                 defender_goal=DefenderGoal(eviction=True),
                 defender_constraint=DefenderConstraint(maintain_sla=0.0),
                 winning_reward=5000.0,
                 losing_reward=0.0,
                 renderer=''
                 ):
        """Arguments
        ===========
        environment               - The CyberBattle network simulation environment
        maximum_total_credentials - Maximum total number of credentials used in a network
        maximum_node_count        - Largest possible size of the network
        maximum_discoverable_credentials_per_action - Maximum number of credentials returned by a given action
        attacker_goal             - Target goal for the attacker to win and stop the simulation.
        defender_goal             - Target goal for the defender to win and stop the simulation.
        defender_constraint       - Constraint to be maintain by the defender to keep the simulation running.
        winning_reward            - Reward granted to the attacker if the simulation ends because the attacker's goal is reached.
        losing_reward             - Reward granted to the attacker if the simulation ends because the Defender's goal is reached.
        renderer                  - the matplotlib renderer (e.g. 'png')
        """

        # maximum number of entities in a given environment
        self.__bounds = EnvironmentBounds.of_identifiers(
            maximum_total_credentials=maximum_total_credentials,
            maximum_node_count=maximum_node_count,
            maximum_discoverable_credentials_per_action=maximum_discoverable_credentials_per_action,
            identifiers=initial_environment.identifiers)

        self.validate_environment(initial_environment)
        self.__attacker_goal: Optional[AttackerGoal] = attacker_goal
        self.__defender_goal: DefenderGoal = defender_goal
        self.__defender_constraint: DefenderConstraint = defender_constraint
        self.__WINNING_REWARD = winning_reward
        self.__LOSING_REWARD = losing_reward
        self.__renderer = renderer

        self.viewer = None

        self.__initial_environment: model.Environment = initial_environment

        # number of entities in the environment network
        self.__defender_agent = defender_agent

        self.__reset_environment()

        self.__node_count = len(initial_environment.network.nodes.items())

        # The Space object defining the valid actions of an attacker.
        local_vulnerabilities_count = self.__bounds.local_attacks_count
        remote_vulnerabilities_count = self.__bounds.remote_attacks_count
        maximum_node_count = self.__bounds.maximum_node_count
        property_count = self.__bounds.property_count
        port_count = self.__bounds.port_count

        action_spaces: ActionSpaceDict = {
            "local_vulnerability": spaces.MultiDiscrete(
                # source_node_id, vulnerability_id
                [maximum_node_count, local_vulnerabilities_count]),
            "remote_vulnerability": spaces.MultiDiscrete(
                # source_node_id, target_node_id, vulnerability_id
                [maximum_node_count, maximum_node_count, remote_vulnerabilities_count]),
            "connect": spaces.MultiDiscrete(
                # source_node_id, target_node_id, target_port, credential_id
                # (by index of discovery: 0 for initial node, 1 for first discovered node, ...)
                [maximum_node_count, maximum_node_count, port_count, maximum_total_credentials])
        }

        self.action_space = DiscriminatedUnion(cast(dict, action_spaces))  # type: ignore

        action_mask_spaces: ActionSpaceDict = {
            "local_vulnerability":
                spaces.MultiBinary(maximum_node_count * local_vulnerabilities_count),
            "remote_vulnerability":
                spaces.MultiBinary(maximum_node_count * maximum_node_count * remote_vulnerabilities_count),
            "connect":
                spaces.MultiBinary(maximum_node_count * maximum_node_count * port_count * maximum_total_credentials)
        }

        # The observation space returning the outcome of each possible action
        self.observation_space = spaces.Dict({
            # how many new nodes were discovered
            'newly_discovered_nodes_count': spaces.Discrete(NA + maximum_node_count),
            # successuflly moved to the target node (1) or not (0)
            'lateral_move': spaces.Discrete(2),
            # boolean: 1 if customer secret data were discovered, 0 otherwise
            'customer_data_found': spaces.MultiBinary(2),
            # whether an attempted probing succeeded or not
            'probe_result': spaces.Discrete(3),
            # Esclation result
            'escalation': spaces.MultiDiscrete(model.PrivilegeLevel.MAXIMUM + 1),
            # Array of slots describing credentials that were leaked
            'leaked_credentials': spaces.Tuple(
                # the 1st component indicates if the slot is used or not (SLOT_USED or SLOT_UNSUED)
                # the 2nd component gives the credential unique index (external identifier exposed to the agent)
                # the 3rd component gives the target node ID
                # the 4th component gives the port number
                #
                #  The actual credential secret is not returned by the environment.
                #  To use the credential as a parameter to another action the agent should refer to it by its index
                #  e.g. (UNUSED_SLOT,_,_,_) encodes an empty slot
                #       (USED_SLOT,1,56,22) encodes a leaked credential identified by its index 1,
                #          that was used to authenticat to target node 56 on port number 22 (e.g. SSH)

                [spaces.MultiDiscrete([NA + 1, self.__bounds.maximum_total_credentials,
                                       maximum_node_count, port_count])]
                * self.__bounds.maximum_discoverable_credentials_per_action),

            # Boolean bitmasks defining the subset of valid actions in the current state.
            # (1 for valid, 0 for invalid). Note: a valid action is not necessariliy guaranteed to succeed.
            # For instance it is a valid action to attempt to connect to a remote node with incorrect credentials,
            # even though such action would 'fail' and potentially yield a negative reward.
            "action_mask": spaces.Dict(action_mask_spaces),

            # size of the credential stack
            'credential_cache_length': spaces.Discrete(maximum_total_credentials),

            # total nodes discovered so far
            'discovered_node_count': spaces.Discrete(maximum_node_count),

            # Matrix of properties for all the discovered nodes
            # 3 values for each matrix cell: set, unset, unknown
            'discovered_nodes_properties': spaces.MultiDiscrete([3] * maximum_node_count * property_count),

            # Escalation level on every discovered node (e.g., 0 if not owned, 1 for admin, 2 for system)
            'nodes_privilegelevel': spaces.MultiDiscrete([CyberBattleEnv.privilege_levels] * maximum_node_count),

            # Encoding of the credential cache of shape: (credential_cache_length, 2)
            #
            # Each row represent a discovered credential, the row index is the
            # the credential index is given by the row index (i.e. order of discovery)
            # A row is of the form: (target_node_discover_index, port_index)
            'credential_cache_matrix': spaces.Tuple(
                [spaces.MultiDiscrete([maximum_node_count, port_count])] * maximum_total_credentials),

            # ---------------------------------------------------------
            # Fields that were previously in the 'info' dict:
            # ---------------------------------------------------------

            # internal IDs of the credentials in the cache
            'credential_cache': DummySpace(sample=[model.CachedCredential('Sharepoint', "HTTPS", "ADPrincipalCreds")]),

            # internal IDs of nodes discovered so far
            'discovered_nodes': DummySpace(sample=['node1', 'node0', 'node2']),

            # The subgraph of nodes discovered so far with annotated edges
            # representing interactions that took place during the simulation. (See
            # actions.EdgeAnnotation)
            'explored_network': DummySpace(sample=networkx.DiGraph()),
        })

        # reward_range: A tuple corresponding to the min and max possible rewards
        self.reward_range = (-float('inf'), float('inf'))

    def __index_to_local_vulnerabilityid(self, vulnerability_index: int) -> model.VulnerabilityID:
        """Return the local vulnerability identifier from its internal encoding index"""
        return self.__initial_environment.identifiers.local_vulnerabilities[vulnerability_index]

    def __index_to_remote_vulnerabilityid(self, vulnerability_index: int) -> model.VulnerabilityID:
        """Return the remote vulnerability identifier from its internal encoding index"""
        return self.__initial_environment.identifiers.remote_vulnerabilities[vulnerability_index]

    def __index_to_port_name(self, port_index: int) -> model.PortName:
        """Return the port name identifier from its internal encoding index"""
        return self.__initial_environment.identifiers.ports[port_index]

    def __portname_to_index(self, port_name: PortName) -> int:
        """Return the internal encoding index of a given port name"""
        return self.__initial_environment.identifiers.ports.index(port_name)

    def __internal_node_id_from_external_node_index(self, node_external_index: int) -> model.NodeID:
        """"Return the internal environment node ID corresponding to the specified
        external node index that is exposed to the Gym agent
                0 -> ID of inital node
                1 -> ID of first discovered node
                ...

        """
        # Ensures that the specified node is known by the agent
        if node_external_index < 0:
            raise OutOfBoundIndexError(f"Node index must be positive, given {node_external_index}")

        length = len(self.__discovered_nodes)
        if node_external_index >= length:
            raise OutOfBoundIndexError(
                f"Node index ({node_external_index}) is invalid; only {length} nodes discovered so far.")

        node_id = self.__discovered_nodes[node_external_index]
        return node_id

    def __find_external_index(self, node_id: model.NodeID) -> int:
        """Find the external index associated with the specified node ID"""
        return self.__discovered_nodes.index(node_id)

    def __agent_owns_node(self, node_id: model.NodeID) -> bool:
        node = self.__environment.get_node(node_id)
        pwned: bool = node.agent_installed
        return pwned

    def apply_mask(self, action: Action, mask: Optional[ActionMask] = None) -> bool:
        """Apply the action mask to a specific action. Returns true just if the action
        is permitted."""
        if mask is None:
            mask = self.compute_action_mask()
        field_name = DiscriminatedUnion.kind(action)
        field_mask, coordinates = mask[field_name], action[field_name]  # type: ignore
        return bool(field_mask[tuple(coordinates)])

    def __get_blank_action_mask(self) -> ActionMask:
        """Return a blank action mask"""
        node_count = self.__node_count
        local_vulnerabilities_count = self.__bounds.local_attacks_count
        remote_vulnerabilities_count = self.__bounds.remote_attacks_count
        port_count = self.__bounds.port_count
        local = numpy.zeros(
            shape=(node_count, local_vulnerabilities_count), dtype=numpy.int32)
        remote = numpy.zeros(
            shape=(node_count, node_count, remote_vulnerabilities_count), dtype=numpy.int32)
        connect = numpy.zeros(
            shape=(node_count, node_count, port_count, self.__bounds.maximum_total_credentials), dtype=numpy.int32)
        return ActionMask(
            local_vulnerability=local,
            remote_vulnerability=remote,
            connect=connect
        )

    def __update_action_mask(self, bitmask: ActionMask) -> None:
        """Update an action mask based on the current state"""
        local_vulnerabilities_count = self.__bounds.local_attacks_count
        remote_vulnerabilities_count = self.__bounds.remote_attacks_count
        port_count = self.__bounds.port_count

        # Compute the vulnerability action bitmask
        #
        # The agent may attempt exploiting vulnerabilities
        # from any node that it owns
        for source_node_id in self.__discovered_nodes:
            if self.__agent_owns_node(source_node_id):
                source_index = self.__find_external_index(source_node_id)

                # Local: since the agent owns the node, all its local vulnerabilities are visible to it
                for vulnerability_index in range(local_vulnerabilities_count):
                    vulnerability_id = self.__index_to_local_vulnerabilityid(vulnerability_index)
                    node_vulnerable = vulnerability_id in self.__environment.vulnerability_library or \
                        vulnerability_id in self.__environment.get_node(
                            source_node_id).vulnerabilities

                    if node_vulnerable:
                        bitmask["local_vulnerability"][source_index, vulnerability_index] = 1

                # Remote: Any other node discovered so far is a potential remote target
                for target_node_id in self.__discovered_nodes:
                    target_index = self.__find_external_index(target_node_id)
                    bitmask["remote_vulnerability"][source_index,
                                                    target_index,
                                                    :remote_vulnerabilities_count] = 1

                    # the agent may attempt to connect to any port
                    # and use any credential from its cache (though it's not guaranteed to succeed)
                    bitmask["connect"][source_index,
                                       target_index,
                                       :port_count,
                                       :len(self.__credential_cache)] = 1

    def compute_action_mask(self) -> ActionMask:
        """Compute the action mask for the current state"""
        bitmask = self.__get_blank_action_mask()
        self.__update_action_mask(bitmask)
        return bitmask

    def __execute_action(self, action: Action) -> actions.ActionResult:
        # Assert that the specified action is consistent (i.e., defining a single action type)
        assert 1 == len(action.keys())

        assert DiscriminatedUnion.kind(action) != ''

        if "local_vulnerability" in action:
            source_node_index, vulnerability_index = action['local_vulnerability']

            return self._actuator.exploit_local_vulnerability(
                self.__internal_node_id_from_external_node_index(source_node_index),
                self.__index_to_local_vulnerabilityid(vulnerability_index))

        elif "remote_vulnerability" in action:
            source_node, target_node, vulnerability_index = action["remote_vulnerability"]
            source_node_id = self.__internal_node_id_from_external_node_index(source_node)
            target_node_id = self.__internal_node_id_from_external_node_index(target_node)

            result = self._actuator.exploit_remote_vulnerability(
                source_node_id,
                target_node_id,
                self.__index_to_remote_vulnerabilityid(vulnerability_index))

            return result

        elif "connect" in action:
            source_node, target_node, port_index, credential_cache_index = action["connect"]
            assert credential_cache_index >= 0
            assert credential_cache_index < len(self.__credential_cache)

            source_node_id = self.__internal_node_id_from_external_node_index(source_node)
            target_node_id = self.__internal_node_id_from_external_node_index(target_node)

            result = self._actuator.connect_to_remote_machine(
                source_node_id,
                target_node_id,
                self.__index_to_port_name(port_index),
                self.__credential_cache[credential_cache_index].credential)

            return result

        raise ValueError("Invalid discriminated union value: " + str(action))

    def __get_blank_observation(self) -> Observation:
        observation = Observation(
            newly_discovered_nodes_count=numpy.int32(0),
            leaked_credentials=tuple(
                [numpy.array([UNUSED_SLOT, 0, 0, 0], dtype=numpy.int32)]
                * self.__bounds.maximum_discoverable_credentials_per_action),
            lateral_move=numpy.int32(0),
            customer_data_found=numpy.int32(0),
            escalation=numpy.int32(PrivilegeLevel.NoAccess),
            action_mask=self.__get_blank_action_mask(),
            probe_result=numpy.int32(0),
            credential_cache_matrix=numpy.zeros((1, 2)),
            credential_cache_length=len(self.__credential_cache),
            discovered_node_count=len(self.__discovered_nodes),
            discovered_nodes_properties=numpy.array(
                [len(self.__discovered_nodes), self.__bounds.property_count], dtype=numpy.int32),
            nodes_privilegelevel=numpy.array([len(self.__discovered_nodes)], dtype=numpy.int32),

            # raw data not actually encoded as a proper gym numeric space
            # (were previously returned in the 'info' dict)
            credential_cache=self.__credential_cache,
            discovered_nodes=self.__discovered_nodes,
            explored_network=self.__get_explored_network()
        )

        return observation

    def __property_vector(self, node_id: model.NodeID, node_info: model.NodeInfo) -> numpy.ndarray:
        """Property vector for specified node
        each cell is either 1 if the property is set, -1 if unset, and 0 if unknown (node is not owned by the agent yet)
        """
        properties_indices = list(self._actuator.get_discovered_properties(node_id))

        is_owned = self._actuator.get_node_privilegelevel(node_id) >= PrivilegeLevel.LocalUser

        if is_owned:
            # if the node is owned then we know all its properties
            # => -1 should be the default value
            vector = numpy.full((self.__bounds.property_count), -1, dtype=numpy.int32)
        else:
            # otherwise we don't know anything about not discovered properties => 0 should be the default value
            vector = numpy.zeros((self.__bounds.property_count), dtype=numpy.int32)

        vector[properties_indices] = 1
        return vector

    def __get_property_matrix(self) -> numpy.ndarray:
        """Return the Node-Property matrix,
        where  1 means that the property is set for that node
              -1 if the property is not set for that node
               0 if unknown

        e.g.: [ 1 -1 -1 1 ]
                0  0  0 0
                -1 1 -1 1 ]
         1st row: properties of 1st discovered and owned node
         2nd row: no known properties for the 2nd discovered node
         3rd row: properties of 3rd discovered and owned node"""
        return numpy.array([
            self.__property_vector(node_id, node_info)
            for node_id, node_info in self._actuator.discovered_nodes()
        ], dtype=numpy.int32)

    def __get__owned_nodes_indices(self) -> List[int]:
        """Get list of indices of all owned nodes"""
        if self.__owned_nodes_indices_cache is None:
            owned_nodeids = self._actuator.get_nodes_with_atleast_privilegelevel(PrivilegeLevel.LocalUser)
            self.__owned_nodes_indices_cache = [self.__find_external_index(n) for n in owned_nodeids]

        return self.__owned_nodes_indices_cache

    def __get_privilegelevel_array(self) -> numpy.ndarray:
        """Return the node escalation level array,
        where  0 means that the node is not owned
               1 if the node is owned
               2 if the node is owned and escalated to admin
               3 if the node is owned and escalated to SYSTEM
               ... further escalation levels defined by the network
        """
        privilegelevel_array = [
            int(self._actuator.get_node_privilegelevel(node))
            for node in self.__discovered_nodes]
        return numpy.array(privilegelevel_array, dtype=numpy.int32)

    def __observation_reward_from_action_result(self, result: actions.ActionResult) -> Tuple[Observation, float]:
        obs = self.__get_blank_observation()
        outcome = result.outcome

        if isinstance(outcome, model.LeakedNodesId):
            # update discovered nodes
            newly_discovered_nodes_count = 0
            for node in outcome.nodes:
                if node not in self.__discovered_nodes:
                    self.__discovered_nodes.append(node)
                    newly_discovered_nodes_count += 1

            obs['newly_discovered_nodes_count'] = numpy.int32(newly_discovered_nodes_count)

        elif isinstance(outcome, model.LeakedCredentials):
            # update discovered nodes and credentials
            newly_discovered_nodes_count = 0
            newly_discovered_creds: List[Tuple[int, model.CachedCredential]] = []
            for cached_credential in outcome.credentials:
                if cached_credential.node not in self.__discovered_nodes:
                    self.__discovered_nodes.append(cached_credential.node)
                    newly_discovered_nodes_count += 1

                if cached_credential not in self.__credential_cache:
                    self.__credential_cache.append(cached_credential)
                    added_credential_index = len(self.__credential_cache) - 1
                    newly_discovered_creds.append((added_credential_index, cached_credential))

            obs['newly_discovered_nodes_count'] = numpy.int32(newly_discovered_nodes_count)

            # Encode the returned credentials in the format expected by the gym agent
            obs['leaked_credentials'] = tuple(
                [numpy.array([USED_SLOT,
                              cache_index,
                              self.__find_external_index(cached_credential.node),
                              self.__portname_to_index(cached_credential.port)], numpy.int32)
                 for cache_index, cached_credential in newly_discovered_creds])

        elif isinstance(outcome, model.LateralMove):
            obs['lateral_move'] = numpy.int32(1)
        elif isinstance(outcome, model.CustomerData):
            obs['customer_data_found'] = numpy.int32(1)
        elif isinstance(outcome, model.ProbeSucceeded):
            obs['probe_result'] = numpy.int32(2)
        elif isinstance(outcome, model.ProbeFailed):
            obs['probe_result'] = numpy.int32(1)
        elif isinstance(outcome, model.PrivilegeEscalation):
            obs['escalation'] = numpy.int32(outcome.level)

        x = numpy.zeros(shape=(len(self.__credential_cache), 2))
        for cache_index, cached_credential in enumerate(self.__credential_cache):
            x[cache_index] = [self.__find_external_index(cached_credential.node),
                              self.__portname_to_index(cached_credential.port)]
        obs['credential_cache_matrix'] = x

        # Dynamic statistics to be refreshed
        obs['credential_cache_length'] = len(self.__credential_cache)
        obs['credential_cache'] = self.__credential_cache
        obs['discovered_node_count'] = len(self.__discovered_nodes)
        obs['discovered_nodes'] = self.__discovered_nodes
        obs['explored_network'] = self.__get_explored_network()
        obs['discovered_nodes_properties'] = self.__get_property_matrix()
        obs['nodes_privilegelevel'] = self.__get_privilegelevel_array()

        self.__update_action_mask(obs['action_mask'])
        return obs, result.reward

    def sample_connect_action_in_expected_range(self) -> Action:
        """Sample an action of type 'connect' where the parameters
        are in the the expected ranges but not necessarily verifying
        inter-component constraints.
        """
        np_random = self.action_space.np_random
        discovered_credential_count = len(self.__credential_cache)

        if discovered_credential_count <= 0:
            raise ValueError("Cannot sample a connect action until the agent discovers more potential target nodes.")

        return Action(connect=numpy.array([
            np_random.choice(self.__get__owned_nodes_indices()),
            np_random.randint(len(self.__discovered_nodes)),
            np_random.randint(self.__bounds.port_count),
            # credential space is sparse so we force sampling
            # from the set of credentials that were discovered so far
            np_random.randint(len(self.__credential_cache))], numpy.int32))

    def sample_action_in_range(self, kinds: Optional[List[int]] = None) -> Action:
        """Sample an action in the expected component ranges but
        not necessarily verifying inter-component constraints.
        (e.g., may return a local_vulnerability action that is not
        supported by the node)

        - kinds -- A list of elements in {0,1,2} indicating what kind of
        action to sample (0:local, 1:remote, 2:connect)
        """
        np_random = self.action_space.np_random

        discovered_credential_count = len(self.__credential_cache)

        if kinds is None:
            kinds = [0, 1, 2]

        if discovered_credential_count == 0:
            # cannot generate a connect action if no cred in the cache
            kinds = [t for t in kinds if t != 2]

        assert kinds, 'Kinds list cannot be empty'

        kind = np_random.choice(kinds)

        if kind == 2:
            action = self.sample_connect_action_in_expected_range()
        elif kind == 1:
            action = Action(local_vulnerability=numpy.array([
                np_random.choice(self.__get__owned_nodes_indices()),
                np_random.randint(self.__bounds.local_attacks_count)], numpy.int32))
        else:
            action = Action(remote_vulnerability=numpy.array([
                np_random.choice(self.__get__owned_nodes_indices()),
                np_random.randint(len(self.__discovered_nodes)),
                np_random.randint(self.__bounds.remote_attacks_count)], numpy.int32))

        return action

    def is_node_owned(self, node: int):
        """Return true if a discovered node (specified by its external node index)
        is owned by the attacker agent"""
        node_id = self.__internal_node_id_from_external_node_index(node)
        node_owned = self._actuator.get_node_privilegelevel(node_id) > PrivilegeLevel.NoAccess
        return node_owned

    def is_action_valid(self, action, action_mask: Optional[ActionMask] = None) -> bool:
        """Determine if an action is valid (i.e. parameters are in expected ranges)"""
        assert 1 == len(action.keys())

        kind = DiscriminatedUnion.kind(action)
        in_range = False
        n_discovered_nodes = len(self.__discovered_nodes)
        if kind == "local_vulnerability":
            source_node, vulnerability_index = action['local_vulnerability']
            in_range = source_node < n_discovered_nodes \
                and self.is_node_owned(source_node) \
                and vulnerability_index < self.__bounds.local_attacks_count
        elif kind == "remote_vulnerability":
            source_node, target_node, vulnerability_index = action["remote_vulnerability"]
            in_range = source_node < n_discovered_nodes \
                and self.is_node_owned(source_node) \
                and target_node < n_discovered_nodes \
                and vulnerability_index < self.__bounds.remote_attacks_count
        elif kind == "connect":
            source_node, target_node, port_index, credential_cache_index = action["connect"]
            in_range = source_node < n_discovered_nodes and \
                self.is_node_owned(source_node) \
                and target_node < n_discovered_nodes \
                and port_index < self.__bounds.port_count \
                and credential_cache_index < len(self.__credential_cache)

        return in_range and self.apply_mask(action, action_mask)

    def sample_valid_action(self, kinds=None) -> Action:
        """Sample an action within the expected ranges until getting a valid one"""
        action_mask = self.compute_action_mask()
        action = self.sample_action_in_range(kinds)
        while not self.apply_mask(action, action_mask):
            action = self.sample_action_in_range(kinds)
        return action

    def sample_valid_action_with_luck(self) -> Action:
        """Sample an action until getting a valid one"""
        action_mask = self.compute_action_mask()
        action = cast(Action, self.action_space.sample())
        while not self.apply_mask(action, action_mask):
            action = cast(Action, self.action_space.sample())
        return action

    def __get_explored_network(self) -> networkx.DiGraph:
        """Returns the graph of nodes discovered so far
        with annotated edges representing interactions
        that took place during the simulation.
        """
        known_nodes = [node_id for node_id, _ in self._actuator.discovered_nodes()]
        network = self.__environment.network
        subgraph = network.subgraph(known_nodes).copy()

        # hide info for nodes that the agent does not own
        for node_id in subgraph.nodes:
            node_info: Optional[model.NodeInfo] = subgraph.nodes[node_id]['data']
            if node_info is not None:
                if not node_info.agent_installed:
                    subgraph.nodes[node_id]['data'] = None

                subgraph.nodes[node_id]['privilege_level'] = int(self._actuator.get_node_privilegelevel(node_id))
                subgraph.nodes[node_id]['flags'] = list(self._actuator.get_discovered_properties(node_id))
                subgraph.nodes[node_id]['flags_bits'] = self.__property_vector(node_id, node_info)

        return subgraph

    def __attacker_goal_reached(self) -> bool:
        goal = self.__attacker_goal

        if not goal:
            return False

        if numpy.sum(self.__episode_rewards) < goal.reward:
            return False

        nodes_owned = self.__get__owned_nodes_indices()
        owned_count = len(nodes_owned)

        if owned_count < goal.own_atleast:
            return False

        if owned_count / self.__node_count < goal.own_atleast_percent:
            return False

        if self.__defender_agent is not None and \
                self._defender_actuator.network_availability >= goal.low_availability:
            return False

        return True

    def __defender_constraints_broken(self) -> bool:
        """Check if any of the defender's constraint is not met"""
        constraint = self.__defender_constraint

        if self.__defender_agent is not None and \
                self._defender_actuator.network_availability < constraint.maintain_sla:
            return True

        return False

    def __defender_goal_reached(self) -> bool:
        """Check if defender's goal is reached(e.g. full eviction of attacker)"""
        goal = self.__defender_goal

        return goal.eviction and not (self.__get__owned_nodes_indices())

    def get_explored_network_as_numpy(self, observation: Observation) -> numpy.ndarray:
        """Return the explored network graph adjacency matrix
        as an numpy array of shape (N,N) where
        N is the number of nodes discovered so far"""
        return convert_matrix.to_numpy_array(observation['explored_network'], weight='kind_as_float')

    def get_explored_network_node_properties_bitmap_as_numpy(self, observation: Observation) -> numpy.ndarray:
        """Return a combined the matrix of adjacencies (left part) and
        node properties bitmap (right part).
        Suppose N is the number of discovered nodes and
                P is the total number of properties then
        Then the return matrix is of the form:

          ^  <---- N -----><------  P ------>
          |  (            |                 )
          N  ( Adjacency  | Node-Properties )
          |  (  Matrix    |     Bitmap      )
          V  (            |                 )

        """
        return numpy.block([convert_matrix.to_numpy_array(observation['explored_network'], weight='kind_as_float'),
                            observation['discovered_nodes_properties']])

    def step(self, action: Action) -> Tuple[Observation, float, bool, StepInfo]:
        if self.__done:
            raise RuntimeError("new episode must be started with env.reset()")

        self.__stepcount += 1
        duration = time.time() - self.__start_time
        try:
            result = self.__execute_action(action)
            observation, reward = self.__observation_reward_from_action_result(result)

            # Execute the defender step if provided
            if self.__defender_agent:
                self._defender_actuator.on_attacker_step_taken()
                self.__defender_agent.step(self.__environment, self._defender_actuator, self.__stepcount)

            self.__owned_nodes_indices_cache = None

            if self.__attacker_goal_reached() or self.__defender_constraints_broken():
                self.__done = True
                reward = self.__WINNING_REWARD
            elif self.__defender_goal_reached():
                self.__done = True
                reward = self.__LOSING_REWARD
            else:
                reward = max(0., reward)

        except OutOfBoundIndexError as error:
            logging.warning('Invalid entity index: ' + error.__str__())
            observation = self.__get_blank_observation()
            reward = 0.

        info = StepInfo(
            description='CyberBattle simulation',
            duration_in_ms=duration,
            step_count=self.__stepcount,
            network_availability=self._defender_actuator.network_availability)
        self.__episode_rewards.append(reward)

        return observation, reward, self.__done, info

    def reset(self) -> Observation:
        LOGGER.info("Resetting the CyberBattle environment")
        self.__reset_environment()
        observation = self.__get_blank_observation()
        observation['action_mask'] = self.compute_action_mask()
        observation['discovered_nodes_properties'] = self.__get_property_matrix()
        observation['nodes_privilegelevel'] = self.__get_privilegelevel_array()
        self.__owned_nodes_indices_cache = None
        return observation

    def render_as_fig(self):
        debug = commandcontrol.EnvironmentDebugging(self._actuator)
        self._actuator.print_all_attacks()

        # plot the cumulative reward and network side by side using plotly
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(go.Scatter(y=numpy.array(self.__episode_rewards).cumsum(),
                                 name='cumulative reward'), row=1, col=1)
        traces, layout = debug.network_as_plotly_traces(xref="x2", yref="y2")
        for t in traces:
            fig.add_trace(t, row=1, col=2)
        fig.update_layout(layout)
        return fig

    def render(self, mode: str = 'human') -> None:
        fig = self.render_as_fig()
        fig.show(renderer=self.__renderer)

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is None:
            self._seed = seed
            return

        self.np_random, seed = seeding.np_random(seed)

    def close(self) -> None:
        return None
