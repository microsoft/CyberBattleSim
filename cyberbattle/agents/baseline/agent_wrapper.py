# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Agent wrapper for CyberBattle envrionments exposing additional
features extracted from the environment observations"""

from cyberbattle._env.cyberbattle_env import EnvironmentBounds
from typing import Optional, List, Tuple
import enum
import numpy as np
from gym import spaces, Wrapper
from numpy import ndarray
import cyberbattle._env.cyberbattle_env as cyberbattle_env
import logging


class StateAugmentation:
    """Default agent state augmentation, consisting of the gym environment
    observation itself and nothing more."""

    def __init__(self, observation: cyberbattle_env.Observation):
        self.observation = observation

    def on_step(self, action: cyberbattle_env.Action, reward: float, done: bool, observation: cyberbattle_env.Observation):
        self.observation = observation

    def on_reset(self, observation: cyberbattle_env.Observation):
        self.observation = observation


class Feature(spaces.MultiDiscrete):
    """
    Feature consisting of multiple discrete dimensions.
    Parameters:
        nvec: is a vector defining the number of possible values
        for each discrete space.
    """

    def __init__(self, env_properties: EnvironmentBounds, nvec):
        self.env_properties = env_properties
        super().__init__(nvec)

    def flat_size(self):
        return np.prod(self.nvec, dtype=int)

    def name(self):
        """Return the name of the feature"""
        p = len(type(Feature(self.env_properties, [])).__name__) + 1
        return type(self).__name__[p:]

    def get(self, a: StateAugmentation, node: Optional[int]) -> np.ndarray:
        """Compute the current value of a feature value at
        the current observation and specific node"""
        raise NotImplementedError

    def pretty_print(self, v):
        return v


class Feature_active_node_properties(Feature):
    """Bitmask of all properties set for the active node"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [2] * p.property_count)

    def get(self, a: StateAugmentation, node) -> ndarray:
        assert node is not None, 'feature only valid in the context of a node'

        node_prop = a.observation['discovered_nodes_properties']

        # list of all properties set/unset on the node
        assert node < len(node_prop), f'invalid node index {node} (not discovered yet)'

        # Remap to get rid of the unknown value (2):
        #   1->1, 0->0, 2->0
        remapped = np.array(node_prop[node] % 2, dtype=np.int_)
        return remapped


class Feature_active_node_age(Feature):
    """How recently was this node discovered?
    (measured by reverse position in the list of discovered nodes)"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_node_count])

    def get(self, a: StateAugmentation, node) -> ndarray:
        assert node is not None, 'feature only valid in the context of a node'

        discovered_node_count = a.observation['discovered_node_count']

        assert node < discovered_node_count, f'invalid node index {node} (not discovered yet)'

        return np.array([discovered_node_count - node - 1], dtype=np.int_)


class Feature_active_node_id(Feature):
    """Return the node id itself"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_node_count] * 1)

    def get(self, a: StateAugmentation, node) -> ndarray:
        return np.array([node], dtype=np.int_)


class Feature_discovered_nodeproperties_sliding(Feature):
    """Bitmask indicating node properties seen in last few cache entries"""
    window_size = 3

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [2] * p.property_count)

    def get(self, a: StateAugmentation, node) -> ndarray:
        n = a.observation['discovered_node_count']
        node_prop = np.array(a.observation['discovered_nodes_properties'])[:n]

        # keep last window of entries
        node_prop_window = node_prop[-self.window_size:, :]

        # Remap to get rid of the unknown value (2)
        node_prop_window_remapped = np.int32(node_prop_window % 2)

        countby = np.sum(node_prop_window_remapped, axis=0)

        bitmask = (countby > 0) * 1
        return bitmask


class Feature_discovered_ports(Feature):
    """Bitmask vector indicating each port seen so far in discovered credentials"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [2] * p.port_count)

    def get(self, a: StateAugmentation, node):
        n = a.observation['credential_cache_length']
        known_credports = np.zeros(self.env_properties.port_count, dtype=np.int32)
        if n > 0:
            ccm = np.array(a.observation['credential_cache_matrix'])[:n]
            known_credports[np.int32(ccm[:, 1])] = 1
        return known_credports


class Feature_discovered_ports_sliding(Feature):
    """Bitmask indicating port seen in last few cache entries"""
    window_size = 3

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [2] * p.port_count)

    def get(self, a: StateAugmentation, node):
        known_credports = np.zeros(self.env_properties.port_count, dtype=np.int32)
        n = a.observation['credential_cache_length']
        if n > 0:
            ccm = np.array(a.observation['credential_cache_matrix'])[:n]
            known_credports[np.int32(ccm[-self.window_size:, 1])] = 1
        return known_credports


class Feature_discovered_ports_counts(Feature):
    """Count of each port seen so far in discovered credentials"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_total_credentials + 1] * p.port_count)

    def get(self, a: StateAugmentation, node):
        n = a.observation['credential_cache_length']
        if n > 0:
            ccm = np.array(a.observation['credential_cache_matrix'])[:n]
            ports = np.int32(ccm[:, 1])
        else:
            ports = np.zeros(0)
        return np.bincount(ports, minlength=self.env_properties.port_count)


class Feature_discovered_credential_count(Feature):
    """number of credentials discovered so far"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_total_credentials + 1])

    def get(self, a: StateAugmentation, node):
        n = a.observation['credential_cache_length']
        return [n]


class Feature_discovered_node_count(Feature):
    """number of nodes discovered so far"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_node_count + 1])

    def get(self, a: StateAugmentation, node):
        return [a.observation['discovered_node_count']]


class Feature_discovered_notowned_node_count(Feature):
    """number of nodes discovered that are not owned yet (optionally clipped)"""

    def __init__(self, p: EnvironmentBounds, clip: Optional[int]):
        self.clip = p.maximum_node_count if clip is None else clip
        super().__init__(p, [self.clip + 1])

    def get(self, a: StateAugmentation, node):
        discovered = a.observation['discovered_node_count']
        node_props = np.array(a.observation['discovered_nodes_properties'][:discovered])
        # here we assume that a node is owned just if all its properties are known
        owned = np.count_nonzero(np.all(node_props != 2, axis=1))
        diff = discovered - owned
        return [min(diff, self.clip)]


class Feature_owned_node_count(Feature):
    """number of owned nodes so far"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_node_count + 1])

    def get(self, a: StateAugmentation, node):
        levels = a.observation['nodes_privilegelevel']
        owned_nodes_indices = np.where(levels > 0)[0]
        return [len(owned_nodes_indices)]


class ConcatFeatures(Feature):
    """ Concatenate a list of features into a single feature
    Parameters:
        feature_selection - a selection of features to combine
    """

    def __init__(self, p: EnvironmentBounds, feature_selection: List[Feature]):
        self.feature_selection = feature_selection
        self.dim_sizes = np.concatenate([f.nvec for f in feature_selection])
        super().__init__(p, [self.dim_sizes])

    def pretty_print(self, v):
        return v

    def get(self, a: StateAugmentation, node=None) -> np.ndarray:
        """Return the feature vector"""
        feature_vector = [f.get(a, node) for f in self.feature_selection]
        return np.concatenate(feature_vector)


class FeatureEncoder(Feature):
    """ Encode a list of featues as a unique index
    """

    feature_selection: List[Feature]

    def vector_to_index(self, feature_vector: np.ndarray) -> int:
        raise NotImplementedError

    def feature_vector_of_observation_at(self, a: StateAugmentation, node: Optional[int]) -> np.ndarray:
        """Return the current feature vector"""
        feature_vector = [f.get(a, node) for f in self.feature_selection]
        # print(f'feature_vector={feature_vector}  self.feature_selection={self.feature_selection}')
        return np.concatenate(feature_vector)

    def feature_vector_of_observation(self, a: StateAugmentation):
        return self.feature_vector_of_observation_at(a, None)

    def encode(self, a: StateAugmentation, node=None) -> int:
        """Return the index encoding of the feature"""
        feature_vector_concat = self.feature_vector_of_observation_at(a, node)
        return self.vector_to_index(feature_vector_concat)

    def encode_at(self, a: StateAugmentation, node) -> int:
        """Return the current feature vector encoding with a node context"""
        feature_vector_concat = self.feature_vector_of_observation_at(a, node)
        return self.vector_to_index(feature_vector_concat)

    def get(self, a: StateAugmentation, node=None) -> np.ndarray:
        """Return the feature vector"""
        return np.array([self.encode(a, node)])

    def name(self):
        """Return a name for the feature encoding"""
        n = ', '.join([f.name() for f in self.feature_selection])
        return f'[{n}]'


class HashEncoding(FeatureEncoder):
    """ Feature defined as a hash of another feature
    Parameters:
       feature_selection: a selection of features to combine
       hash_dim: dimension after hashing with hash(str(feature_vector)) or -1 for no hashing
    """

    def __init__(self, p: EnvironmentBounds, feature_selection: List[Feature], hash_size: int):
        self.feature_selection = feature_selection
        self.hash_size = hash_size
        super().__init__(p, [hash_size])

    def flat_size(self):
        return self.hash_size

    def vector_to_index(self, feature_vector) -> int:
        """Hash the state vector"""
        return hash(str(feature_vector)) % self.hash_size

    def pretty_print(self, index):
        return f'#{index}'


class RavelEncoding(FeatureEncoder):
    """ Combine a set of features into a single feature with a unique index
     (calculated by raveling the original indices)
    Parameters:
        feature_selection - a selection of features to combine
    """

    def __init__(self, p: EnvironmentBounds, feature_selection: List[Feature]):
        self.feature_selection = feature_selection
        self.dim_sizes = np.concatenate([f.nvec for f in feature_selection])
        self.ravelled_size: np.int64 = np.prod(self.dim_sizes)
        assert np.shape(self.ravelled_size) == (), f'! {np.shape(self.ravelled_size)}'
        super().__init__(p, [self.ravelled_size])

    def vector_to_index(self, feature_vector):
        assert len(self.dim_sizes) == len(feature_vector), \
            f'feature vector of size {len(feature_vector)}, ' \
            f'expecting {len(self.dim_sizes)}: {feature_vector} -- {self.dim_sizes}'
        index: np.int32 = np.ravel_multi_index(list(feature_vector), list(self.dim_sizes))
        assert index < self.ravelled_size, \
            f'feature vector out of bound ({feature_vector}, dim={self.dim_sizes}) ' \
            f'-> index={index}, max_index={self.ravelled_size-1})'
        return index

    def unravel_index(self, index) -> Tuple:
        return np.unravel_index(index, self.dim_sizes)

    def pretty_print(self, index):
        return self.unravel_index(index)


def owned_nodes(observation):
    """Return the list of owned nodes"""
    return np.nonzero(observation['nodes_privilegelevel'])[0]


def discovered_nodes_notowned(observation):
    """Return the list of discovered nodes that are not owned yet"""
    return np.nonzero(observation['nodes_privilegelevel'] == 0)[0]


class AbstractAction(Feature):
    """An abstraction of the gym state space that reduces
    the space dimension for learning use to just
        - local_attack(vulnid)    (source_node provided)
        - remote_attack(vulnid)   (source_node provided, target_node forgotten)
        - connect(port)           (source_node provided, target_node forgotten, credentials infered from cache)
    """

    def __init__(self, p: EnvironmentBounds):
        self.n_local_actions = p.local_attacks_count
        self.n_remote_actions = p.remote_attacks_count
        self.n_connect_actions = p.port_count
        self.n_actions = self.n_local_actions + self.n_remote_actions + self.n_connect_actions
        super().__init__(p, [self.n_actions])

    def specialize_to_gymaction(self, source_node: np.int32, observation, abstract_action_index: np.int32
                                ) -> Optional[cyberbattle_env.Action]:
        """Specialize an abstract "q"-action into a gym action.
        Return an adjustement weight (1.0 if the choice was deterministic, 1/n if a choice was made out of n)
        and the gym action"""

        abstract_action_index_int = int(abstract_action_index)

        discovered_nodes_count = observation['discovered_node_count']

        if abstract_action_index_int < self.n_local_actions:
            vuln = abstract_action_index_int
            return {'local_vulnerability': np.array([source_node, vuln])}

        abstract_action_index_int -= self.n_local_actions
        if abstract_action_index_int < self.n_remote_actions:
            vuln = abstract_action_index_int

            if discovered_nodes_count <= 1:
                return None

            # NOTE: We can do better here than random pick: ultimately this
            # should be learnt from target node properties

            # pick any node from the discovered ones
            # excluding the source node itself
            target = (source_node + 1 + np.random.choice(discovered_nodes_count - 1)) % discovered_nodes_count

            return {'remote_vulnerability': np.array([source_node, target, vuln])}

        abstract_action_index_int -= self.n_remote_actions
        port = np.int32(abstract_action_index_int)

        n_discovered_creds = observation['credential_cache_length']
        if n_discovered_creds <= 0:
            # no credential available in the cache: cannot poduce a valid connect action
            return None
        discovered_credentials = np.array(observation['credential_cache_matrix'])[:n_discovered_creds]

        nodes_not_owned = discovered_nodes_notowned(observation)

        # Pick a matching cred from the discovered_cred matrix
        # (at random if more than one exist for this target port)
        match_port = discovered_credentials[:, 1] == port
        match_port_indices = np.where(match_port)[0]

        credential_indices_choices = [c for c in match_port_indices
                                      if discovered_credentials[c, 0] in nodes_not_owned]

        if credential_indices_choices:
            logging.debug('found matching cred in the credential cache')
        else:
            logging.debug('no cred matching requested port, trying instead creds used to access other ports')
            credential_indices_choices = [i for (i, n) in enumerate(discovered_credentials[:, 0])
                                          if n in nodes_not_owned]

            if credential_indices_choices:
                logging.debug('found cred in the credential cache without matching port name')
            else:
                logging.debug('no cred to use from the credential cache')
                return None

        cred = np.int32(np.random.choice(credential_indices_choices))
        target = np.int32(discovered_credentials[cred, 0])
        return {'connect': np.array([source_node, target, port, cred], dtype=np.int32)}

    def abstract_from_gymaction(self, gym_action: cyberbattle_env.Action) -> np.int32:
        """Abstract a gym action into an action to be index in the Q-matrix"""
        if 'local_vulnerability' in gym_action:
            return gym_action['local_vulnerability'][1]
        elif 'remote_vulnerability' in gym_action:
            r = gym_action['remote_vulnerability']
            return self.n_local_actions + r[2]

        assert 'connect' in gym_action
        c = gym_action['connect']

        a = self.n_local_actions + self.n_remote_actions + c[2]
        assert a < self.n_actions
        return np.int32(a)


class ActionTrackingStateAugmentation(StateAugmentation):
    """An agent state augmentation consisting of
    the environment observation augmented with the following dynamic information:
       - success_action_count: count of action taken and succeeded at the current node
       - failed_action_count: count of action taken and failed at the current node
     """

    def __init__(self, p: EnvironmentBounds, observation: cyberbattle_env.Observation):
        self.aa = AbstractAction(p)
        self.success_action_count = np.zeros(shape=(p.maximum_node_count, self.aa.n_actions), dtype=np.int32)
        self.failed_action_count = np.zeros(shape=(p.maximum_node_count, self.aa.n_actions), dtype=np.int32)
        self.env_properties = p
        super().__init__(observation)

    def on_step(self, action: cyberbattle_env.Action, reward: float, done: bool, observation: cyberbattle_env.Observation):
        node = cyberbattle_env.sourcenode_of_action(action)
        abstract_action = self.aa.abstract_from_gymaction(action)
        if reward > 0:
            self.success_action_count[node, abstract_action] += 1
        else:
            self.failed_action_count[node, abstract_action] += 1
        super().on_step(action, reward, done, observation)

    def on_reset(self, observation: cyberbattle_env.Observation):
        p = self.env_properties
        self.success_action_count = np.zeros(shape=(p.maximum_node_count, self.aa.n_actions), dtype=np.int32)
        self.failed_action_count = np.zeros(shape=(p.maximum_node_count, self.aa.n_actions), dtype=np.int32)
        super().on_reset(observation)


class Feature_actions_tried_at_node(Feature):
    """A bit mask indicating which actions were already tried
    a the current node: 0 no tried, 1 tried"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [2] * AbstractAction(p).n_actions)

    def get(self, a: ActionTrackingStateAugmentation, node: int):
        return ((a.failed_action_count[node, :] + a.success_action_count[node, :]) != 0) * 1


class Feature_success_actions_at_node(Feature):
    """number of time each action succeeded at a given node"""

    max_action_count = 100

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [self.max_action_count] * AbstractAction(p).n_actions)

    def get(self, a: ActionTrackingStateAugmentation, node: int):
        return np.minimum(a.success_action_count[node, :], self.max_action_count - 1)


class Feature_failed_actions_at_node(Feature):
    """number of time each action failed at a given node"""

    max_action_count = 100

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [self.max_action_count] * AbstractAction(p).n_actions)

    def get(self, a: ActionTrackingStateAugmentation, node: int):
        return np.minimum(a.failed_action_count[node, :], self.max_action_count - 1)


class Verbosity(enum.Enum):
    """Verbosity of the learning function"""
    Quiet = 0
    Normal = 1
    Verbose = 2


class AgentWrapper(Wrapper):
    """Gym wrapper to update the agent state on every step"""

    def __init__(self, env: cyberbattle_env.CyberBattleEnv, state: StateAugmentation):
        super().__init__(env)
        self.state = state

    def step(self, action: cyberbattle_env.Action):
        observation, reward, done, info = self.env.step(action)
        self.state.on_step(action, reward, done, observation)
        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        self.state.on_reset(observation)
        return observation
