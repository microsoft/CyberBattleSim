from cyberbattle._env.cyberbattle_env import EnvironmentBounds
from typing import Optional, List
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
        return np.prod(self.nvec)

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
        # Remap to get rid of unknown value 0: 1 -> 1, and -1 -> 0 (and 0-> 0)
        assert node < len(node_prop), f'invalid node index {node} (not discovered yet)'
        remapped = np.array((1 + node_prop[node]) / 2, dtype=int)
        return remapped


class Feature_active_node_id(Feature):
    """Return the node id itself"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_node_count] * 1)

    def get(self, a: StateAugmentation, node) -> ndarray:
        return np.array([node], dtype=int)


class Feature_discovered_credential_count(Feature):
    """number of credentials discovered so far"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_total_credentials + 1])

    def get(self, a: StateAugmentation, node):
        return [len(a.observation['credential_cache_matrix'])]


class Feature_discovered_not_owned_nodes_sliding(Feature):
    """array of which of discovered nodes not owned by name"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_node_count + 1])

    def get(self, a: StateAugmentation, node):
        discovered = a.observation['discovered_nodes']
        levels = a.observation['nodes_privilegelevel']
        owned_nodes_indices = np.where(levels > 0)[0]
        owned = []
        for i in owned_nodes_indices:
            owned.append(discovered[i])
        discovered_not_owned = []
        for node in discovered:
            if node not in owned:
                discovered_not_owned.append(node)
        discovered_not_owned_sliding = np.zeros(self.env_properties.maximum_node_count, np.int32)
        for node_id in discovered_not_owned:
            if node_id == 'client':
                discovered_not_owned_sliding[0] = 1
            elif node_id == 'Website':
                discovered_not_owned_sliding[1] = 1
            elif node_id == 'Website.Directory':
                discovered_not_owned_sliding[2] = 1
            elif node_id == 'Website[user=monitor]':
                discovered_not_owned_sliding[3] = 1
            elif node_id == 'GitHubProject':
                discovered_not_owned_sliding[4] = 1
            elif node_id == 'AzureStorage':
                discovered_not_owned_sliding[5] = 1
            elif node_id == 'Sharepoint':
                discovered_not_owned_sliding[6] = 1
            elif node_id == 'AzureResourceManager':
                discovered_not_owned_sliding[7] = 1
            elif node_id == 'AzureResourceManager[user-monitor]':
                discovered_not_owned_sliding[8] = 1
            elif node_id == 'AzureVM':
                discovered_not_owned_sliding[9] = 1
        return discovered_not_owned_sliding
        
class Feature_active_node_id(Feature):
    """number asigned to each type of node in toy-ctf"""

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [p.maximum_node_count + 1])

    def get(self, a: StateAugmentation, node):
        node_id = a.observation['discovered_nodes'][node]
        
        node_id_array = np.zeros(1, np.int32)
        if node_id == 'client':
            node_id_array[0] = 0
        elif node_id == 'Website':
            node_id_array[0] = 1
        elif node_id == 'Website.Directory':
            node_id_array[0] = 2
        elif node_id == 'Website[user=monitor]':
            node_id_array[0] = 3
        elif node_id == 'GitHubProject':
            node_id_array[0] = 4
        elif node_id == 'AzureStorage':
            node_id_array[0] = 5
        elif node_id == 'Sharepoint':
            node_id_array[0] = 6
        elif node_id == 'AzureResourceManager':
            node_id_array[0] = 7
        elif node_id == 'AzureResourceManager[user-monitor]':
            node_id_array[0] = 8
        elif node_id == 'AzureVM':
            node_id_array[0] = 9
        else:
            node_id_array[0] = 10
        return node_id_array


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

    def abstract_to_gymaction(self, source_node, observation, abstract_action, target_node):
        """Takes a statring node and an abstract action number and returns a gym action"""

        if abstract_action < self.n_local_actions:
            vuln = abstract_action
            return {'local_vulnerability': np.array([source_node, vuln])}

        node_prop = observation['discovered_nodes_properties']
        abstract_action -= self.n_local_actions
        if abstract_action < self.n_remote_actions:
            vuln = abstract_action

            discovered_nodes_count = len(node_prop)
            if discovered_nodes_count <= 1:
                return None

            return {'remote_vulnerability': np.array([source_node, target_node, vuln])}

        abstract_action -= self.n_remote_actions
        port = np.int32(abstract_action)

        discovered_credentials = np.array(observation['credential_cache_matrix'])
        n_discovered_creds = len(discovered_credentials)
        if n_discovered_creds <= 0:
            return None

        nodes_not_owned = discovered_nodes_notowned(observation)
        match_port = discovered_credentials[:, 1] == port
        match_port_indicies = np.where(match_port)[0]

        credential_indices_choices = [c for c in match_port_indicies
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
        """Turns a gym action into it's abstract action number"""
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


class Feature_success_actions_at_node(Feature):
    """number of time each action succeeded at a given node"""

    max_action_count = 100

    def __init__(self, p: EnvironmentBounds):
        super().__init__(p, [self.max_action_count] * AbstractAction(p).n_actions)

    def get(self, a: ActionTrackingStateAugmentation, node: int):
        return np.minimum(a.success_action_count[node, :], self.max_action_count - 1)


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