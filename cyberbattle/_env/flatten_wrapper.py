"""Space flattening wrappers fro the CyberBattleEnv gym environment.
"""
from collections import OrderedDict
from sqlite3 import NotSupportedError
from gym import spaces
import numpy as np
from cyberbattle._env.cyberbattle_env import DummySpace, CyberBattleEnv, Action
from gym.core import ObservationWrapper, ActionWrapper


class FlattenObservationWrapper(ObservationWrapper):
    """
    Flatten all nested dictionaries and tuples from the
     observation space of a CyberBattleSim environment`CyberBattleEnv`.
     The resulting observation space is a dictionary containing only
     subspaces of types: `Discrete`, `MultiBinary`, and `MultiDiscrete`.
    """

    def flatten_multibinary_space(self, space: spaces.Space):
        if isinstance(space, spaces.MultiBinary):
            if type(space.n) in [tuple, list, np.ndarray]:
                flatten_dim = np.multiply.reduce(space.n)
                print(f'// MultiBinary flattened from {space.n} -> {flatten_dim}')
                return spaces.MultiBinary(flatten_dim)
            else:
                print(f'// MultiBinary already flat: {space.n}')
                return space
        else:
            return space

    def __init__(self, env: CyberBattleEnv, ignore_fields=['action_mask']):
        ObservationWrapper.__init__(self, env)
        self.env = env
        self.ignore_fields = ignore_fields

        space_dict = OrderedDict({})
        for key, space in env.observation_space.spaces.items():
            if key in ignore_fields:
                print('Filtering out field', key)
            elif isinstance(space, spaces.Dict):
                for k2, subspace in space.items():
                    space_dict[f"{key}_{k2}"] = self.flatten_multibinary_space(subspace)
            elif isinstance(space, spaces.Tuple):
                for i, subspace in enumerate(space.spaces):
                    space_dict[f"{key}_{i}"] = self.flatten_multibinary_space(subspace)
            elif isinstance(space, spaces.MultiBinary):
                space_dict[key] = self.flatten_multibinary_space(space)
            elif isinstance(space, spaces.Discrete) or isinstance(space, spaces.MultiDiscrete):
                space_dict[key] = space
            elif isinstance(space, DummySpace):
                print(f'warning: unsupported observation space: {space} : {type(space)}')
            else:
                raise NotImplementedError(f"Case not handled: {key} - type {type(space)}")

        self.observation_space = spaces.Dict(space_dict)

    def flatten_multibinary_observation(self, space, o):
        if isinstance(space, spaces.MultiBinary) and \
                type(space.n) in [tuple, list, np.ndarray] and \
                len(space.n) > 1:
            flatten_dim = np.multiply.reduce(space.n)
            return tuple(o.reshape(flatten_dim))
        else:
            return o

    def observation(self, observation: dict):
        o = OrderedDict({})
        for key, space in self.env.observation_space.spaces.items():
            value = observation[key]
            if key in self.ignore_fields:
                continue
            elif isinstance(space, spaces.Dict):
                for subkey, subspace in space.items():
                    o[f"{key}_{subkey}"] = self.flatten_multibinary_observation(subspace, value[subkey])
            elif isinstance(space, spaces.Tuple):
                for i, subspace in enumerate(space.spaces):
                    o[f"{key}_{i}"] = self.flatten_multibinary_observation(subspace, value[i])
            elif isinstance(space, spaces.MultiBinary):
                o[key] = self.flatten_multibinary_observation(space, value)
            elif isinstance(space, spaces.Discrete) or isinstance(space, spaces.MultiDiscrete):
                o[key] = value
            elif isinstance(space, DummySpace):
                continue
            else:
                raise NotImplementedError(f"Case not handled: {key} - type {type(space)}")

        return o


class FlattenActionWrapper(ActionWrapper):
    """
    Flatten all nested dictionaries and tuples from the
     action space of a CyberBattleSim environment`CyberBattleEnv`.
     The resulting action space is a dictionary containing only
     subspaces of types: `Discrete`, `MultiBinary`, and `MultiDiscrete`.
    """

    def __init__(self, env: CyberBattleEnv):
        ActionWrapper.__init__(self, env)
        self.env = env

        self.action_space = spaces.MultiDiscrete([
            # connect, local vulnerabilities, remote vulnerabilities
            1 + env.bounds.local_attacks_count + env.bounds.remote_attacks_count,

            # source node
            env.bounds.maximum_node_count,

            # target  node
            env.bounds.maximum_node_count,

            # target port (for connect action only)
            env.bounds.port_count,

            # target port (credentials used, for connect action only)
            env.bounds.maximum_total_credentials
        ]
        )

    def action(self, action: np.ndarray) -> Action:
        action_type = action[0]
        if action_type == 0:
            return {'connect': action[1:5]}

        action_type -= 1
        if action_type < self.env.bounds.local_attacks_count:
            return {'local_vulnerability': np.array([action[1], action_type])}

        action_type -= self.env.bounds.local_attacks_count
        if action_type < self.env.bounds.remote_attacks_count:
            return {'remote_vulnerability': np.array([action[1], action[2], action_type])}

        raise NotSupportedError(f'Unsupported action: {action}')

    def reverse_action(self, action):
        raise NotImplementedError
