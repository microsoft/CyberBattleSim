# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Random agent with credential lookup (notebook)
"""

# pylint: disable=invalid-name

from .agent_wrapper import AgentWrapper
from .learner import Learner
from typing import Optional
import cyberbattle._env.cyberbattle_env as cyberbattle_env
import numpy as np
import logging
import cyberbattle.agents.baseline.agent_wrapper as w


def exploit_credentialcache(observation) -> Optional[cyberbattle_env.Action]:
    """Exploit the credential cache to connect to
    a node not owned yet."""

    # Pick source node at random (owned and with the desired feature encoding)
    potential_source_nodes = w.owned_nodes(observation)
    if len(potential_source_nodes) == 0:
        return None

    source_node = np.random.choice(potential_source_nodes)

    discovered_credentials = np.array(observation['credential_cache_matrix'])
    n_discovered_creds = len(discovered_credentials)
    if n_discovered_creds <= 0:
        # no credential available in the cache: cannot poduce a valid connect action
        return None

    nodes_not_owned = w.discovered_nodes_notowned(observation)

    match_port__target_notowned = [c for c in range(n_discovered_creds)
                                   if discovered_credentials[c, 0] in nodes_not_owned]

    if match_port__target_notowned:
        logging.debug('found matching cred in the credential cache')
        cred = np.int32(np.random.choice(match_port__target_notowned))
        target = np.int32(discovered_credentials[cred, 0])
        port = np.int32(discovered_credentials[cred, 1])
        return {'connect': np.array([source_node, target, port, cred], dtype=np.int32)}
    else:
        return None


class CredentialCacheExploiter(Learner):
    """A learner that just exploits the credential cache"""

    def parameters_as_string(self):
        return ''

    def explore(self, wrapped_env: AgentWrapper):
        return "explore", wrapped_env.env.sample_valid_action([0, 1]), None

    def exploit(self, wrapped_env: AgentWrapper, observation):
        gym_action = exploit_credentialcache(observation)
        if gym_action:
            if wrapped_env.env.is_action_valid(gym_action, observation['action_mask']):
                return 'exploit', gym_action, None
            else:
                # fallback on random exploration
                return 'exploit[invalid]->explore', None, None
        else:
            return 'exploit[undefined]->explore', None, None

    def stateaction_as_string(self, actionmetadata):
        return ''

    def on_step(self, wrapped_env: AgentWrapper, observation, reward, done, info, action_metadata):
        return None

    def end_of_iteration(self, t, done):
        return None

    def end_of_episode(self, i_episode, t):
        return None

    def loss_as_string(self):
        return ''

    def new_episode(self):
        return None
