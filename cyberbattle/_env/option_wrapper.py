# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import NamedTuple

import gym
from gym.spaces import Space, Discrete, Tuple
import numpy as onp


class Env(NamedTuple):
    observation_space: Space
    action_space: Space


def context_spaces(observation_space, action_space):
    K = 3  # noqa: N806
    N, L = action_space.spaces['local_vulnerability'].nvec  # noqa: N806
    N, N, R = action_space.spaces['remote_vulnerability'].nvec  # noqa: N806
    N, N, P, C = action_space.spaces['connect'].nvec  # noqa: N806
    return {
        'kind': Env(observation_space, Discrete(K)),
        'local_node_id': Env(Tuple((observation_space, Discrete(K))), Discrete(N)),
        'local_vuln_id': Env(Tuple((observation_space, Discrete(N))), Discrete(L)),
        'remote_node_id': Env(Tuple((observation_space, Discrete(K), Discrete(N))), Discrete(N)),
        'remote_vuln_id': Env(Tuple((observation_space, Discrete(N), Discrete(N))), Discrete(R)),
        'cred_id': Env(observation_space, Discrete(C)),
    }


class ContextWrapper(gym.Wrapper):
    __kinds = ('local_vulnerability', 'remote_vulnerability', 'connect')

    def __init__(self, env, options):

        super().__init__(env)
        self.env = env
        assert isinstance(options, dict) and set(options) == {
            'kind', 'local_node_id', 'local_vuln_id', 'remote_node_id', 'remote_vuln_id', 'cred_id'}
        self._options = options
        self._bounds = env.bounds
        self._action_context = []

    def reset(self):
        self._action_context = onp.full(5, -1, dtype=onp.int32)
        self._observation = self.env.reset()
        return self._observation

    def step(self, dummy=None):
        obs = self._observation
        kind = self._options['kind'](obs)
        local_node_id = self._options['local_node_id']((obs, kind))
        if kind == 0:
            local_vuln_id = self._options['local_vuln_id']((obs, local_node_id))
            a = {self.__kinds[kind]: onp.array([local_node_id, local_vuln_id])}
        else:
            remote_node_id = self._options['remote_node_id']((obs, kind, local_node_id))
            if kind == 1:
                remote_vuln_id = \
                    self._options['remote_vuln_id']((obs, local_node_id, remote_node_id))
                a = {self.__kinds[kind]: onp.array([local_node_id, remote_node_id, remote_vuln_id])}
            else:
                cred_id = self._options['cred_id'](obs)
                assert cred_id < obs['credential_cache_length']
                node_id, port_id = obs['credential_cache_matrix'][cred_id].astype('int32')
                a = {self.__kinds[kind]: onp.array([local_node_id, node_id, port_id, cred_id])}

        self._observation, reward, done, info = self.env.step(a)
        return self._observation, reward, done, {**info, 'action': a}


# --- random option policies --------------------------------------------------------------------- #

def pi_kind(s):
    kinds = ('local_vulnerability', 'remote_vulnerability', 'connect')
    masked = onp.array([i for i, k in enumerate(kinds) if onp.any(s['action_mask'][k])])
    return onp.random.choice(masked)


def pi_local_node_id(s):
    s, k = s
    if k == 0:
        local_node_ids, _ = onp.argwhere(s['action_mask']['local_vulnerability']).T
    elif k == 1:
        local_node_ids, _, _ = onp.argwhere(s['action_mask']['remote_vulnerability']).T
    else:
        local_node_ids, _, _, _ = onp.argwhere(s['action_mask']['connect']).T
    return onp.random.choice(local_node_ids)


def pi_local_vuln_id(s):
    s, local_node_id = s
    local_node_ids, local_vuln_ids = onp.argwhere(s['action_mask']['local_vulnerability']).T
    masked = local_vuln_ids[local_node_ids == local_node_id]
    return onp.random.choice(masked)


def pi_remote_node_id(s):
    s, k, local_node_id = s
    assert k != 0
    if k == 1:
        local_node_ids, remote_node_ids, _ = onp.argwhere(s['action_mask']['remote_vulnerability']).T
    else:
        local_node_ids, remote_node_ids, _, _ = onp.argwhere(s['action_mask']['connect']).T
    return onp.random.choice(remote_node_ids[local_node_ids == local_node_id])


def pi_remote_vuln_id(s):
    s, local_node_id, remote_node_id = s
    local_node_ids, remote_node_ids, remote_vuln_ids = \
        onp.argwhere(s['action_mask']['remote_vulnerability']).T
    mask = (local_node_ids == local_node_id) & (remote_node_ids == remote_node_id)
    return onp.random.choice(remote_vuln_ids[mask])


def pi_cred_id(s):
    return onp.random.choice(s['credential_cache_length'])


random_options = {
    'kind': pi_kind,
    'local_node_id': pi_local_node_id,
    'local_vuln_id': pi_local_vuln_id,
    'remote_node_id': pi_remote_node_id,
    'remote_vuln_id': pi_remote_vuln_id,
    'cred_id': pi_cred_id,
}
