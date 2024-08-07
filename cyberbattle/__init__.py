# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Initialize CyberBattleSim module"""

from gymnasium.envs.registration import registry, EnvSpec
from gymnasium.error import Error

from . import simulation
from . import agents
from ._env.cyberbattle_env import AttackerGoal, DefenderGoal
from .samples.chainpattern import chainpattern
from .samples.toyctf import toy_ctf
from .samples.active_directory import generate_ad
from .simulation import generate_network, model

__all__ = (
    "simulation",
    "agents",
)


def register(id: str, cyberbattle_env_identifiers: model.Identifiers, **kwargs):
    """same as gym.envs.registry.register, but adds CyberBattle specs to env.spec"""
    if id in registry:
        raise Error("Cannot re-register id: {}".format(id))
    spec = EnvSpec(id, **kwargs)
    registry[id] = spec


if "CyberBattleToyCtf-v0" in registry:
    del registry["CyberBattleToyCtf-v0"]

register(
    id="CyberBattleToyCtf-v0",
    cyberbattle_env_identifiers=toy_ctf.ENV_IDENTIFIERS,
    entry_point="cyberbattle._env.cyberbattle_toyctf:CyberBattleToyCtf",
    kwargs={"defender_agent": None, "attacker_goal": AttackerGoal(own_atleast=6), "defender_goal": DefenderGoal(eviction=True)},
    # max_episode_steps=2600,
)

if "CyberBattleTiny-v0" in registry:
    del registry["CyberBattleTiny-v0"]

register(
    id="CyberBattleTiny-v0",
    cyberbattle_env_identifiers=toy_ctf.ENV_IDENTIFIERS,
    entry_point="cyberbattle._env.cyberbattle_tiny:CyberBattleTiny",
    kwargs={"defender_agent": None, "attacker_goal": AttackerGoal(own_atleast=6), "defender_goal": DefenderGoal(eviction=True), "maximum_total_credentials": 10, "maximum_node_count": 10},
    # max_episode_steps=2600,
)


if "CyberBattleRandom-v0" in registry:
    del registry["CyberBattleRandom-v0"]

register(
    id="CyberBattleRandom-v0",
    cyberbattle_env_identifiers=generate_network.ENV_IDENTIFIERS,
    entry_point="cyberbattle._env.cyberbattle_random:CyberBattleRandom",
)

if "CyberBattleChain-v0" in registry:
    del registry["CyberBattleChain-v0"]

register(
    id="CyberBattleChain-v0",
    cyberbattle_env_identifiers=chainpattern.ENV_IDENTIFIERS,
    entry_point="cyberbattle._env.cyberbattle_chain:CyberBattleChain",
    kwargs={"size": 4, "defender_agent": None, "attacker_goal": AttackerGoal(own_atleast_percent=1.0), "defender_goal": DefenderGoal(eviction=True), "winning_reward": 5000.0, "losing_reward": 0.0},
    reward_threshold=2200,
)

ad_envs = [f"ActiveDirectory-v{i}" for i in range(0, 10)]
for index, env in enumerate(ad_envs):
    if env in registry:
        del registry[env]

    register(
        id=env,
        cyberbattle_env_identifiers=generate_ad.ENV_IDENTIFIERS,
        entry_point="cyberbattle._env.active_directory:CyberBattleActiveDirectory",
        kwargs={
            "seed": index,
            "maximum_discoverable_credentials_per_action": 50000,
            "maximum_node_count": 30,
            "maximum_total_credentials": 50000,
        },
    )

if "ActiveDirectoryTiny-v0" in registry:
    del registry["ActiveDirectoryTiny-v0"]
register(
    id="ActiveDirectoryTiny-v0",
    cyberbattle_env_identifiers=chainpattern.ENV_IDENTIFIERS,
    entry_point="cyberbattle._env.active_directory:CyberBattleActiveDirectoryTiny",
    kwargs={"maximum_discoverable_credentials_per_action": 50000, "maximum_node_count": 30, "maximum_total_credentials": 50000},
)
