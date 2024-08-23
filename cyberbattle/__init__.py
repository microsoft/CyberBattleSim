# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Initialize CyberBattleSim module"""

from gymnasium.envs.registration import registry, EnvSpec
from gymnasium.error import Error

from . import simulation
from . import agents
from ._env.cyberbattle_env import AttackerGoal, DefenderGoal
from .samples.fsec import fsecenv
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


if "FinancialNetworkSim" in registry:
    del registry["FinancialNetworkSim"]

register(
    id="FinancialNetworkSim",
    cyberbattle_env_identifiers=fsecenv.ENV_IDENTIFIERS,
    entry_point="cyberbattle._env.cyberbattle_fsecenv:CyberBattleFsec",
    kwargs={
        "defender_agent": None,
        "attacker_goal": AttackerGoal(own_atleast=6),
        "defender_goal": DefenderGoal(eviction=True)
    },
    # max_episode_steps=2600,
)
