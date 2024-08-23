# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ..samples.fsec import fsecenv
from . import cyberbattle_env


class CyberBattleFsec(cyberbattle_env.CyberBattleEnv):
    """CyberBattle simulation based on a simulation for Financial network"""

    def __init__(self, **kwargs):
        super().__init__(initial_environment=fsecenv.new_environment(), **kwargs)
