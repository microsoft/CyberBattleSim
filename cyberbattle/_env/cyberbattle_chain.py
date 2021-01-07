# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CyberBattle environment based on a simple chain network structure"""

from ..samples.chainpattern import chainpattern
from . import cyberbattle_env


class CyberBattleChain(cyberbattle_env.CyberBattleEnv):
    """CyberBattle environment based on a simple chain network structure"""

    def __init__(self, size, **kwargs):
        self.size = size
        super().__init__(
            initial_environment=chainpattern.new_environment(size),
            **kwargs)

    @ property
    def name(self) -> str:
        return f"CyberBattleChain-{self.size}"
