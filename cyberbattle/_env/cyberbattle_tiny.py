# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ..samples.toyctf import tinytoy
from . import cyberbattle_env


class CyberBattleTiny(cyberbattle_env.CyberBattleEnv):
    """CyberBattle simulation on a tiny environment. (Useful for debugging purpose)"""

    def __init__(self, **kwargs):
        super().__init__(
            initial_environment=tinytoy.new_environment(),
            **kwargs)
