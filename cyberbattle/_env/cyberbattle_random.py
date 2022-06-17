# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A CyberBattle simulation over a randomly generated network"""

from ..simulation import generate_network
from . import cyberbattle_env


class CyberBattleRandom(cyberbattle_env.CyberBattleEnv):
    """A sample CyberBattle environment"""

    def __init__(self):
        super().__init__(initial_environment=generate_network.new_environment(n_servers_per_protocol=15),
                         maximum_discoverable_credentials_per_action=32)
