# %%
from cyberbattle.simulation.environment_generation import create_random_environment
import cyberbattle.simulation.commandcontrol as commandcontrol

# %%
print(commandcontrol.__file__)


# %%
env = create_random_environment('test', 10)
# %%
env
# %%
c2 = commandcontrol.CommandControl(env)
# %%
c2.print_all_attacks()

# %%

c2.list_local_attacks('0')
# %%

# # %%
# !pip freeze

# # %%
# !python - -version
# # %%

# %%
