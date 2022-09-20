# %%
# !pip install stable-baselines3[extra]

# %%
from typing import cast
from cyberbattle._env.cyberbattle_env import CyberBattleEnv
from cyberbattle._env.cyberbattle_toyctf import CyberBattleToyCtf
import logging
import sys
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.ppo.ppo import PPO
from cyberbattle._env.flatten_wrapper import FlattenObservationWrapper, FlattenActionWrapper
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
retrain = ['a2c']


# %%

logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")

# %%
env = CyberBattleToyCtf(
    maximum_node_count=12,
    maximum_total_credentials=10,
    observation_padding=True,
    throws_on_invalid_actions=False,
)


# %%
env1 = FlattenActionWrapper(env)

# %%
# MultiBinary
#  'action_mask',
#  'customer_data_found',
# MultiDiscrete space
#  'nodes_privilegelevel',
#  'leaked_credentials',
#  'credential_cache_matrix'
#  'discovered_nodes_properties',

ignore_fields = [
    # DummySpace
    '_credential_cache',
    '_discovered_nodes',
    '_explored_network',
]
env2 = FlattenObservationWrapper(cast(CyberBattleEnv, env1), ignore_fields=ignore_fields)

# %%
if 'a2c' in retrain:
    model_a2c = A2C("MultiInputPolicy", env2).learn(10000)
    model_a2c.save('a2c_trained_toyctf')


# %%
if 'ppo' in retrain:
    model_ppo = PPO("MultiInputPolicy", env2).learn(100)
    model_ppo.save('ppo_trained_toyctf')


# %%
model = A2C("MultiInputPolicy", env2).load('a2c_trained_toyctf')
# model = PPO("MultiInputPolicy", env2).load('ppo_trained_toyctf')


# %%
obs = env2.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env2.step(action)

env2.render()
env2.close()
