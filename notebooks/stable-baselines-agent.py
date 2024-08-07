'''Stable-baselines agent for CyberBattle Gym environment'''

# %%
from typing import cast
from cyberbattle._env.cyberbattle_toyctf import CyberBattleToyCtf
import logging
import sys
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.ppo.ppo import PPO
from cyberbattle._env.flatten_wrapper import (
    FlattenObservationWrapper,
    FlattenActionWrapper,
)
import os
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.env_checker import check_env

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
retrain = ["a2c", "ppo"]


logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")

# %%
env = CyberBattleToyCtf(
    maximum_node_count=12,
    maximum_total_credentials=10,
    observation_padding=True,
    throws_on_invalid_actions=False,
)

# %%
flatten_action_env = FlattenActionWrapper(env)

# %%
flatten_obs_env = FlattenObservationWrapper(flatten_action_env, ignore_fields=[
    # DummySpace
    "_credential_cache",
    "_discovered_nodes",
    "_explored_network",
])

#%%
env_as_gym = cast(GymEnv, flatten_obs_env)

#%%
o, _ = env_as_gym.reset()
print(o)

#%%
check_env(flatten_obs_env)


# %%
if "a2c" in retrain:
    model_a2c = A2C("MultiInputPolicy", env_as_gym).learn(10000)
    model_a2c.save("a2c_trained_toyctf")


# %%
if "ppo" in retrain:
    model_ppo = PPO("MultiInputPolicy", env_as_gym).learn(100)
    model_ppo.save("ppo_trained_toyctf")


# %%
model = A2C("MultiInputPolicy", env_as_gym).load("a2c_trained_toyctf")
# model = PPO("MultiInputPolicy", env2).load('ppo_trained_toyctf')


# %%
obs , _= env_as_gym.reset()


# %%
for i in range(1000):
    assert isinstance(obs, dict)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = flatten_obs_env.step(action)

flatten_obs_env.render()
flatten_obs_env.close()

# %%
