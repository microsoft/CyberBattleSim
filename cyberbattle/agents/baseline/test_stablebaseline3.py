'''Stable-baselines agent for CyberBattle Gym environment'''

import os
from typing import cast
from cyberbattle._env.cyberbattle_toyctf import CyberBattleToyCtf
from cyberbattle._env.flatten_wrapper import (
    FlattenObservationWrapper,
    FlattenActionWrapper,
)
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.a2c.a2c import A2C

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def test_stablebaseline3(training_steps=3, eval_steps=10):

    cybersinm_env = CyberBattleToyCtf(
        maximum_node_count=12,
        maximum_total_credentials=10,
        observation_padding=True,
        throws_on_invalid_actions=False,
    )

    flatten_action_env = FlattenActionWrapper(cybersinm_env)

    flatten_obs_env = FlattenObservationWrapper(flatten_action_env, ignore_fields=[
        "_credential_cache",
        "_discovered_nodes",
        "_explored_network",
    ])

    env_as_gym = cast(GymEnv, flatten_obs_env)

    check_env(flatten_obs_env)

    model_a2c = A2C("MultiInputPolicy", env_as_gym).learn(training_steps)
    model_a2c.save("a2c_trained_toyctf")
    model = A2C("MultiInputPolicy", env_as_gym).load("a2c_trained_toyctf")

    obs , _= env_as_gym.reset()
    for i in range(eval_steps):
        assert isinstance(obs, dict)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = flatten_obs_env.step(action)

    flatten_obs_env.render()
    flatten_obs_env.close()
