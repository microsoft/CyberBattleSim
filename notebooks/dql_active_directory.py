# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DQL agent running on the Active Directory sample environment

# %%
import logging, sys
import gymnasium as gym
import cyberbattle.agents.baseline.learner as learner
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.agent_dql as dqla
from cyberbattle.agents.baseline.agent_wrapper import ActionTrackingStateAugmentation, AgentWrapper, Verbosity

logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")
# %matplotlib inline

# %% tags=["parameters"]
ngyms = 9
iteration_count = 1000

# %%
gymids = [f"ActiveDirectory-v{i}" for i in range(0, ngyms)]

# %%
from typing import cast
from cyberbattle._env.cyberbattle_env import CyberBattleEnv

envs = [cast(CyberBattleEnv, gym.make(gymid).unwrapped) for gymid in gymids]
map(lambda g: g.reset(seed=1), envs)
ep = w.EnvironmentBounds.of_identifiers(maximum_node_count=30, maximum_total_credentials=50, identifiers=envs[0].identifiers)

# %%
# Evaluate the Deep Q-learning agent for each env using transfer learning
_l = dqla.DeepQLearnerPolicy(
    ep=ep,
    gamma=0.015,
    replay_memory_size=10000,
    target_update=5,
    batch_size=512,
    learning_rate=0.01,  # torch default learning rate is 1e-2
)
for i, env in enumerate(envs):
    epsilon = (10 - i) / 10
    # at least 1 runs and max 10 for the 10 envs
    training_episode_count = 1 + (9 - i)
    dqn_learning_run = learner.epsilon_greedy_search(
        cyberbattle_gym_env=env,
        environment_properties=ep,
        learner=_l,
        episode_count=training_episode_count,
        iteration_count=iteration_count,
        epsilon=epsilon,
        epsilon_exponential_decay=50000,
        epsilon_minimum=0.1,
        verbosity=Verbosity.Quiet,
        render=False,
        plot_episodes_length=False,
        title=f"DQL {i}",
    )
    _l = dqn_learning_run["learner"]

# %%
tiny = cast(CyberBattleEnv, gym.make(f"ActiveDirectory-v{ngyms}"))
current_o, _ = tiny.reset()
tiny.reset(seed=1)
wrapped_env = AgentWrapper(tiny, ActionTrackingStateAugmentation(ep, current_o))
# Use the trained agent to run the steps one by one
max_steps = 1000
# next action suggested by DQL agent
# h = []
for i in range(max_steps):
    # run the suggested action
    _, next_action, _ = _l.exploit(wrapped_env, current_o)
    # h.append((tiny.get_explored_network_node_properties_bitmap_as_numpy(current_o), next_action))
    if next_action is None:
        print("No more learned moves")
        break
    current_o, _, is_done, _, _ = wrapped_env.step(next_action)
    if is_done:
        print("Finished simulation")
        break
tiny.render()
