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
#     display_name: cybersim
#     language: python
#     name: cybersim
# ---

# %% [markdown] magic_args="[markdown]"
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.

# %%
# %%
import plotly.io.orca
import gymnasium as gym
import numpy
from typing import cast
from cyberbattle._env.cyberbattle_env import CyberBattleEnv

# %%
# %matplotlib inline
plotly.io.orca.config.executable = "~/.npm-packages/bin/orca"  # type: ignore

# %%
from cyberbattle._env.cyberbattle_env import AttackerGoal
from cyberbattle._env.option_wrapper import ContextWrapper, random_options

# %%
rnd = numpy.random.RandomState(13)

# %%
env = cast(CyberBattleEnv, gym.make("CyberBattleChain-v0", size=10, attacker_goal=AttackerGoal(reward=4000)))
env = ContextWrapper(env, options=random_options)

# %%
s, _ = env.reset()
env.render()

# %%
n = 10

# %%
for t in range(100):
    s, r, done, _, info = env.step()
    if r > 0:
        print(r, done, info["action"])
    #         env.render()
    if done:
        if n == 0:
            break
        n -= r > 0
        env.reset()
