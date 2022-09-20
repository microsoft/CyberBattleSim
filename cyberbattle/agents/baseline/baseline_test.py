# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# -*- coding: utf-8 -*-
"""Test training of baseline agents. """
import torch
import gym
import logging
import sys
import cyberbattle._env.cyberbattle_env as cyberbattle_env
from cyberbattle.agents.baseline.agent_wrapper import Verbosity
import cyberbattle.agents.baseline.agent_dql as dqla
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.learner as learner
import cyberbattle.agents.baseline.agent_tabularqlearning as tqa

logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")

print(f"torch cuda available={torch.cuda.is_available()}")

cyberbattlechain = gym.make('CyberBattleChain-v0',
                            size=4,
                            attacker_goal=cyberbattle_env.AttackerGoal(
                                own_atleast_percent=1.0,
                                reward=100))

ep = w.EnvironmentBounds.of_identifiers(
    maximum_total_credentials=10,
    maximum_node_count=10,
    identifiers=cyberbattlechain.identifiers
)

training_episode_count = 2
iteration_count = 5


def test_agent_training() -> None:
    dqn_learning_run = learner.epsilon_greedy_search(
        cyberbattle_gym_env=cyberbattlechain,
        environment_properties=ep,
        learner=dqla.DeepQLearnerPolicy(
            ep=ep,
            gamma=0.015,
            replay_memory_size=10000,
            target_update=10,
            batch_size=512,
            learning_rate=0.01),  # torch default is 1e-2
        episode_count=training_episode_count,
        iteration_count=iteration_count,
        epsilon=0.90,
        render=False,
        # epsilon_multdecay=0.75,  # 0.999,
        epsilon_exponential_decay=5000,  # 10000
        epsilon_minimum=0.10,
        verbosity=Verbosity.Quiet,
        title="DQL"
    )
    assert dqn_learning_run

    random_run = learner.epsilon_greedy_search(
        cyberbattlechain,
        ep,
        learner=learner.RandomPolicy(),
        episode_count=training_episode_count,
        iteration_count=iteration_count,
        epsilon=1.0,  # purely random
        render=False,
        verbosity=Verbosity.Quiet,
        title="Random search"
    )

    assert random_run


def test_tabularq_agent_training() -> None:
    tabularq_run = learner.epsilon_greedy_search(
        cyberbattlechain,
        ep,
        learner=tqa.QTabularLearner(
            ep,
            gamma=0.015, learning_rate=0.01, exploit_percentile=100),
        episode_count=training_episode_count,
        iteration_count=iteration_count,
        epsilon=0.90,
        epsilon_exponential_decay=5000,
        epsilon_minimum=0.01,
        verbosity=Verbosity.Quiet,
        render=False,
        plot_episodes_length=False,
        title="Tabular Q-learning"
    )

    assert tabularq_run
