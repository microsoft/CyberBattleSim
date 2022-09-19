# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Learner helpers and epsilon greedy search"""
import math
import sys

from .plotting import PlotTraining, plot_averaged_cummulative_rewards
from .agent_wrapper import AgentWrapper, EnvironmentBounds, Verbosity, ActionTrackingStateAugmentation
import logging
import numpy as np
from cyberbattle._env import cyberbattle_env
from typing import Tuple, Optional, TypedDict, List
import progressbar
import abc


class Learner(abc.ABC):
    """Interface to be implemented by an epsilon-greedy learner"""

    def new_episode(self) -> None:
        return None

    def end_of_episode(self, i_episode, t) -> None:
        return None

    def end_of_iteration(self, t, done) -> None:
        return None

    @abc.abstractmethod
    def explore(self, wrapped_env: AgentWrapper) -> Tuple[str, cyberbattle_env.Action, object]:
        """Exploration function.
        Returns (action_type, gym_action, action_metadata) where
        action_metadata is a custom object that gets passed to the on_step callback function"""
        raise NotImplementedError

    @abc.abstractmethod
    def exploit(self, wrapped_env: AgentWrapper, observation) -> Tuple[str, Optional[cyberbattle_env.Action], object]:
        """Exploit function.
        Returns (action_type, gym_action, action_metadata) where
        action_metadata is a custom object that gets passed to the on_step callback function"""
        raise NotImplementedError

    @abc.abstractmethod
    def on_step(self, wrapped_env: AgentWrapper, observation, reward, done, info, action_metadata) -> None:
        raise NotImplementedError

    def parameters_as_string(self) -> str:
        return ''

    def all_parameters_as_string(self) -> str:
        return ''

    def loss_as_string(self) -> str:
        return ''

    def stateaction_as_string(self, action_metadata) -> str:
        return ''


class RandomPolicy(Learner):
    """A policy that does not learn and only explore"""

    def explore(self, wrapped_env: AgentWrapper) -> Tuple[str, cyberbattle_env.Action, object]:
        gym_action = wrapped_env.env.sample_valid_action()
        return "explore", gym_action, None

    def exploit(self, wrapped_env: AgentWrapper, observation) -> Tuple[str, Optional[cyberbattle_env.Action], object]:
        raise NotImplementedError

    def on_step(self, wrapped_env: AgentWrapper, observation, reward, done, info, action_metadata):
        return None


Breakdown = TypedDict('Breakdown', {
    'local': int,
    'remote': int,
    'connect': int
})

Outcomes = TypedDict('Outcomes', {
    'reward': Breakdown,
    'noreward': Breakdown
})

Stats = TypedDict('Stats', {
    'exploit': Outcomes,
    'explore': Outcomes,
    'exploit_deflected_to_explore': int
})

TrainedLearner = TypedDict('TrainedLearner', {
    'all_episodes_rewards': List[List[float]],
    'all_episodes_availability': List[List[float]],
    'learner': Learner,
    'trained_on': str,
    'title': str
})


def print_stats(stats):
    """Print learning statistics"""
    def print_breakdown(stats, actiontype: str):
        def ratio(kind: str) -> str:
            x, y = stats[actiontype]['reward'][kind], stats[actiontype]['noreward'][kind]
            sum = x + y
            if sum == 0:
                return 'NaN'
            else:
                return f"{(x / sum):.2f}"

        def print_kind(kind: str):
            print(
                f"    {actiontype}-{kind}: {stats[actiontype]['reward'][kind]}/{stats[actiontype]['noreward'][kind]} "
                f"({ratio(kind)})")
        print_kind('local')
        print_kind('remote')
        print_kind('connect')

    print("  Breakdown [Reward/NoReward (Success rate)]")
    print_breakdown(stats, 'explore')
    print_breakdown(stats, 'exploit')
    print(f"  exploit deflected to exploration: {stats['exploit_deflected_to_explore']}")


def epsilon_greedy_search(
    cyberbattle_gym_env: cyberbattle_env.CyberBattleEnv,
    environment_properties: EnvironmentBounds,
    learner: Learner,
    title: str,
    episode_count: int,
    iteration_count: int,
    epsilon: float,
    epsilon_minimum=0.0,
    epsilon_multdecay: Optional[float] = None,
    epsilon_exponential_decay: Optional[int] = None,
    render=True,
    render_last_episode_rewards_to: Optional[str] = None,
    verbosity: Verbosity = Verbosity.Normal,
    plot_episodes_length=True
) -> TrainedLearner:
    """Epsilon greedy search for CyberBattle gym environments

    Parameters
    ==========

    - cyberbattle_gym_env -- the CyberBattle environment to train on

    - learner --- the policy learner/exploiter

    - episode_count -- Number of training episodes

    - iteration_count -- Maximum number of iterations in each episode

    - epsilon -- explore vs exploit
        - 0.0 to exploit the learnt policy only without exploration
        - 1.0 to explore purely randomly

    - epsilon_minimum -- epsilon decay clipped at this value.
    Setting this value too close to 0 may leed the search to get stuck.

    - epsilon_decay -- epsilon gets multiplied by this value after each episode

    - epsilon_exponential_decay - if set use exponential decay. The bigger the value
    is, the slower it takes to get from the initial `epsilon` to `epsilon_minimum`.

    - verbosity -- verbosity of the `print` logging

    - render -- render the environment interactively after each episode

    - render_last_episode_rewards_to -- render the environment to the specified file path
    with an index appended to it each time there is a positive reward
    for the last episode only

    - plot_episodes_length -- Plot the graph showing total number of steps by episode
    at th end of the search.

    Note on convergence
    ===================

    Setting 'minimum_espilon' to 0 with an exponential decay <1
    makes the learning converge quickly (loss function getting to 0),
    but that's just a forced convergence, however, since when
    epsilon approaches 0, only the q-values that were explored so
    far get updated and so only that subset of cells from
    the Q-matrix converges.

    """

    print(f"###### {title}\n"
          f"Learning with: episode_count={episode_count},"
          f"iteration_count={iteration_count},"
          f"ϵ={epsilon},"
          f'ϵ_min={epsilon_minimum}, '
          + (f"ϵ_multdecay={epsilon_multdecay}," if epsilon_multdecay else '')
          + (f"ϵ_expdecay={epsilon_exponential_decay}," if epsilon_exponential_decay else '') +
          f"{learner.parameters_as_string()}")

    initial_epsilon = epsilon

    all_episodes_rewards = []
    all_episodes_availability = []

    wrapped_env = AgentWrapper(cyberbattle_gym_env,
                               ActionTrackingStateAugmentation(environment_properties, cyberbattle_gym_env.reset()))
    steps_done = 0
    plot_title = f"{title} (epochs={episode_count}, ϵ={initial_epsilon}, ϵ_min={epsilon_minimum}," \
        + (f"ϵ_multdecay={epsilon_multdecay}," if epsilon_multdecay else '') \
        + (f"ϵ_expdecay={epsilon_exponential_decay}," if epsilon_exponential_decay else '') \
        + learner.parameters_as_string()
    plottraining = PlotTraining(title=plot_title, render_each_episode=render)

    render_file_index = 1

    for i_episode in range(1, episode_count + 1):

        print(f"  ## Episode: {i_episode}/{episode_count} '{title}' "
              f"ϵ={epsilon:.4f}, "
              f"{learner.parameters_as_string()}")

        observation = wrapped_env.reset()
        total_reward = 0.0
        all_rewards = []
        all_availability = []
        learner.new_episode()

        stats = Stats(exploit=Outcomes(reward=Breakdown(local=0, remote=0, connect=0),
                                       noreward=Breakdown(local=0, remote=0, connect=0)),
                      explore=Outcomes(reward=Breakdown(local=0, remote=0, connect=0),
                                       noreward=Breakdown(local=0, remote=0, connect=0)),
                      exploit_deflected_to_explore=0
                      )

        episode_ended_at = None
        sys.stdout.flush()

        bar = progressbar.ProgressBar(
            widgets=[
                'Episode ',
                f'{i_episode}',
                '|Iteration ',
                progressbar.Counter(),
                '|',
                progressbar.Variable(name='reward', width=6, precision=10),
                '|',
                progressbar.Variable(name='last_reward_at', width=4),
                '|',
                progressbar.Timer(),
                progressbar.Bar()
            ],
            redirect_stdout=False)

        for t in bar(range(1, 1 + iteration_count)):

            if epsilon_exponential_decay:
                epsilon = epsilon_minimum + math.exp(-1. * steps_done /
                                                     epsilon_exponential_decay) * (initial_epsilon - epsilon_minimum)

            steps_done += 1

            x = np.random.rand()
            if x <= epsilon:
                action_style, gym_action, action_metadata = learner.explore(wrapped_env)
            else:
                action_style, gym_action, action_metadata = learner.exploit(wrapped_env, observation)
                if not gym_action:
                    stats['exploit_deflected_to_explore'] += 1
                    _, gym_action, action_metadata = learner.explore(wrapped_env)

            # Take the step
            logging.debug(f"gym_action={gym_action}, action_metadata={action_metadata}")
            observation, reward, done, info = wrapped_env.step(gym_action)

            action_type = 'exploit' if action_style == 'exploit' else 'explore'
            outcome = 'reward' if reward > 0 else 'noreward'
            if 'local_vulnerability' in gym_action:
                stats[action_type][outcome]['local'] += 1
            elif 'remote_vulnerability' in gym_action:
                stats[action_type][outcome]['remote'] += 1
            else:
                stats[action_type][outcome]['connect'] += 1

            learner.on_step(wrapped_env, observation, reward, done, info, action_metadata)
            assert np.shape(reward) == ()

            all_rewards.append(reward)
            all_availability.append(info['network_availability'])
            total_reward += reward
            bar.update(t, reward=total_reward)
            if reward > 0:
                bar.update(t, last_reward_at=t)

            if verbosity == Verbosity.Verbose or (verbosity == Verbosity.Normal and reward > 0):
                sign = ['-', '+'][reward > 0]

                print(f"    {sign} t={t} {action_style} r={reward} cum_reward:{total_reward} "
                      f"a={action_metadata}-{gym_action} "
                      f"creds={len(observation['credential_cache_matrix'])} "
                      f" {learner.stateaction_as_string(action_metadata)}")

            if i_episode == episode_count \
                    and render_last_episode_rewards_to is not None \
                    and reward > 0:
                fig = cyberbattle_gym_env.render_as_fig()
                fig.write_image(f"{render_last_episode_rewards_to}-e{i_episode}-{render_file_index}.png")
                render_file_index += 1

            learner.end_of_iteration(t, done)

            if done:
                episode_ended_at = t
                bar.finish(dirty=True)
                break

        sys.stdout.flush()

        loss_string = learner.loss_as_string()
        if loss_string:
            loss_string = "loss={loss_string}"

        if episode_ended_at:
            print(f"  Episode {i_episode} ended at t={episode_ended_at} {loss_string}")
        else:
            print(f"  Episode {i_episode} stopped at t={iteration_count} {loss_string}")

        print_stats(stats)

        all_episodes_rewards.append(all_rewards)
        all_episodes_availability.append(all_availability)

        length = episode_ended_at if episode_ended_at else iteration_count
        learner.end_of_episode(i_episode=i_episode, t=length)
        if plot_episodes_length:
            plottraining.episode_done(length)
        if render:
            wrapped_env.render()

        if epsilon_multdecay:
            epsilon = max(epsilon_minimum, epsilon * epsilon_multdecay)

    wrapped_env.close()
    print("simulation ended")
    if plot_episodes_length:
        plottraining.plot_end()

    return TrainedLearner(
        all_episodes_rewards=all_episodes_rewards,
        all_episodes_availability=all_episodes_availability,
        learner=learner,
        trained_on=cyberbattle_gym_env.name,
        title=plot_title
    )


def transfer_learning_evaluation(
    environment_properties: EnvironmentBounds,
    trained_learner: TrainedLearner,
    eval_env: cyberbattle_env.CyberBattleEnv,
    eval_epsilon: float,
    eval_episode_count: int,
    iteration_count: int,
    benchmark_policy: Learner = RandomPolicy(),
    benchmark_training_args=dict(title="Benchmark", epsilon=1.0)
):
    """Evaluated a trained agent on another environment of different size"""

    eval_oneshot_all = epsilon_greedy_search(
        eval_env,
        environment_properties,
        learner=trained_learner['learner'],
        episode_count=eval_episode_count,  # one shot from learnt Q matric
        iteration_count=iteration_count,
        epsilon=eval_epsilon,
        render=False,
        verbosity=Verbosity.Quiet,
        title=f"One shot on {eval_env.name} - Trained on {trained_learner['trained_on']}"
    )

    eval_random = epsilon_greedy_search(
        eval_env,
        environment_properties,
        learner=benchmark_policy,
        episode_count=eval_episode_count,
        iteration_count=iteration_count,
        render=False,
        verbosity=Verbosity.Quiet,
        **benchmark_training_args
    )

    plot_averaged_cummulative_rewards(
        all_runs=[eval_oneshot_all, eval_random],
        title=f"Transfer learning {trained_learner['trained_on']}->{eval_env.name} "
        f'-- max_nodes={environment_properties.maximum_node_count}, '
        f'episodes={eval_episode_count},\n'
        f"{trained_learner['learner'].all_parameters_as_string()}")
