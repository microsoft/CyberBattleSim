import math
import sys

from plotting import PlotTraining, plot_averaged_cummulative_rewards
from agent_wrapper import AgentWrapper, EnvironmentBounds, Verbosity, ActionTrackingStateAugmentation
import logging
import numpy as np
from cyberbattle._env import cyberbattle_env
from typing import Tuple, Optional, TypedDict, List
import progressbar
import abc

class Agent(abc.ABC):

    @abc.abstractmethod
    def explore(self, wrapped_env: AgentWrapper) -> Tuple[cyberbattle_env.Action, object]:
        """Exploration function.
        Returns (action_type, gym_action, action_metadata) where
        action_metadata is a custom object that gets passed to the on_step callback function"""
        raise NotImplementedError

    @abc.abstractmethod
    def exploit(self, wrapped_env: AgentWrapper, observation) -> Tuple[Optional[cyberbattle_env.Action], object]:
        """Exploit function.
        Returns (action_type, gym_action, action_metadata) where
        action_metadata is a custom object that gets passed to the on_step callback function"""
        raise NotImplementedError

    @abc.abstractmethod
    def on_step(self, wrapped_env: AgentWrapper, observation, reward, done, action_metadata, epsilon) -> None:
        raise NotImplementedError

    def parameters_as_string(self) -> str:
        return ''

    def all_parameters_as_string(self) -> str:
        return ''

    def loss_as_string(self) -> str:
        return ''

    def stateaction_as_string(self, action_metadata) -> str:
        return ''

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
    'explore': Outcomes
})

TrainedAgent = TypedDict('TrainedAgent', {
    'all_episodes_rewards': List[List[float]],
    'all_episodes_availability': List[List[float]],
    'agent': Agent,
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

def epsilon_greedy_search(
    cyberbattle_gym_env: cyberbattle_env.CyberBattleEnv,
    environment_properties: EnvironmentBounds,
    agent: Agent,
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
) -> TrainedAgent:

    print(f"###### {title}\n"
          f"Learning with: episode_count={episode_count},"
          f"iteration_count={iteration_count},"
          f"ϵ={epsilon},"
          f'ϵ_min={epsilon_minimum}, '
          + (f"ϵ_multdecay={epsilon_multdecay}," if epsilon_multdecay else '')
          + (f"ϵ_expdecay={epsilon_exponential_decay}," if epsilon_exponential_decay else '') +
          f"{agent.parameters_as_string()}")

    initial_epsilon = epsilon

    all_episodes_rewards = []
    all_episodes_availability = []

    wrapped_env = AgentWrapper(cyberbattle_gym_env,
                               ActionTrackingStateAugmentation(environment_properties, cyberbattle_gym_env.reset()))
    steps_done = 0
    plot_title = f"{title} (epochs={episode_count}, ϵ={initial_epsilon}, ϵ_min={epsilon_minimum}," \
        + (f"ϵ_multdecay={epsilon_multdecay}," if epsilon_multdecay else '') \
        + (f"ϵ_expdecay={epsilon_exponential_decay}," if epsilon_exponential_decay else '') \
        + agent.parameters_as_string()
    plottraining = PlotTraining(title=plot_title, render_each_episode=render)

    render_file_index = 1

    for i_episode in range(1, episode_count + 1):

        print(f"  ## Episode: {i_episode}/{episode_count} '{title}' "
              f"ϵ={epsilon:.4f}, "
              f"{agent.parameters_as_string()}")

        observation = wrapped_env.reset()
        total_reward = 0.0
        all_rewards = []
        all_availability = []
        agent.new_episode()

        stats = Stats(exploit=Outcomes(reward=Breakdown(local=0, remote=0, connect=0),
                                       noreward=Breakdown(local=0, remote=0, connect=0)),
                      explore=Outcomes(reward=Breakdown(local=0, remote=0, connect=0),
                                       noreward=Breakdown(local=0, remote=0, connect=0))
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
                gym_action, action_metadata = agent.explore(wrapped_env)
                action_style = "explore"
            else:
                gym_action, action_metadata = agent.exploit(wrapped_env, observation)
                action_style = "exploit"

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

            agent.on_step(wrapped_env, observation, reward, done, action_metadata, epsilon)
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
                      f" {agent.stateaction_as_string(action_metadata)}")

            if i_episode == episode_count \
                    and render_last_episode_rewards_to is not None \
                    and reward > 0:
                fig = cyberbattle_gym_env.render_as_fig()
                fig.write_image(f"{render_last_episode_rewards_to}-e{i_episode}-{render_file_index}.png")
                render_file_index += 1

            agent.end_of_iteration(t, done)

            if done:
                episode_ended_at = t
                bar.finish(dirty=True)
                break

        sys.stdout.flush()

        loss_string = agent.loss_as_string()
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
        agent.end_of_episode(i_episode=i_episode, t=length)
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

    return TrainedAgent(
        all_episodes_rewards=all_episodes_rewards,
        all_episodes_availability=all_episodes_availability,
        agent=agent,
        trained_on=cyberbattle_gym_env.name,
        title=plot_title
    )