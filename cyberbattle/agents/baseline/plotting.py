# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Plotting helpers for agent banchmarking"""

import matplotlib.pyplot as plt  # type:ignore
import numpy as np

import matplotlib  # type: ignore

matplotlib.use("Agg")


def new_plot(title):
    """Prepare a new plot of cumulative rewards"""
    plt.figure(figsize=(10, 8))
    plt.ylabel("cumulative reward", fontsize=20)
    plt.xlabel("step", fontsize=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.title(title, fontsize=12)


def pad(array, length):
    """Pad an array with 0s to make it of desired length"""
    padding = np.zeros((length,))
    padding[: len(array)] = array
    return padding


def plot_episodes_rewards_averaged(results):
    """Plot cumulative rewards for a given set of specified episodes"""
    max_iteration_count = np.max([len(r) for r in results["all_episodes_rewards"]])

    all_episodes_rewards_padded = [
        pad(rewards, max_iteration_count) for rewards in results["all_episodes_rewards"]
    ]
    cumrewards = np.cumsum(all_episodes_rewards_padded, axis=1)
    avg = np.average(cumrewards, axis=0)
    std = np.std(cumrewards, axis=0)
    x = [i for i in range(len(std))]
    plt.plot(x, avg, label=results["title"])
    plt.fill_between(x, avg - std, avg + std, alpha=0.5)


def fill_with_latest_value(array, length):
    pad = length - len(array)
    if pad > 0:
        return np.pad(array, (0, pad), mode="edge")
    else:
        return array


def plot_episodes_availability_averaged(results):
    """Plot availability for a given set of specified episodes"""
    data = results["all_episodes_availability"]
    longest_episode_length = np.max([len(r) for r in data])

    all_episodes_padded = [
        fill_with_latest_value(av, longest_episode_length) for av in data
    ]
    avg = np.average(all_episodes_padded, axis=0)
    std = np.std(all_episodes_padded, axis=0)
    x = [i for i in range(len(std))]
    plt.plot(x, avg, label=results["title"])
    plt.fill_between(x, avg - std, avg + std, alpha=0.5)


def plot_episodes_length(learning_results):
    """Plot length of every episode"""
    plt.figure(figsize=(10, 8))
    plt.ylabel("#iterations", fontsize=20)
    plt.xlabel("episode", fontsize=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.title("Length of each episode", fontsize=12)

    for results in learning_results:
        iterations = [len(e) for e in results["all_episodes_rewards"]]
        episode = [i for i in range(len(results["all_episodes_rewards"]))]
        plt.plot(episode, iterations, label=f"{results['title']}")

    plt.legend(loc="upper right")
    plt.show()


def plot_each_episode(results):
    """Plot cumulative rewards for each episode"""
    for i, episode in enumerate(results["all_episodes_rewards"]):
        cumrewards = np.cumsum(episode)
        x = [i for i in range(len(cumrewards))]
        plt.plot(x, cumrewards, label=f"Episode {i}")


def plot_all_episodes(r):
    """Plot cumulative rewards for every episode"""
    new_plot(r["title"])
    plot_each_episode(r)
    plt.legend(loc="lower right")
    plt.show()


def plot_averaged_cummulative_rewards(title, all_runs, show=True):
    """Plot averaged cumulative rewards"""
    new_plot(title)
    for r in all_runs:
        plot_episodes_rewards_averaged(r)
    plt.legend(loc="lower right")
    if show:
        plt.show()


def plot_averaged_availability(title, all_runs, show=False):
    """Plot averaged network availability"""
    plt.figure(figsize=(10, 8))
    plt.ylabel("network availability", fontsize=20)
    plt.xlabel("step", fontsize=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.title(title, fontsize=12)
    for r in all_runs:
        plot_episodes_availability_averaged(r)
    plt.legend(loc="lower right")
    if show:
        plt.show()


def new_plot_loss():
    """Plot MSE loss averaged over all episodes"""
    plt.figure(figsize=(10, 8))
    plt.ylabel("loss", fontsize=20)
    plt.xlabel("episodes", fontsize=20)
    plt.xticks(size=12)
    plt.yticks(size=20)
    plt.title("Loss", fontsize=12)


def plot_all_episodes_loss(all_episodes_losses, name, label):
    """Plot loss for one learning episode"""
    x = [i for i in range(len(all_episodes_losses))]
    plt.plot(x, all_episodes_losses, label=f"{name} {label}")


def running_mean(x, size):
    """return moving average of x for a window of lenght 'size'"""
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return np.subtract(cumsum[size:], cumsum[:-size]) / float(size)


class PlotTraining:
    """Plot training-related stats"""

    def __init__(self, title, render_each_episode):
        self.episode_durations = []
        self.title = title
        self.render_each_episode = render_each_episode

    def plot_durations(self, average_window=5):
        # plt.figure(2)
        plt.figure()
        # plt.clf()
        durations_t = np.array(self.episode_durations, dtype=np.float32)
        plt.title("Training...")
        plt.xlabel("Episode")
        plt.ylabel("Duration")
        plt.title(self.title, fontsize=12)

        episodes = [i + 1 for i in range(len(self.episode_durations))]
        plt.plot(episodes, durations_t)
        # plot episode running averages
        if len(durations_t) >= average_window:
            means = running_mean(durations_t, average_window)
            means = np.concatenate((np.zeros(average_window - 1), means))
            plt.plot(episodes, means)

        # display.display(plt.gcf())
        plt.show()

    def episode_done(self, length):
        self.episode_durations.append(length)
        if self.render_each_episode:
            self.plot_durations()

    def plot_end(self):
        self.plot_durations()
        plt.ioff()  # type: ignore
        # plt.show()


def length_of_all_episodes(run):
    """Get the length of every episode"""
    return [len(e) for e in run["all_episodes_rewards"]]


def reduce(x, desired_width):
    return [np.average(c) for c in np.array_split(x, desired_width)]


def episodes_rewards_averaged(run):
    """Plot cumulative rewards for a given set of specified episodes"""
    max_iteration_count = np.max([len(r) for r in run["all_episodes_rewards"]])
    all_episodes_rewards_padded = [
        pad(rewards, max_iteration_count) for rewards in run["all_episodes_rewards"]
    ]
    cumrewards = np.cumsum(all_episodes_rewards_padded, axis=1)
    avg = np.average(cumrewards, axis=0)
    return list(avg)


def episodes_lengths_for_all_runs(all_runs):
    return [length_of_all_episodes(run) for run in all_runs]


def averaged_cummulative_rewards(all_runs, width):
    return [reduce(episodes_rewards_averaged(run), width) for run in all_runs]
