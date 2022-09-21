# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Q-learning agent applied to chain network (notebook)
This notebooks can be run directly from VSCode, to generate a
traditional Jupyter Notebook to open in your browser
 you can run the VSCode command `Export Currenty Python File As Jupyter Notebook`.
"""

# pylint: disable=invalid-name

from typing import NamedTuple, Optional, Tuple
import numpy as np
import logging

from cyberbattle._env import cyberbattle_env
from .agent_wrapper import EnvironmentBounds
from .agent_randomcredlookup import CredentialCacheExploiter
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.learner as learner


def random_argmax(array):
    """Just like `argmax` but if there are multiple elements with the max
    return a random index to break ties instead of returning the first one."""
    max_value = np.max(array)
    max_index = np.where(array == max_value)[0]

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)

    return max_value, max_index


def random_argtop_percentile(array: np.ndarray, percentile: float):
    """Just like `argmax` but if there are multiple elements with the max
    return a random index to break ties instead of returning the first one."""
    top_percentile = np.percentile(array, percentile)
    indices = np.where(array >= top_percentile)[0]

    if len(indices) == 0:
        return random_argmax(array)
    elif indices.shape[0] > 1:
        max_index = int(np.random.choice(indices, size=1))
    else:
        max_index = int(indices)

    return top_percentile, max_index


class QMatrix:
    """Q-Learning matrix for a given state and action space
        state_space  - Features defining the state space
        action_space - Features defining the action space
        qm           - Optional: initialization values for the Q matrix
    """
    # The Quality matrix
    qm: np.ndarray

    def __init__(self, name,
                 state_space: w.Feature,
                 action_space: w.Feature,
                 qm: Optional[np.ndarray] = None):
        """Initialize the Q-matrix"""

        self.name = name
        self.state_space = state_space
        self.action_space = action_space
        self.statedim = state_space.flat_size()
        self.actiondim = action_space.flat_size()
        self.qm = self.clear() if qm is None else qm

        # error calculated for the last update to the Q-matrix
        self.last_error = 0

    def shape(self):
        return (self.statedim, self.actiondim)

    def clear(self):
        """Re-initialize the Q-matrix to 0"""
        self.qm = np.zeros(shape=self.shape())
        # self.qm = np.random.rand(*self.shape()) / 100
        return self.qm

    def print(self):
        print(f"[{self.name}]\n"
              f"state: {self.state_space}\n"
              f"action: {self.action_space}\n"
              f"shape = {self.shape()}")

    def update(self, current_state: int, action: int, next_state: int, reward, gamma, learning_rate):
        """Update the Q matrix after taking `action` in state 'current_State'
        and obtaining reward=R[current_state, action]"""

        maxq_atnext, max_index = random_argmax(self.qm[next_state, ])

        # bellman equation for Q-learning
        temporal_difference = reward + gamma * maxq_atnext - self.qm[current_state, action]
        self.qm[current_state, action] += learning_rate * temporal_difference

        # The loss is calculated using the squared difference between
        # target Q-Value and predicted Q-Value
        square_error = temporal_difference * temporal_difference
        self.last_error = square_error

        return self.qm[current_state, action]

    def exploit(self, features, percentile) -> Tuple[int, float]:
        """exploit: leverage the Q-matrix.
        Returns the expected Q value and the chosen action."""
        expected_q, action = random_argtop_percentile(self.qm[features, :], percentile)
        return int(action), float(expected_q)


class QLearnAttackSource(QMatrix):
    """ Top-level Q matrix to pick the attack
        State space: global state info
        Action space: feature encodings of suggested nodes
    """

    def __init__(self, ep: EnvironmentBounds, qm: Optional[np.ndarray] = None):
        self.ep = ep

        self.state_space = w.HashEncoding(ep, [
            # Feature_discovered_node_count(),
            # Feature_discovered_credential_count(),
            w.Feature_discovered_ports_sliding(ep),
            w.Feature_discovered_nodeproperties_sliding(ep),
            w.Feature_discovered_notowned_node_count(ep, 3)
        ], 5000)  # should not be too small, pick something big to avoid collision

        self.action_space = w.RavelEncoding(ep, [
            w.Feature_active_node_properties(ep)])

        super().__init__("attack_source", self.state_space, self.action_space, qm)


class QLearnBestAttackAtSource(QMatrix):
    """ Top-level Q matrix to pick the attack from a pre-chosen source node
        State space: feature encodings of suggested node states
        Action space: a SimpleAbstract action
    """

    def __init__(self, ep: EnvironmentBounds, qm: Optional[np.ndarray] = None) -> None:

        self.state_space = w.HashEncoding(ep, [
            w.Feature_active_node_properties(ep),
            w.Feature_active_node_age(ep)
            # w.Feature_actions_tried_at_node(ep)
        ], 7000)

        # NOTE: For debugging purpose it's convenient instead to use
        # Ravel encoding for node properties
        self.state_space_debugging = w.RavelEncoding(ep, [
            w.HashEncoding(ep, [
                # Feature_discovered_node_count(),
                # Feature_discovered_credential_count(),
                w.Feature_discovered_ports_sliding(ep),
                w.Feature_discovered_nodeproperties_sliding(ep),
                w.Feature_discovered_notowned_node_count(ep, 3),
            ], 100),
            w.Feature_active_node_properties(ep)
        ])

        self.action_space = w.AbstractAction(ep)

        super().__init__("attack_at_source", self.state_space, self.action_space, qm)


# TODO: We should try scipy for sparse matrices and OpenBLAS (MKL Intel version of BLAS, faster than openBLAS) for numpy

# %%
class LossEval:
    """Loss evaluation for a Q-Learner,
    learner -- The Q learner
    """

    def __init__(self, qmatrix: QMatrix):
        self.qmatrix = qmatrix
        self.this_episode = []
        self.all_episodes = []

    def new_episode(self):
        self.this_episode = []

    def end_of_iteration(self, t, done):
        self.this_episode.append(self.qmatrix.last_error)

    def current_episode_loss(self):
        return np.average(self.this_episode)

    def end_of_episode(self, i_episode, t):
        """Average out the overall loss for this episode"""
        self.all_episodes.append(self.current_episode_loss())


class ChosenActionMetadata(NamedTuple):
    """Additional information associated with the action chosen by the agent"""
    Q_source_state: int
    Q_source_expectedq: float
    Q_attack_expectedq: float
    source_node: int
    source_node_encoding: int
    abstract_action: np.int32
    Q_attack_state: int


class QTabularLearner(learner.Learner):
    """Tabular Q-learning

    Parameters
    ==========
    gamma -- discount factor

    learning_rate -- learning rate

    ep -- environment global properties

    trained -- another QTabularLearner that is pretrained to initialize the Q matrices from (referenced, not copied)

    exploit_percentile -- (experimental) Randomly pick actions above this percentile in the Q-matrix.
    Setting 100 gives the argmax as in standard Q-learning.

    The idea is that a value less than 100 helps compensate for the
    approximation made when updating the Q-matrix caused by
    the abstraction of the action space (attack parameters are abstracted away
    in the Q-matrix, and when an abstract action is picked, it
    gets specialized via a random process.)
    When running in non-learning mode (lr=0), setting this value too close to 100
    may lead to get stuck, being more permissive (e.g. in the 80-90 range)
    typically gives better results.

    """

    def __init__(self,
                 ep: EnvironmentBounds,
                 gamma: float,
                 learning_rate: float,
                 exploit_percentile: float,
                 trained=None,  # : Optional[QTabularLearner]
                 ):
        if trained:
            self.qsource = trained.qsource
            self.qattack = trained.qattack
        else:
            self.qsource = QLearnAttackSource(ep)
            self.qattack = QLearnBestAttackAtSource(ep)

        self.loss_qsource = LossEval(self.qsource)
        self.loss_qattack = LossEval(self.qattack)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.exploit_percentile = exploit_percentile
        self.credcache_policy = CredentialCacheExploiter()

    def on_step(self, wrapped_env: w.AgentWrapper, observation, reward, done, info, action_metadata: ChosenActionMetadata):

        agent_state = wrapped_env.state

        # Update the top-level Q matrix for the state of the selected source node
        after_toplevel_state = self.qsource.state_space.encode(agent_state)
        self.qsource.update(action_metadata.Q_source_state,
                            action_metadata.source_node_encoding,
                            after_toplevel_state,
                            reward, self.gamma, self.learning_rate)

        # Update the second Q matrix for the abstract action chosen
        qattack_state_after = self.qattack.state_space.encode_at(agent_state, action_metadata.source_node)
        self.qattack.update(action_metadata.Q_attack_state,
                            int(action_metadata.abstract_action),
                            qattack_state_after,
                            reward, self.gamma, self.learning_rate)

    def end_of_iteration(self, t, done):
        self.loss_qsource.end_of_iteration(t, done)
        self.loss_qattack.end_of_iteration(t, done)

    def end_of_episode(self, i_episode, t):
        self.loss_qsource.end_of_episode(i_episode, t)
        self.loss_qattack.end_of_episode(i_episode, t)

    def loss_as_string(self):
        return f"[loss_source={self.loss_qsource.current_episode_loss():0.3f}" \
               f" loss_attack={self.loss_qattack.current_episode_loss():0.3f}]"

    def new_episode(self):
        self.loss_qsource.new_episode()
        self.loss_qattack.new_episode()

    def exploit(self, wrapped_env: w.AgentWrapper, observation):

        agent_state = wrapped_env.state

        qsource_state = self.qsource.state_space.encode(agent_state)

        #############
        # first, attempt to exploit the credential cache
        # using the crecache_policy
        action_style, gym_action, _ = self.credcache_policy.exploit(wrapped_env, observation)
        if gym_action:
            source_node = cyberbattle_env.sourcenode_of_action(gym_action)
            return action_style, gym_action, ChosenActionMetadata(
                Q_source_state=qsource_state,
                Q_source_expectedq=-1,
                Q_attack_expectedq=-1,
                source_node=source_node,
                source_node_encoding=self.qsource.action_space.encode_at(
                    agent_state, source_node),
                abstract_action=np.int32(self.qattack.action_space.abstract_from_gymaction(gym_action)),
                Q_attack_state=self.qattack.state_space.encode_at(agent_state, source_node)
            )
        #############

        # Pick action: pick random source state among the ones with the maximum Q-value
        action_style = "exploit"
        source_node_encoding, qsource_expectedq = self.qsource.exploit(
            qsource_state, percentile=100)

        # Pick source node at random (owned and with the desired feature encoding)
        potential_source_nodes = [
            from_node
            for from_node in w.owned_nodes(observation)
            if source_node_encoding == self.qsource.action_space.encode_at(agent_state, from_node)
        ]

        if len(potential_source_nodes) == 0:
            logging.debug(f'No node with encoding {source_node_encoding}, fallback on explore')
            # NOTE: we should make sure that it does not happen too often,
            # the penalty should be much smaller than typical rewards, small nudge
            # not a new feedback signal.

            # Learn the lack of node availability
            self.qsource.update(qsource_state,
                                source_node_encoding,
                                qsource_state,
                                reward=0, gamma=self.gamma, learning_rate=self.learning_rate)

            return "exploit-1->explore", None, None
        else:
            source_node = np.random.choice(potential_source_nodes)

            qattack_state = self.qattack.state_space.encode_at(agent_state, source_node)

            abstract_action, qattack_expectedq = self.qattack.exploit(
                qattack_state, percentile=self.exploit_percentile)

            gym_action = self.qattack.action_space.specialize_to_gymaction(
                source_node, observation, np.int32(abstract_action))

            assert int(abstract_action) < self.qattack.action_space.flat_size(), \
                f'abstract_action={abstract_action} gym_action={gym_action}'

            if gym_action and wrapped_env.env.is_action_valid(gym_action, observation['action_mask']):
                logging.debug(f'  exploit gym_action={gym_action} source_node_encoding={source_node_encoding}')
                return action_style, gym_action, ChosenActionMetadata(
                    Q_source_state=qsource_state,
                    Q_source_expectedq=qsource_expectedq,
                    Q_attack_expectedq=qsource_expectedq,
                    source_node=source_node,
                    source_node_encoding=source_node_encoding,
                    abstract_action=np.int32(abstract_action),
                    Q_attack_state=qattack_state
                )
            else:
                # NOTE: We should make the penalty reward smaller than
                # the average/typical non-zero reward of the env (e.g. 1/1000 smaller)
                # The idea of weighing the learning_rate when taking a chance is
                # related to "Inverse propensity weighting"

                # Learn the non-validity of the action
                self.qsource.update(qsource_state,
                                    source_node_encoding,
                                    qsource_state,
                                    reward=0, gamma=self.gamma, learning_rate=self.learning_rate)

                self.qattack.update(qattack_state,
                                    int(abstract_action),
                                    qattack_state,
                                    reward=0, gamma=self.gamma, learning_rate=self.learning_rate)

                # fallback on random exploration
                return ('exploit[invalid]->explore' if gym_action else 'exploit[undefined]->explore'), None, None

    def explore(self, wrapped_env: w.AgentWrapper):
        agent_state = wrapped_env.state
        gym_action = wrapped_env.env.sample_valid_action(kinds=[0, 1, 2])
        abstract_action = self.qattack.action_space.abstract_from_gymaction(gym_action)

        assert int(abstract_action) < self.qattack.action_space.flat_size(
        ), f'Q_attack_action={abstract_action} gym_action={gym_action}'

        source_node = cyberbattle_env.sourcenode_of_action(gym_action)

        return "explore", gym_action, ChosenActionMetadata(
            Q_source_state=self.qsource.state_space.encode(agent_state),
            Q_source_expectedq=-1,
            Q_attack_expectedq=-1,
            source_node=source_node,
            source_node_encoding=self.qsource.action_space.encode_at(agent_state, source_node),
            abstract_action=abstract_action,
            Q_attack_state=self.qattack.state_space.encode_at(agent_state, source_node)
        )

    def stateaction_as_string(self, actionmetadata) -> str:
        return f"Qsource[state={actionmetadata.Q_source_state} err={self.qsource.last_error:0.2f}"\
            f"Q={actionmetadata.Q_source_expectedq:.2f}] " \
            f"Qattack[state={actionmetadata.Q_attack_state} err={self.qattack.last_error:0.2f} "\
            f"Q={actionmetadata.Q_attack_expectedq:.2f}] "

    def parameters_as_string(self) -> str:
        return f"γ={self.gamma}," \
               f"learning_rate={self.learning_rate},"\
               f"Q%={self.exploit_percentile}"

    def all_parameters_as_string(self) -> str:
        return f' dimension={self.qsource.state_space.flat_size()}x{self.qsource.action_space.flat_size()},' \
            f'{self.qattack.state_space.flat_size()}x{self.qattack.action_space.flat_size()}\n' \
            f'Q1={[f.name() for f in self.qsource.state_space.feature_selection]}' \
            f' -> {[f.name() for f in self.qsource.action_space.feature_selection]}\n' \
            f"Q2={[f.name() for f in self.qattack.state_space.feature_selection]} -> 'action'"
