# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Function DeepQLearnerPolicy.optimize_model:
#   Copyright (c) 2017, Pytorch contributors
#   All rights reserved.
#   https://github.com/pytorch/tutorials/blob/master/LICENSE

"""Deep Q-learning agent applied to chain network (notebook)
This notebooks can be run directly from VSCode, to generate a
traditional Jupyter Notebook to open in your browser
 you can run the VSCode command `Export Currenty Python File As Jupyter Notebook`.

Requirements:
    Nvidia CUDA drivers for WSL2: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
    PyTorch
"""

# pylint: disable=invalid-name

# %% [markdown]
# # Chain network CyberBattle Gym played by a Deeo Q-learning agent

# %%
from numpy import ndarray
from cyberbattle._env import cyberbattle_env
import numpy as np
from typing import List, NamedTuple, Optional, Tuple, Union
import random

# deep learning packages
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import torch.cuda
from torch.nn.utils.clip_grad import clip_grad_norm_

from .learner import Learner
from .agent_wrapper import EnvironmentBounds
import cyberbattle.agents.baseline.agent_wrapper as w
from .agent_randomcredlookup import CredentialCacheExploiter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CyberBattleStateActionModel:
    """ Define an abstraction of the state and action space
        for a CyberBattle environment, to be used to train a Q-function.
    """

    def __init__(self, ep: EnvironmentBounds):
        self.ep = ep

        self.global_features = w.ConcatFeatures(ep, [
            # w.Feature_discovered_node_count(ep),
            # w.Feature_owned_node_count(ep),
            w.Feature_discovered_notowned_node_count(ep, None)

            # w.Feature_discovered_ports(ep),
            # w.Feature_discovered_ports_counts(ep),
            # w.Feature_discovered_ports_sliding(ep),
            # w.Feature_discovered_credential_count(ep),
            # w.Feature_discovered_nodeproperties_sliding(ep),
        ])

        self.node_specific_features = w.ConcatFeatures(ep, [
            # w.Feature_actions_tried_at_node(ep),
            w.Feature_success_actions_at_node(ep),
            w.Feature_failed_actions_at_node(ep),
            w.Feature_active_node_properties(ep),
            w.Feature_active_node_age(ep)
            # w.Feature_active_node_id(ep)
        ])

        self.state_space = w.ConcatFeatures(ep, self.global_features.feature_selection +
                                            self.node_specific_features.feature_selection)

        self.action_space = w.AbstractAction(ep)

    def get_state_astensor(self, state: w.StateAugmentation):
        state_vector = self.state_space.get(state, node=None)
        state_vector_float = np.array(state_vector, dtype=np.float32)
        state_tensor = torch.from_numpy(state_vector_float).unsqueeze(0)
        return state_tensor

    def implement_action(
            self,
            wrapped_env: w.AgentWrapper,
            actor_features: ndarray,
            abstract_action: np.int32) -> Tuple[str, Optional[cyberbattle_env.Action], Optional[int]]:
        """Specialize an abstract model action into a CyberBattle gym action.

            actor_features -- the desired features of the actor to use (source CyberBattle node)
            abstract_action -- the desired type of attack (connect, local, remote).

            Returns a gym environment implementing the desired attack at a node with the desired embedding.
        """

        observation = wrapped_env.state.observation

        # Pick source node at random (owned and with the desired feature encoding)
        potential_source_nodes = [
            from_node
            for from_node in w.owned_nodes(observation)
            if np.all(actor_features == self.node_specific_features.get(wrapped_env.state, from_node))
        ]

        if len(potential_source_nodes) > 0:
            source_node = np.random.choice(potential_source_nodes)

            gym_action = self.action_space.specialize_to_gymaction(
                source_node, observation, np.int32(abstract_action))

            if not gym_action:
                return "exploit[undefined]->explore", None, None

            elif wrapped_env.env.is_action_valid(gym_action, observation['action_mask']):
                return "exploit", gym_action, source_node
            else:
                return "exploit[invalid]->explore", None, None
        else:
            return "exploit[no_actor]->explore", None, None

# %%

# Deep Q-learning


class Transition(NamedTuple):
    """One taken transition and its outcome"""
    state: Union[Tuple[Tensor], List[Tensor]]
    action: Union[Tuple[Tensor], List[Tensor]]
    next_state: Union[Tuple[Tensor], List[Tensor]]
    reward: Union[Tuple[Tensor], List[Tensor]]


class ReplayMemory(object):
    """Transition replay memory"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """The Deep Neural Network used to estimate the Q function"""

    def __init__(self, ep: EnvironmentBounds):
        super(DQN, self).__init__()

        model = CyberBattleStateActionModel(ep)
        linear_input_size = len(model.state_space.dim_sizes)
        output_size = model.action_space.flat_size()

        self.hidden_layer1 = nn.Linear(linear_input_size, 1024)
        # self.bn1 = nn.BatchNorm1d(256)
        self.hidden_layer2 = nn.Linear(1024, 512)
        self.hidden_layer3 = nn.Linear(512, 128)
        # self.hidden_layer4 = nn.Linear(128, 64)
        self.head = nn.Linear(128, output_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.hidden_layer1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.hidden_layer2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.hidden_layer3(x))
        # x = F.relu(self.hidden_layer4(x))
        return self.head(x.view(x.size(0), -1))


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


class ChosenActionMetadata(NamedTuple):
    """Additonal info about the action chosen by the DQN-induced policy"""
    abstract_action: np.int32
    actor_node: int
    actor_features: ndarray
    actor_state: ndarray

    def __repr__(self) -> str:
        return f"[abstract_action={self.abstract_action}, actor={self.actor_node}, state={self.actor_state}]"


class DeepQLearnerPolicy(Learner):
    """Deep Q-Learning on CyberBattle environments

    Parameters
    ==========
    ep -- global parameters of the environment
    model -- define a state and action abstraction for the gym environment
    gamma -- Q discount factor
    replay_memory_size -- size of the replay memory
    batch_size    -- Deep Q-learning batch
    target_update -- Deep Q-learning replay frequency (in number of episodes)
    learning_rate -- the learning rate

    Parameters from DeepDoubleQ paper
        - learning_rate = 0.00025
        - linear epsilon decay
        - gamma = 0.99

    Pytorch code from tutorial at
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self,
                 ep: EnvironmentBounds,
                 gamma: float,
                 replay_memory_size: int,
                 target_update: int,
                 batch_size: int,
                 learning_rate: float
                 ):

        self.stateaction_model = CyberBattleStateActionModel(ep)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.policy_net = DQN(ep).to(device)
        self.target_net = DQN(ep).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.target_update = target_update

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(replay_memory_size)

        self.credcache_policy = CredentialCacheExploiter()

    def parameters_as_string(self):
        return f'γ={self.gamma}, lr={self.learning_rate}, replaymemory={self.memory.capacity},\n' \
               f'batch={self.batch_size}, target_update={self.target_update}'

    def all_parameters_as_string(self) -> str:
        model = self.stateaction_model
        return f'{self.parameters_as_string()}\n' \
            f'dimension={model.state_space.flat_size()}x{model.action_space.flat_size()}, ' \
            f'Q={[f.name() for f in model.state_space.feature_selection]} ' \
            f"-> 'abstract_action'"

    def optimize_model(self, norm_clipping=False):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        # converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map((lambda s: s is not None), batch.next_state)),
                                      device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # print(f'state_batch={state_batch.shape} input={len(self.stateaction_model.state_space.dim_sizes)}')
        output = self.policy_net(state_batch)

        # print(f'output={output.shape} batch.action={transitions[0].action.shape} action_batch={action_batch.shape}')
        state_action_values = output.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if norm_clipping:
            clip_grad_norm_(self.policy_net.parameters(), 1.0)
        else:
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def get_actor_state_vector(self, global_state: ndarray, actor_features: ndarray) -> ndarray:
        return np.concatenate((np.array(global_state, dtype=np.float32),
                               np.array(actor_features, dtype=np.float32)))

    def update_q_function(self,
                          reward: float,
                          actor_state: ndarray,
                          abstract_action: np.int32,
                          next_actor_state: Optional[ndarray]):
        # store the transition in memory
        reward_tensor = torch.tensor([reward], device=device, dtype=torch.float)
        action_tensor = torch.tensor([[np.int_(abstract_action)]], device=device, dtype=torch.long)
        current_state_tensor = torch.as_tensor(actor_state, dtype=torch.float, device=device).unsqueeze(0)
        if next_actor_state is None:
            next_state_tensor = None
        else:
            next_state_tensor = torch.as_tensor(next_actor_state, dtype=torch.float, device=device).unsqueeze(0)
        self.memory.push(current_state_tensor, action_tensor, next_state_tensor, reward_tensor)

        # optimize the target network
        self.optimize_model()

    def on_step(self, wrapped_env: w.AgentWrapper,
                observation, reward: float, done: bool, info, action_metadata):
        agent_state = wrapped_env.state
        if done:
            self.update_q_function(reward,
                                   actor_state=action_metadata.actor_state,
                                   abstract_action=action_metadata.abstract_action,
                                   next_actor_state=None)
        else:
            next_global_state = self.stateaction_model.global_features.get(agent_state, node=None)
            next_actor_features = self.stateaction_model.node_specific_features.get(
                agent_state, action_metadata.actor_node)
            next_actor_state = self.get_actor_state_vector(next_global_state, next_actor_features)

            self.update_q_function(reward,
                                   actor_state=action_metadata.actor_state,
                                   abstract_action=action_metadata.abstract_action,
                                   next_actor_state=next_actor_state)

    def end_of_episode(self, i_episode, t):
        # Update the target network, copying all weights and biases in DQN
        if i_episode % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def lookup_dqn(self, states_to_consider: List[ndarray]) -> Tuple[List[np.int32], List[np.int32]]:
        """ Given a set of possible current states return:
            - index, in the provided list, of the state that would yield the best possible outcome
            - the best action to take in such a state"""
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # action: np.int32 = self.policy_net(states_to_consider).max(1)[1].view(1, 1).item()

            state_batch = torch.tensor(states_to_consider).to(device)
            dnn_output = self.policy_net(state_batch).max(1)
            action_lookups = dnn_output[1].tolist()
            expectedq_lookups = dnn_output[0].tolist()

        return action_lookups, expectedq_lookups

    def metadata_from_gymaction(self, wrapped_env, gym_action):
        current_global_state = self.stateaction_model.global_features.get(wrapped_env.state, node=None)
        actor_node = cyberbattle_env.sourcenode_of_action(gym_action)
        actor_features = self.stateaction_model.node_specific_features.get(wrapped_env.state, actor_node)
        abstract_action = self.stateaction_model.action_space.abstract_from_gymaction(gym_action)
        return ChosenActionMetadata(
            abstract_action=abstract_action,
            actor_node=actor_node,
            actor_features=actor_features,
            actor_state=self.get_actor_state_vector(current_global_state, actor_features))

    def explore(self, wrapped_env: w.AgentWrapper
                ) -> Tuple[str, cyberbattle_env.Action, object]:
        """Random exploration that avoids repeating actions previously taken in the same state"""
        # sample local and remote actions only (excludes connect action)
        gym_action = wrapped_env.env.sample_valid_action(kinds=[0, 1, 2])
        metadata = self.metadata_from_gymaction(wrapped_env, gym_action)
        return "explore", gym_action, metadata

    def try_exploit_at_candidate_actor_states(
            self,
            wrapped_env,
            current_global_state,
            actor_features,
            abstract_action):

        actor_state = self.get_actor_state_vector(current_global_state, actor_features)

        action_style, gym_action, actor_node = self.stateaction_model.implement_action(
            wrapped_env, actor_features, abstract_action)

        if gym_action:
            assert actor_node is not None, 'actor_node should be set together with gym_action'

            return action_style, gym_action, ChosenActionMetadata(
                abstract_action=abstract_action,
                actor_node=actor_node,
                actor_features=actor_features,
                actor_state=actor_state)
        else:
            # learn the failed exploit attempt in the current state
            self.update_q_function(reward=0.0,
                                   actor_state=actor_state,
                                   next_actor_state=actor_state,
                                   abstract_action=abstract_action)

            return "exploit[undefined]->explore", None, None

    def exploit(self,
                wrapped_env,
                observation
                ) -> Tuple[str, Optional[cyberbattle_env.Action], object]:

        # first, attempt to exploit the credential cache
        # using the crecache_policy
        # action_style, gym_action, _ = self.credcache_policy.exploit(wrapped_env, observation)
        # if gym_action:
        #     return action_style, gym_action, self.metadata_from_gymaction(wrapped_env, gym_action)

        # Otherwise on exploit learnt Q-function

        current_global_state = self.stateaction_model.global_features.get(wrapped_env.state, node=None)

        # Gather the features of all the current active actors (i.e. owned nodes)
        active_actors_features: List[ndarray] = [
            self.stateaction_model.node_specific_features.get(wrapped_env.state, from_node)
            for from_node in w.owned_nodes(observation)
        ]

        unique_active_actors_features: List[ndarray] = list(np.unique(active_actors_features, axis=0))

        # array of actor state vector for every possible set of node features
        candidate_actor_state_vector: List[ndarray] = [
            self.get_actor_state_vector(current_global_state, node_features)
            for node_features in unique_active_actors_features]

        remaining_action_lookups, remaining_expectedq_lookups = self.lookup_dqn(candidate_actor_state_vector)
        remaining_candidate_indices = list(range(len(candidate_actor_state_vector)))

        while remaining_candidate_indices:
            _, remaining_candidate_index = random_argmax(remaining_expectedq_lookups)
            actor_index = remaining_candidate_indices[remaining_candidate_index]
            abstract_action = remaining_action_lookups[remaining_candidate_index]

            actor_features = unique_active_actors_features[actor_index]

            action_style, gym_action, metadata = self.try_exploit_at_candidate_actor_states(
                wrapped_env,
                current_global_state,
                actor_features,
                abstract_action)

            if gym_action:
                return action_style, gym_action, metadata

            remaining_candidate_indices.pop(remaining_candidate_index)
            remaining_expectedq_lookups.pop(remaining_candidate_index)
            remaining_action_lookups.pop(remaining_candidate_index)

        return "exploit[undefined]->explore", None, None

    def stateaction_as_string(self, action_metadata) -> str:
        return ''
