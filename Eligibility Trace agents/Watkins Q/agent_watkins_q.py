import random
from typing import NamedTuple, Optional, Tuple, Union, List
import numpy as np
from numpy import ndarray
import logging
import boolean

from cyberbattle._env import cyberbattle_env
from agent_wrapper import EnvironmentBounds, discovered_nodes_notowned
from gym import spaces, Wrapper

import agent_wrapper as w
from learner import Learner

from torch import Tensor
import torch

class StateActionModel:

    def __init__(self, ep: EnvironmentBounds):
        self.ep = ep

        self.global_features = w.ConcatFeatures(ep, [
            w.Feature_discovered_not_owned_nodes_sliding(ep),
            w.Feature_discovered_credential_count(ep)
        ])

        self.source_node_features = w.ConcatFeatures(ep, [
            w.Feature_active_node_properties(ep),
            w.Feature_success_actions_at_node(ep)
        ])

        self.target_node_features = w.ConcatFeatures(ep, [
            w.Feature_active_node_id(ep)
        ])

        self.state_space = w.ConcatFeatures(ep, self.global_features.feature_selection +
                                            self.source_node_features.feature_selection +
                                            self.target_node_features.feature_selection)

        self.action_space = w.AbstractAction(ep)

    def valid_actions(self, wrapped_env: w.AgentWrapper, observation):
        """returns a list of valid actions and the nodes they can be carried out from"""

        nodes_and_actions = []
        discovered_nodes = np.union1d(w.owned_nodes(observation), w.discovered_nodes_notowned(observation))

        for from_node in w.owned_nodes(observation):
            for local_action in range(self.action_space.n_local_actions):
                trial_action = self.action_space.abstract_to_gymaction(from_node, observation, local_action, None)
                if trial_action and wrapped_env.env.is_action_valid(trial_action, observation['action_mask']):
                    nodes_and_actions.append((from_node, local_action, -1))

            for remote_action in range(self.action_space.n_local_actions, self.action_space.n_local_actions + self.action_space.n_remote_actions):
                for target_node in discovered_nodes:
                    if target_node != from_node:
                        trial_action = self.action_space.abstract_to_gymaction(from_node, observation, remote_action, target_node)
                        if trial_action and wrapped_env.env.is_action_valid(trial_action, observation['action_mask']):
                            nodes_and_actions.append((from_node, remote_action, target_node))

            for connect_action in range(self.action_space.n_local_actions + self.action_space.n_remote_actions, self.action_space.n_actions):
                trial_action = self.action_space.abstract_to_gymaction(from_node, observation, connect_action, None)
                if trial_action and wrapped_env.env.is_action_valid(trial_action, observation['action_mask']):
                    nodes_and_actions.append((from_node, connect_action, -1))

        return nodes_and_actions

class Memory:

    def __init__(self, ep:EnvironmentBounds, hash_size):
        self.hash_size = hash_size

        self.memory = torch.zeros([2, hash_size], dtype=torch.float64)

    def state_action_index(self, state_space, abstract_action):
        """Turns a state action pair into an index for the memory tensor"""
        feature_vector = np.append(state_space, abstract_action)
        hash_number = abs(hash(str(feature_vector)))

        return hash_number % self.hash_size
        
class ChosenActionMetadata(NamedTuple):
    
    abstract_action: np.int32
    actor_node: int
    actor_features: ndarray
    actor_state: ndarray

    def __repr__(self) -> str:
        return f"[abstract_action={self.abstract_action}, actor={self.actor_node}, state={self.actor_state}]"

class WatkinsQPolicy(Learner):

    def __init__(self,
                 ep: EnvironmentBounds,
                 gamma: float,
                 λ: float,
                 learning_rate: float,
                 hash_size: int
                 ):
        
        self.model = StateActionModel(ep)
        self.n_local_actions = ep.local_attacks_count
        self.n_remote_actions = ep.remote_attacks_count
        self.n_actions = self.n_local_actions + self.n_remote_actions + ep.port_count
        self.gamma = gamma
        self.λ = λ
        self.learning_rate = learning_rate
        self.hash_size = hash_size

        self.memory = Memory(ep, hash_size)

    def parameters_as_string(self):
        return f'γ={self.gamma}, lr={self.learning_rate}, λ={self.λ},\n'

    def all_parameters_as_string(self) -> str:
        model = self.model
        return f'{self.parameters_as_string()}\n' \
            f'dimension={model.state_space.flat_size()}x{model.action_space.flat_size()}, ' \
            f'Q={[f.name() for f in model.state_space.feature_selection]} ' \
            f"-> 'abstract_action'"

    def get_actor_state_vector(self, global_state: ndarray, actor_features: ndarray, target_features: Optional[ndarray]) -> ndarray:
        """Turns seperate state features into one vector"""
        if target_features is None:
            return np.concatenate((np.array(global_state, dtype=np.float32),
                                np.array(actor_features, dtype=np.float32)))
        else:
            return np.concatenate((np.array(global_state, dtype=np.float32),
                                np.array(actor_features, dtype=np.float32),
                                np.array(target_features, dtype=np.float32)))

    def update_memory(self, 
                    reward: float,
                    actor_state: ndarray,
                    abstract_action: int,
                    next_actor_state: Optional[ndarray],
                    next_abstract_action: Optional[int],
                    chosen_action_is_max = boolean):
        
        current_state_action_index = self.memory.state_action_index(actor_state, abstract_action)
        if next_actor_state is None:
            δ = reward - self.memory.memory[0][current_state_action_index].item()
        else:
            next_state_action_index = self.memory.state_action_index(next_actor_state, next_abstract_action)
            δ = reward + (self.gamma * self.memory.memory[0][next_state_action_index].item()) - self.memory.memory[0][current_state_action_index].item()

        self.memory.memory[1][current_state_action_index] +=  1

        non_zero_indicies_q = torch.argwhere(self.memory.memory[0]).numpy()
        non_zero_indicies_e = torch.argwhere(self.memory.memory[1]).numpy()
        non_zero_indicies = np.union1d(non_zero_indicies_q, non_zero_indicies_e)

        for i in non_zero_indicies:

            self.memory.memory[0][i] = self.memory.memory[0][i].item() + float(self.learning_rate * δ * self.memory.memory[1][i].item())
            self.memory.memory[0][i] = round(self.memory.memory[0][i].item(), 5)
            self.memory.memory[0][i] = max(0, self.memory.memory[0][i].item())
            self.memory.memory[0][i] = min(100, self.memory.memory[0][i].item())

            if chosen_action_is_max:
                self.memory.memory[1][i] = self.memory.memory[1][i].item() * float(self.gamma * self.λ)
                self.memory.memory[1][i] = round(self.memory.memory[0][i].item(), 5)
            else:
                self.memory.memory[1][i] = 0

    def on_step(self, wrapped_env: w.AgentWrapper,
                observation, reward: float, done: bool, action_metadata, epsilon):

        if done:
            self.update_memory(reward,
                            actor_state=action_metadata.actor_state,
                            abstract_action=action_metadata.abstract_action,
                            next_actor_state=None,
                            next_abstract_action=None,
                            chosen_action_is_max=False           
                            )
        else:
            x = np.random.rand()
            if x <= epsilon:
                _, _, chosen_next_action_metadata = self.explore(wrapped_env)
            else:
                _, _, chosen_next_action_metadata = self.exploit(wrapped_env, observation)

            chosen_action_pair = ((list(chosen_next_action_metadata.actor_state), chosen_next_action_metadata.abstract_action))
            max_action_pairs = self.max_action_in_state(wrapped_env, observation)

            if chosen_action_pair in max_action_pairs:
                next_action_pair = chosen_action_pair
                chosen_action_is_max = True
            else:
                next_action_pair = random.choice(max_action_pairs)
                chosen_action_is_max = False

            self.update_memory(reward,
                            actor_state=action_metadata.actor_state,
                            abstract_action=action_metadata.abstract_action,
                            next_actor_state=next_action_pair[0],
                            next_abstract_action=next_action_pair[1],
                            chosen_action_is_max=chosen_action_is_max
                            )
        

    def new_episode(self):
        torch.mul(self.memory.memory[1], 0)

    def end_of_episode(self, i_episode, t):
        return None

    def end_of_iteration(self, t, done):
        return None

    def metadata_from_gymaction(self, wrapped_env, gym_action):
        """Takes in a gym action and returns it's metadata"""
        current_global_state = self.model.global_features.get(wrapped_env.state, node=None)
        actor_node = cyberbattle_env.sourcenode_of_action(gym_action)
        actor_features = self.model.source_node_features.get(wrapped_env.state, actor_node)
        abstract_action = self.model.action_space.abstract_from_gymaction(gym_action)

        if 'remote_vulnerability' in gym_action:
            target_node = self.model.target_node_features.get(wrapped_env.state, gym_action['remote_vulnerability'][1])
        else:
            target_node = None

        return ChosenActionMetadata(
            abstract_action=abstract_action,
            actor_node=actor_node,
            actor_features=actor_features,
            actor_state=self.get_actor_state_vector(current_global_state, actor_features, target_node))

    def stateaction_as_string(self, action_metadata) -> str:
        return ''

    def explore(self, wrapped_env: w.AgentWrapper
                ) -> Tuple[str, cyberbattle_env.Action, object]:

        gym_action = wrapped_env.env.sample_valid_action(kinds=[0, 1, 2])
        metadata = self.metadata_from_gymaction(wrapped_env, gym_action)
        return gym_action, metadata

    def exploit(self, wrapped_env: w.AgentWrapper, observation) -> Tuple[str, Optional[cyberbattle_env.Action], object]:

        current_global_state = self.model.global_features.get(wrapped_env.state, node=None)
        valid_nodes_and_actions = self.model.valid_actions(wrapped_env, observation)

        #The q_values are the estimated returns from an action taken in the current state
        q_values = []
        for item in valid_nodes_and_actions:
            source_node_features = self.model.source_node_features.get(wrapped_env.state, item[0])

            if item[1] < self.n_local_actions or item[1] - self.n_local_actions > self.n_remote_actions:
                actor_state_vector = self.get_actor_state_vector(current_global_state, source_node_features, None)
            else:
                target_node_features = self.model.target_node_features.get(wrapped_env.state, item[2])
                actor_state_vector = self.get_actor_state_vector(current_global_state, source_node_features, target_node_features)

            action_state_index = self.memory.state_action_index(actor_state_vector, item[1])

            q_values.append(self.memory.memory[0][action_state_index].item())

        indicies_of_chosen_actions = [i for i, x in enumerate(q_values) if x == max(q_values)]
        chosen_action_index = random.choice(indicies_of_chosen_actions)
        chosen_action = valid_nodes_and_actions[chosen_action_index]

        if chosen_action[1] < self.n_local_actions or chosen_action[1] - self.n_local_actions > self.n_remote_actions:
            gym_action = self.model.action_space.abstract_to_gymaction(chosen_action[0], observation, chosen_action[1], None)
        else:
            gym_action = self.model.action_space.abstract_to_gymaction(chosen_action[0], observation, chosen_action[1], chosen_action[2])

        metadata = self.metadata_from_gymaction(wrapped_env, gym_action)

        return gym_action, metadata

    def max_action_in_state(self, wrapped_env: w.AgentWrapper, observation):
        current_global_state = self.model.global_features.get(wrapped_env.state, node=None)
        valid_nodes_and_actions = self.model.valid_actions(wrapped_env, observation)

        q_values = []
        states_and_actions = []
        for item in valid_nodes_and_actions:
            source_node_features = self.model.source_node_features.get(wrapped_env.state, item[0])

            if item[1] < self.n_local_actions:
                actor_state_vector = self.get_actor_state_vector(current_global_state, source_node_features, None)
            elif item[1] - self.n_local_actions < self.n_remote_actions:
                target_node_features = self.model.target_node_features.get(wrapped_env.state, item[2])
                actor_state_vector = self.get_actor_state_vector(current_global_state, source_node_features, target_node_features)
            else:
                actor_state_vector = self.get_actor_state_vector(current_global_state, source_node_features, None)

            action_state_index = self.memory.state_action_index(actor_state_vector, item[1])

            q_values.append(self.memory.memory[0][action_state_index].item())
            states_and_actions.append((list(actor_state_vector), item[1]))

        indicies_of_chosen_actions = [i for i, x in enumerate(q_values) if x == max(q_values)]
        to_return = []
        for i in range(len(states_and_actions)):
            if i in indicies_of_chosen_actions:
                to_return.append(states_and_actions[i])

        return to_return