# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).
# The PyTorch code was developed by Sheelabhadra Dey (sheelabhadra@tamu.edu).

import random
from copy import deepcopy
from collections import deque
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import AdamW
from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


class QFunction(nn.Module):
    """
    Q-network definition.
    Updated architecture:
    1. Input: Concatenation of Observation + Action
    2. Output: Single neuron (Q-value for that specific s, a pair)
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
    ):
        super().__init__()
        # Input is Observation dim + Action dimensions
        sizes = [obs_dim + act_dim] + hidden_sizes + [1]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs, act):
        # Concatenate observation and action vector
        x = torch.cat([obs, act], dim=-1)
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x).squeeze(dim=-1)


class MultiDQN(AbstractSolver):
    def __init__(self, env, eval_env, options):
        # Check for Discrete or MultiDiscrete action spaces
        self.is_multi_discrete = hasattr(env.action_space, 'nvec')
        
        if self.is_multi_discrete:
            # MultiDiscrete: [n1, n2, n3]
            self.nvec = env.action_space.nvec
            self.num_dims = len(self.nvec)
            
            # Generate ALL possible action combinations (Cartesian product)
            # e.g., if nvec=[2, 2], we get [[0,0], [0,1], [1,0], [1,1]]
            ranges = [np.arange(n) for n in self.nvec]
            self.all_actions = np.array(list(itertools.product(*ranges)))
        else:
            # Standard Discrete
            assert isinstance(env.action_space, (torch.gym.spaces.Discrete, type(env.action_space))), \
                str(self) + " requires Discrete or MultiDiscrete action spaces"
            self.nvec = [env.action_space.n]
            self.num_dims = 1
            # For discrete, actions are just [[0], [1], ..., [n-1]]
            self.all_actions = np.arange(env.action_space.n).reshape(-1, 1)

        super().__init__(env, eval_env, options)
        
        # Store all actions as a tensor for fast batch evaluation
        self.all_actions_tensor = torch.as_tensor(self.all_actions, dtype=torch.float32)
        
        # Create Q-network
        # act_dim is now the NUMBER of dimensions (length of action vector), not total cardinality
        self.model = QFunction(
            env.observation_space.shape[0],
            self.num_dims,
            self.options.layers,
        )
        # Create target Q-network
        self.target_model = deepcopy(self.model)
        
        # Set up the optimizer
        self.optimizer = AdamW(
            self.model.parameters(), lr=self.options.alpha, amsgrad=True
        )
        # Define the loss function
        self.loss_fn = nn.SmoothL1Loss()

        # Freeze target network parameters
        for p in self.target_model.parameters():
            p.requires_grad = False

        # Replay buffer
        self.replay_memory = deque(maxlen=options.replay_memory_size)

        # Number of training steps so far
        self.n_steps = 0

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state, epsilon=0.0):
        """
        Selects an action using epsilon-greedy logic.
        Evaluates Q(s, a) for EVERY possible 'a' and takes the max.
        """
        if np.random.rand() < epsilon:
            # Random selection from the list of all possible actions
            idx = np.random.randint(0, len(self.all_actions))
            action = self.all_actions[idx]
        else:
            state_t = torch.as_tensor(state, dtype=torch.float32)
            
            # 1. Repeat state to match number of possible actions
            # Shape: [Total_Combinations, Obs_Dim]
            state_repeated = state_t.repeat(len(self.all_actions), 1)
            
            # 2. Forward pass with (State, All_Actions)
            with torch.no_grad():
                # self.all_actions_tensor is [Total_Combinations, Act_Dim]
                q_values = self.model(state_repeated, self.all_actions_tensor)
            
            # 3. Find index of max Q-value
            best_idx = torch.argmax(q_values).item()
            action = self.all_actions[best_idx]

        if self.is_multi_discrete:
            return action
        else:
            return action[0]

    def compute_target_values(self, next_states, rewards, dones):
        """
        Computes the target q values.
        Calculates max_a Q(s', a) by evaluating ALL actions for the batch of next states.
        """
        batch_size = next_states.shape[0]
        num_actions = len(self.all_actions)
        
        # 1. Expand Next States: [Batch, 1, Obs] -> [Batch, N_Actions, Obs]
        next_states_expanded = next_states.unsqueeze(1).expand(-1, num_actions, -1)
        
        # 2. Expand All Actions: [1, N_Actions, Act_Dim] -> [Batch, N_Actions, Act_Dim]
        all_actions_expanded = self.all_actions_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 3. Flatten to pass through network: [Batch * N_Actions, ...]
        flat_next_states = next_states_expanded.reshape(-1, next_states.shape[-1])
        flat_all_actions = all_actions_expanded.reshape(-1, self.num_dims)
        
        # 4. Compute Q values for all pairs
        flat_q_values = self.target_model(flat_next_states, flat_all_actions)
        
        # 5. Reshape back to [Batch, N_Actions]
        q_values_matrix = flat_q_values.view(batch_size, num_actions)
        
        # 6. Take Max over actions dimension
        max_q, _ = torch.max(q_values_matrix, dim=1)

        target_q = rewards + self.options.gamma * max_q * (1 - dones)
        return target_q

    def replay(self):
        """
        TD learning for q values on past transitions.
        """
        if len(self.replay_memory) > self.options.batch_size:
            minibatch = random.sample(self.replay_memory, self.options.batch_size)
            minibatch = [
                np.array(
                    [
                        transition[idx]
                        for transition, idx in zip(minibatch, [i] * len(minibatch))
                    ]
                )
                for i in range(5)
            ]
            states, actions, rewards, next_states, dones = minibatch
            
            # Convert numpy arrays to torch tensors
            states = torch.as_tensor(states, dtype=torch.float32)
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            next_states = torch.as_tensor(next_states, dtype=torch.float32)
            dones = torch.as_tensor(dones, dtype=torch.float32)
            
            # Actions tensor
            # If MultiDiscrete, actions is [batch, num_dims]. 
            # If Discrete, it might be [batch] or [batch, 1].
            actions = torch.as_tensor(actions, dtype=torch.float32) # Use float for NN input
            if not self.is_multi_discrete and actions.ndim == 1:
                actions = actions.unsqueeze(1) # Ensure [batch, 1]

            # 1. Calculate Current Q(s, a)
            # Unlike standard DQN, we don't gather. We pass (s, a) directly.
            current_q = self.model(states, actions) 

            # 2. Calculate Target Q(s', a')
            with torch.no_grad():
                target_q = self.compute_target_values(next_states, rewards, dones)

            # Calculate loss
            loss_q = self.loss_fn(current_q, target_q)

            # Optimize the Q-network
            self.optimizer.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
            self.optimizer.step()

    def memorize(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def train_episode(self):
        """
        Perform a single episode of the Q-Learning algorithm.
        """
        # Reset the environment
        state, _ = self.env.reset()

        for _ in range(self.options.steps):
            
            action = self.select_action(state, self.options.epsilon)
            
            next_state, reward, done, _, _ = self.env.step(action) 
            
            self.memorize(state, action, reward, next_state, done)
            
            state = next_state
            
            self.replay()
            
            self.n_steps += 1
            if self.n_steps % self.options.update_target_estimator_every == 0:
                self.update_target_model()
            
            if done:
                break

    def __str__(self):
        return "MultiDQN"

    def plot(self, stats, smoothing_window, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.
        Returns a function that takes an observation as input and returns a greedy action.
        """
        def policy_fn(state):
            return self.select_action(state, epsilon=0.0)

        return policy_fn