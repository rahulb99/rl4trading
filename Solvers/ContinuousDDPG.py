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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

from torch.optim import Adam
from torch.distributions.normal import Normal

from Solvers.Abstract_Solver import AbstractSolver, Statistics
from lib import plotting

# # 

# # class QNetwork(nn.Module):
# #     def __init__(self, obs_dim, act_dim, hidden_sizes):
# #         super().__init__()
# #         sizes = [obs_dim + act_dim] + hidden_sizes + [1]
# #         self.layers = nn.ModuleList()
# #         self.norms = nn.ModuleList()

# #         for i in range(len(sizes) - 1):
# #             layer = nn.Linear(sizes[i], sizes[i + 1])
# #             # Kaiming Init
# #             nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
# #             nn.init.constant_(layer.bias, 0.0)
# #             self.layers.append(layer)
            
# #             # Add LayerNorm for every layer except the last output
# #             if i < len(sizes) - 2:
# #                 self.norms.append(nn.LayerNorm(sizes[i+1]))

# #         # Final layer init
# #         nn.init.uniform_(self.layers[-1].weight, -0.003, 0.003)
# #         nn.init.uniform_(self.layers[-1].bias, -0.003, 0.003)

# #     def forward(self, obs, act):
# #         # Ensure action has at least 2 dims (Batch, Action_Dim)
# #         if len(act.shape) == 1:
# #             act = act.unsqueeze(-1)
        
# #         # Concatenate State and Action
# #         x = torch.cat([obs, act], dim=-1)
        
# #         for i in range(len(self.layers) - 1):
# #             x = self.layers[i](x)
# #             x = self.norms[i](x) 
# #             x = F.leaky_relu(x, negative_slope=0.01)
            
# #         return self.layers[-1](x)


# # class PolicyNetwork(nn.Module):
# #     def __init__(self, obs_dim, act_dim, act_lim, hidden_sizes):
# #         super().__init__()
# #         sizes = [obs_dim] + hidden_sizes + [act_dim]
# #         self.act_lim = act_lim
# #         self.layers = nn.ModuleList()
# #         self.norms = nn.ModuleList()

# #         for i in range(len(sizes) - 1):
# #             layer = nn.Linear(sizes[i], sizes[i + 1])
# #             nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
# #             nn.init.constant_(layer.bias, 0.0)
# #             self.layers.append(layer)
            
# #             if i < len(sizes) - 2:
# #                 self.norms.append(nn.LayerNorm(sizes[i+1]))
            
# #         nn.init.uniform_(self.layers[-1].weight, -0.003, 0.003)
# #         nn.init.uniform_(self.layers[-1].bias, -0.003, 0.003)

# #     def forward(self, obs):
# #         x = obs
# #         for i in range(len(self.layers) - 1):
# #             x = self.layers[i](x)
# #             x = self.norms[i](x)
# #             x = F.leaky_relu(x, negative_slope=0.01)
            
# #         # Returns vector of shape (Batch, Act_Dim)
# #         return self.act_lim * torch.tanh(self.layers[-1](x))

# class QNetwork(nn.Module):
#     def __init__(self, obs_dim, act_dim, hidden_sizes):
#         super().__init__()
#         sizes = [obs_dim + act_dim] + hidden_sizes + [1]
#         self.layers = nn.ModuleList()
#         for i in range(len(sizes) - 1):
#             self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

#     def forward(self, obs, act):
#         x = torch.cat([obs, act], dim=-1)
#         for i in range(len(self.layers) - 1):
#             x = F.relu(self.layers[i](x))
#         return self.layers[-1](x).squeeze(dim=-1)


# class PolicyNetwork(nn.Module):
#     def __init__(self, obs_dim, act_dim, act_lim, hidden_sizes):
#         super().__init__()
#         sizes = [obs_dim] + hidden_sizes + [act_dim]
#         self.act_lim = act_lim
#         self.layers = nn.ModuleList()
#         for i in range(len(sizes) - 1):
#             self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

#     def forward(self, obs):
#         x = torch.cat([obs], dim=-1)
#         for i in range(len(self.layers) - 1):
#             x = F.relu(self.layers[i](x))
#         return self.act_lim * F.tanh(self.layers[-1](x))

# class ActorCriticNetwork(nn.Module):
#     def __init__(self, obs_dim, act_dim, act_lim, hidden_sizes):
#         super().__init__()
#         self.q = QNetwork(obs_dim, act_dim, hidden_sizes)
#         self.pi = PolicyNetwork(obs_dim, act_dim, act_lim, hidden_sizes)


# class ContinuousDDPG(AbstractSolver):
#     def __init__(self, env, eval_env, options):
#         super().__init__(env, eval_env, options)
        
#         obs_dim = env.observation_space.shape[0]
        
#         # --- [MODIFICATION] Robust Action Dimension Detection ---
#         if hasattr(env.action_space, 'shape') and env.action_space.shape is not None:
#             self.act_dim = env.action_space.shape[0]
#         else:
#             self.act_dim = 1
            
#         act_lim = float(options.max_k) 

#         self.actor_critic = ActorCriticNetwork(
#             obs_dim, self.act_dim, act_lim, self.options.layers
#         )
#         self.target_actor_critic = deepcopy(self.actor_critic)

#         self.optimizer_q = Adam(self.actor_critic.q.parameters(), lr=self.options.alpha, weight_decay=1e-5)
#         self.optimizer_pi = Adam(self.actor_critic.pi.parameters(), lr=self.options.alpha, weight_decay=1e-5)

#         for param in self.target_actor_critic.parameters():
#             param.requires_grad = False

#         self.replay_memory = deque(maxlen=options.replay_memory_size)
#         self.noise_scale = options.noise_scale
        
#         self.warmup_steps = 1000
#         self.total_steps = 0
        
#     @torch.no_grad()
#     def update_target_networks(self, tau=0.995):
#         for param, param_targ in zip(self.actor_critic.parameters(), self.target_actor_critic.parameters()):
#             param_targ.data.mul_(tau)
#             param_targ.data.add_((1 - tau) * param.data)

#     def create_greedy_policy(self):
#         @torch.no_grad()
#         def policy_fn(state):
#             state = torch.as_tensor(state, dtype=torch.float32)
#             return self.actor_critic.pi(state).numpy()
#         return policy_fn

#     @torch.no_grad()
#     def select_action(self, state, training=True):
#         act_lim = self.actor_critic.pi.act_lim
        
#         # --- [MODIFICATION] Vectorized Random Sampling ---
#         if training and self.total_steps < self.warmup_steps:
#             # if self.act_dim == 1:
#             #     return np.array([np.random.uniform(-act_lim, act_lim)], dtype=np.float32)
#             # else:
#             #     return np.random.uniform(-act_lim, act_lim, size=(self.act_dim,)).astype(np.float32)
#             return np.random.uniform(-act_lim, act_lim, size=(self.act_dim,)).astype(np.float32)

    
#         state = torch.as_tensor(state, dtype=torch.float32)
#         mu = self.actor_critic.pi(state)
        
#         if training:
#             noise = torch.randn_like(mu) * self.noise_scale
#             action = mu + noise
#         else:
#             action = mu

#         action = torch.clamp(action, -act_lim, act_lim)
        
#         if training:
#             # --- [MODIFICATION] Vectorized Dead Zone Logic ---
#             # Applies to every stock in the portfolio vector simultaneously
#             # If 0.1 < |val| < 1.0, round to -1 or 1 to avoid fractional micro-trades
#             # if self.act_dim == 1:
#             #     val = action.item()
#             #     # If inside the Dead Zone (-1 to 1) but not exactly 0
#             #     if abs(val) > 0.1 and abs(val) < 1.0:
#             #         # Round up/down to nearest integer magnitude
#             #         action[0] = 1.0 if val > 0 else -1.0
#             # else:
#             #     mask = (torch.abs(action) > 0.1) & (torch.abs(action) < 1.0)
#             #     action[mask] = torch.sign(action[mask]) * 1.0
#             mask = (torch.abs(action) > 0.1) & (torch.abs(action) < 1.0)
#             action[mask] = torch.sign(action[mask]) * 1.0
#         return action.numpy()

#     @torch.no_grad()
#     def compute_target_values(self, next_states, rewards, dones):
#         next_actions = self.target_actor_critic.pi(next_states)
#         target_q_values = self.target_actor_critic.q(next_states, next_actions)
#         target_q_values = target_q_values.squeeze(-1)
#         return rewards + self.options.gamma * (1 - dones) * target_q_values

#     def replay(self):
#         if len(self.replay_memory) > self.options.batch_size:
#             minibatch = random.sample(self.replay_memory, self.options.batch_size)
            
#             states = torch.as_tensor(np.stack([x[0] for x in minibatch]), dtype=torch.float32)
#             actions = torch.as_tensor(np.stack([x[1] for x in minibatch]), dtype=torch.float32)
#             rewards = torch.as_tensor(np.stack([x[2] for x in minibatch]), dtype=torch.float32)
#             next_states = torch.as_tensor(np.stack([x[3] for x in minibatch]), dtype=torch.float32)
#             dones = torch.as_tensor(np.stack([x[4] for x in minibatch]), dtype=torch.float32)

#             # --- [MODIFICATION] Shape Safety ---
#             # Ensure actions are (Batch, Act_Dim)
#             if len(actions.shape) == 1:
#                 actions = actions.unsqueeze(-1)

#             # 1. Update Critic
#             current_q = self.actor_critic.q(states, actions).squeeze(-1)
#             target_q = self.compute_target_values(next_states, rewards, dones)
            
#             loss_q = F.mse_loss(current_q, target_q)
            
#             self.optimizer_q.zero_grad()
#             loss_q.backward()
#             torch.nn.utils.clip_grad_norm_(self.actor_critic.q.parameters(), 1.0)
#             self.optimizer_q.step()

#             # 2. Update Actor
#             for p in self.actor_critic.q.parameters():
#                 p.requires_grad = False

#             # policy(states) returns (Batch, Act_Dim), Q expects (Batch, Obs_Dim) + (Batch, Act_Dim)
#             actor_loss = -self.actor_critic.q(states, self.actor_critic.pi(states)).mean()
            
#             self.optimizer_pi.zero_grad()
#             actor_loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.actor_critic.pi.parameters(), 1.0)
#             self.optimizer_pi.step()

#             for p in self.actor_critic.q.parameters():
#                 p.requires_grad = True

#             self.update_target_networks()

#     def memorize(self, state, action, reward, next_state, done):
#         self.replay_memory.append((state, action, reward, next_state, done))

#     def train_episode(self):
#         state, _ = self.env.reset()
#         for _ in range(self.options.steps):
            
#             action = self.select_action(state)
            
#             next_state, reward, done, _, _  = self.env.step(action)
            
#             self.memorize(state, action, reward, next_state, done)
#             self.replay()
            
#             self.statistics[Statistics.Rewards.value] += reward
#             self.statistics[Statistics.Steps.value] += 1
            
#             self.total_steps += 1
            
#             state = next_state
#             if done:
#                 break

#     def __str__(self):
#         return "Continuous DDPG"
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam
from torch.distributions.normal import Normal

from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim + act_dim] + hidden_sizes + [1]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x).squeeze(dim=-1)


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, act_lim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.act_lim = act_lim
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs):
        x = torch.cat([obs], dim=-1)
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.act_lim * F.tanh(self.layers[-1](x))


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, act_lim, hidden_sizes):
        super().__init__()
        self.q = QNetwork(obs_dim, act_dim, hidden_sizes)
        self.pi = PolicyNetwork(obs_dim, act_dim, act_lim, hidden_sizes)


class ContinuousDDPG(AbstractSolver):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)
        # Create actor-critic network
        self.actor_critic = ActorCriticNetwork(
            env.observation_space.shape[0], # obs_dim
            env.action_space.shape[0], # act_dim
            env.action_space.high[0], # act_lim
            self.options.layers, # hidden_sizes
        ).to(self.device)
        # Create target actor-critic network
        self.target_actor_critic = deepcopy(self.actor_critic).to(self.device)

        self.policy = self.create_greedy_policy()

        self.optimizer_q = Adam(self.actor_critic.q.parameters(), lr=self.options.alpha)
        self.optimizer_pi = Adam(
            self.actor_critic.pi.parameters(), lr=self.options.alpha
        )

        # Freeze target actor critic network parameters
        for param in self.target_actor_critic.parameters():
            param.requires_grad = False

        # Replay buffer
        self.replay_memory = deque(maxlen=options.replay_memory_size)
        
        # Action noise
        self.noise_scale = options.noise_scale
        
        
    @torch.no_grad()
    def update_target_networks(self, tau=0.995):
        """
        Copy params from actor_critic to target_actor_critic using Polyak averaging.
        """
        for param, param_targ in zip(
            self.actor_critic.parameters(), self.target_actor_critic.parameters()
        ):
            param_targ.data.mul_(tau)
            param_targ.data.add_((1 - tau) * param.data)

    def create_greedy_policy(self):
        """
        Creates a greedy policy.

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        @torch.no_grad()
        def policy_fn(state):
            # --- [GPU CHANGE] Move state to device ---
            state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
            # --- [GPU CHANGE] Move output to cpu before numpy conversion ---
            return self.actor_critic.pi(state).cpu().numpy()

        return policy_fn

    @torch.no_grad()

    def select_action(self, state, training=True):
        """
        Selects an action given state.
        """
        # --- [GPU CHANGE] Move state to device ---
        state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        mu = self.actor_critic.pi(state)
        
        # --- [GPU CHANGE] Create Normal distribution on the correct device ---
        m = Normal(
            torch.zeros(self.env.action_space.shape[0], device=self.device),
            torch.ones(self.env.action_space.shape[0], device=self.device),
        )
        action_limit = self.env.action_space.high[0]
        action = mu + training * self.noise_scale * m.sample()
        
        # --- [GPU CHANGE] Clip and move to CPU ---
        return torch.clip(
            action,
            -action_limit,
            action_limit,
        ).cpu().numpy()

    @torch.no_grad()
    def compute_target_values(self, next_states, rewards, dones):
        """
        Computes the target q values.

        Use:
            self.target_actor_critic.pi(states): Returns the greedy action at states.
            self.target_actor_critic.q(states, actions): Returns the Q-values 
                for (states, actions).

        Returns:
            The target q value (as a tensor).
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        target_q_values = self.target_actor_critic.q(next_states, self.target_actor_critic.pi(next_states)) # Get Q-values for next states and actions from target policy
        return rewards + self.options.gamma * (1 - dones) * target_q_values # Compute target values


    def replay(self):
        """
        Samples transitions from the replay memory and updates actor_critic network.
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
            states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
            dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

            # Current Q-values
            current_q = self.actor_critic.q(states, actions)
            # Target Q-values
            target_q = self.compute_target_values(next_states, rewards, dones)

            # Optimize critic network
            loss_q = self.q_loss(current_q, target_q).mean()
            self.optimizer_q.zero_grad()
            loss_q.backward()
            self.optimizer_q.step()

            # Optimize actor network
            loss_pi = self.pi_loss(states).mean()
            self.optimizer_pi.zero_grad()
            loss_pi.backward()
            self.optimizer_pi.step()

    def memorize(self, state, action, reward, next_state, done):
        """
        Adds transitions to the replay buffer.
        """
        self.replay_memory.append((state, action, reward, next_state, done))

    def train_episode(self):
        """
        Runs a single episode of the DDPG algorithm.

        Use:
            self.select_action(state): Sample an action from the policy.
            self.step(action): Performs an action in the env.
            self.memorize(state, action, reward, next_state, done): store the transition in
                the replay buffer.
            self.replay(): Sample transitions and update actor_critic.
            self.update_target_networks(): Update target_actor_critic using Polyak averaging.
        """

        state, _ = self.env.reset()
        for _ in range(self.options.steps):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            action = self.select_action(state)
            next_state, reward, done, _ = self.step(action)
            self.memorize(state, action, reward, next_state, done)
            self.replay()
            self.update_target_networks()
            state = next_state
            if done:
                break
            

    def q_loss(self, current_q, target_q):
        """
        The q loss function.

        args:
            current_q: Current Q-values.
            target_q: Target Q-values.

        Returns:
            The unreduced loss (as a tensor).
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        return F.mse_loss(current_q, target_q, reduction='none')

    def pi_loss(self, states):
        """
        The policy gradient loss function.
        Note that you are required to define the Loss^PG
        which should be the integral of the policy gradient
        The "returns" is the one-hot encoded (return - baseline) value for each action a_t
        ('0' for unchosen actions).

        args:
            states:

        Use:
            self.actor_critic.pi(states): Returns the greedy action at states.
            self.actor_critic.q(states, actions): Returns the Q-values for (states, actions).

        Returns:
            The unreduced loss (as a tensor).
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        return -1 * self.actor_critic.q(states, self.actor_critic.pi(states))

    def __str__(self):
        return "DDPG"

    def plot(self, stats, smoothing_window=20, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)
