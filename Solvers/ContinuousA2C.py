# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code was developed by Guni Sharon (guni@tamu.edu).
# The PyTorch code was developed by Sheelabhadra Dey (sheelabhadra@tamu.edu).

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributions as distributions

from torch.optim import Adam

from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, action_limit):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes
        self.layers = nn.ModuleList()
        
        # Shared layers
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            
        # Actor Mean (Mu) head
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        
        # Actor Standard Deviation (Sigma) head
        self.sigma_layer = nn.Linear(hidden_sizes[-1], act_dim)
        
        # Critic head
        self.value_layer = nn.Linear(hidden_sizes[-1], 1)

        self.action_limit = action_limit

    def forward(self, obs):
        x = torch.cat([obs], dim=-1)
        for layer in self.layers:
            x = F.relu(layer(x))
            
        # Actor head: Mu
        # Tanh maps to [-1, 1], then we scale by action_limit
        mu = torch.tanh(self.mu_layer(x)) * self.action_limit
        
        # Actor head: Sigma
        # Softplus ensures standard deviation is always positive
        sigma = F.softplus(self.sigma_layer(x)) + 1e-5 

        # Critic head
        value = self.value_layer(x)

        return mu, sigma, torch.squeeze(value, -1)

class MultiContinuousA2C(AbstractSolver):
    """
    To support continuous actions, the A2C algorithm must be adapted to use a Gaussian Policy 
    (predicting Mean and Standard Deviation) instead of a Categorical Policy (predicting discrete probabilities).
    
    ActorCriticNetwork: Now outputs mu (scaled by the action limit) and sigma (using softplus to ensure positivity).
    select_action: Uses torch.distributions.Normal to sample continuous actions and calculates the log_prob.
    actor_loss: Uses the log_prob directly 
    """
    
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)

        # Create actor-critic network
        self.actor_critic = ActorCriticNetwork(
            env.observation_space.shape[0], 
            env.action_space.shape[0], 
            self.options.layers,
            env.action_space.high[0]
        )
        self.policy = self.create_greedy_policy()

        self.optimizer = Adam(self.actor_critic.parameters(), lr=self.options.alpha)
        self.entropy_pct = options.entropy_pct

    def create_greedy_policy(self):
        """
        Creates a greedy policy (Deterministic).
        For continuous A2C, the greedy policy returns the Mean (mu).
        """
        def policy_fn(state):
            state = torch.as_tensor(state, dtype=torch.float32)
            mu, _, _ = self.actor_critic(state)
            return mu.detach().numpy()

        return policy_fn

    def select_action(self, state, training=True):
        """
        Selects an action given state.

        Returns:
            The selected action (numpy array)
            The log_probability of the selected action (tensor)
            The critic's value estimate (tensor)
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        mu, sigma, value = self.actor_critic(state)

        # Create a Normal distribution
        dist = distributions.Normal(mu, sigma)

        if training:
            # Sample from the distribution
            action = dist.sample()
            
            # Calculate log_prob
            # We sum over the last dimension because the actions are independent
            # p(a1, a2) = p(a1) * p(a2) -> log_p = log_p(a1) + log_p(a2)
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            entropy = dist.entropy().sum(dim=-1)
            
            return action.detach().numpy(), log_prob, value, entropy
        else:
            # During evaluation, just return the mean
            return mu.detach().numpy()

    def update_actor_critic(self, advantage, log_prob, value, entropy):
        """
        Performs actor critic update.

        args:
            advantage: Advantage of the chosen action (tensor).
            log_prob: Log-Probability associated with the chosen action (tensor).
            value: Critic's state value estimate (tensor).
        """
        # Compute loss
        # Note: We pass log_prob directly now
        actor_loss = self.actor_loss(advantage.detach(), log_prob).mean()
        critic_loss = self.critic_loss(advantage.detach(), value).mean()

        loss = actor_loss + critic_loss

        entropy_loss = -self.entropy_pct * entropy.mean()

        loss = actor_loss + critic_loss + entropy_loss
        
        # Update actor critic
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping: Essential for financial data to stop exploding gradients
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1.0)
        
        self.optimizer.step()

    def train_episode(self):
        state, _ = self.env.reset()
        for _ in range(self.options.steps):
            
            # Select action using Gaussian Policy
            action, log_prob, value, entropy = self.select_action(state)
            
            # Step in env
            next_state, reward, done, _ = self.step(action)
            
            next_value = 0
            if not done:
                # We only need the value estimate for the next state
                _, _, next_v_tensor = self.actor_critic(
                    torch.as_tensor(next_state, dtype=torch.float32)
                )
                next_value = next_v_tensor

            # Calculate Target and Advantage
            td_target = reward + self.options.gamma * next_value
            advantage = td_target - value
            
            # Update
            self.update_actor_critic(advantage, log_prob, value, entropy)
            
            state = next_state
            if done:
                break

    def actor_loss(self, advantage, log_prob):
        """
        The policy gradient loss function for Continuous Action.
        Loss = - log_prob * advantage

        args:
            advantage: Advantage of the chosen action.
            log_prob: Log probability of the chosen action.

        Returns:
            The unreduced loss (as a tensor).
        """
        return -log_prob * advantage

    def critic_loss(self, advantage, value):
        """
        The integral of the critic gradient.
        Usually MSE or similar, but here defined via advantage * value for consistency 
        with specific gradient derivations or simply MSE (advantage^2).
        
        Note: The original discrete code used -advantage * value. 
        Assuming this follows the specific derivation provided in your course material.
        Standard A2C critic loss is often F.mse_loss(td_target, value).
        Keeping original implementation logic:
        """
        return -advantage * value

    def __str__(self):
        return "Continuous A2C"

    def plot(self, stats, smoothing_window=20, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)
