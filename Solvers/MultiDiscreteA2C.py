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

from Solvers.Abstract_Solver import AbstractSolver, Statistics
from lib import plotting


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim, n_assets, n_actions_per_asset, hidden_sizes):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes
        self.layers = nn.ModuleList()
        
        # Shared layers
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            
        self.n_assets = n_assets
        self.n_actions_per_asset = n_actions_per_asset
        
        # Actor head: Outputs logits for every action of every asset
        # Output size: n_assets * n_actions_per_asset
        self.actor_layer = nn.Linear(hidden_sizes[-1], n_assets * n_actions_per_asset)
        
        # Critic head: Value function (scalar)
        self.value_layer = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, obs):
        x = torch.cat([obs], dim=-1)
        for layer in self.layers:
            x = F.relu(layer(x))
            
        # Actor head
        # Reshape to (Batch, n_assets, n_actions_per_asset) so we can create
        # a batch of Categorical distributions
        logits_flat = self.actor_layer(x)
        logits = logits_flat.view(-1, self.n_assets, self.n_actions_per_asset)
        
        # Critic head
        value = self.value_layer(x)

        # If input was 1D (no batch dim), squeeze the batch dim back out for cleaner handling
        if len(obs.shape) == 1:
            logits = logits.squeeze(0)
            
        return logits, torch.squeeze(value, -1)


class MultiDiscreteA2C(AbstractSolver):
    """
    Adapted A2C algorithm for MultiDiscrete Action Spaces.
    
    ActorCriticNetwork: Outputs logits for a Categorical distribution for each asset.
    select_action: Samples from independent Categorical distributions.
    loss: Standard Policy Gradient loss (log_prob * advantage) + Value loss.
    """
    
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)

        # Extract dimensions from the MultiDiscrete space
        # env.action_space.nvec gives the number of actions for each dimension
        # In MAG7TradingEnv, this is [2*k+1, 2*k+1, ...]
        self.nvec = env.action_space.nvec
        n_assets = len(self.nvec)
        n_actions_per_asset = int(self.nvec[0]) # Assuming uniform action space per asset

        # Create actor-critic network
        self.actor_critic = ActorCriticNetwork(
            obs_dim=env.observation_space.shape[0], 
            n_assets=n_assets,
            n_actions_per_asset=n_actions_per_asset,
            hidden_sizes=self.options.layers
        )
        
        self.policy = self.create_greedy_policy()
        self.optimizer = Adam(self.actor_critic.parameters(), lr=self.options.alpha)
        self.entropy_pct = options.entropy_pct

    def create_greedy_policy(self):
        """
        Creates a greedy policy (Deterministic).
        For Discrete/MultiDiscrete, this returns the action with the highest probability (logit).
        """
        def policy_fn(state):
            state = torch.as_tensor(state, dtype=torch.float32)
            logits, _ = self.actor_critic(state)
            # Argmax over the last dimension (actions per asset)
            action = torch.argmax(logits, dim=-1)
            return action.detach().numpy()

        return policy_fn

    def select_action(self, state, training=True):
        """
        Selects an action given state.

        Returns:
            The selected action (numpy array)
            The log_probability sum (tensor)
            The critic's value estimate (tensor)
            The entropy sum (tensor)
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        logits, value = self.actor_critic(state)

        # Create a Categorical distribution
        # logits shape: (n_assets, n_actions_per_asset)
        # Treating this as a batch of 'n_assets' independent distributions
        dist = distributions.Categorical(logits=logits)

        if training:
            # Sample from the distribution
            # action shape: (n_assets,)
            action = dist.sample()
            
            # Calculate log_prob
            # dist.log_prob(action) returns a tensor of shape (n_assets,)
            # We sum over the assets because the joint probability is the product of individual probabilities:
            # log P(a1, a2...) = log P(a1) + log P(a2) ...
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Calculate Entropy
            # Sum entropy across all assets
            entropy = dist.entropy().sum(dim=-1)
            
            return action.detach().numpy(), log_prob, value, entropy
        else:
            # During evaluation (or if greedy), pick the action with max logit
            action = torch.argmax(logits, dim=-1)
            return action.detach().numpy()

    def update_actor_critic(self, advantage, log_prob, value, entropy):
        """
        Performs actor critic update.
        """
        # Actor Loss: -log_prob * advantage
        actor_loss = self.actor_loss(advantage.detach(), log_prob).mean()
        
        # Critic Loss: -advantage * value (following original derivation provided)
        # Or standard MSE: F.mse_loss(target, value)
        critic_loss = self.critic_loss(advantage.detach(), value).mean()

        # Entropy regularization
        entropy_loss = -self.entropy_pct * entropy.mean() # Reduced coefficient slightly for discrete stability

        loss = actor_loss + critic_loss + entropy_loss
        
        # Update actor critic
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1.0)
        
        self.optimizer.step()

    def train_episode(self):
        state, _ = self.env.reset()
        for _ in range(self.options.steps):
            
            # Select action
            action, log_prob, value, entropy = self.select_action(state)
            
            # Step in env
            next_state, reward, done, _, _ = self.env.step(action)
            
            # We must manually add the step reward to the total episode reward
            self.statistics[Statistics.Rewards.value] += reward
            self.statistics[Statistics.Steps.value] += 1
            
            next_value = 0
            if not done:
                # We only need the value estimate for the next state
                _, next_v_tensor = self.actor_critic(
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
        return -log_prob * advantage

    def critic_loss(self, advantage, value):
        # Following the specific derivation form from the provided snippet
        return -advantage * value

    def __str__(self):
        return "MultiDiscrete A2C"

    def plot(self, stats, smoothing_window=20, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)