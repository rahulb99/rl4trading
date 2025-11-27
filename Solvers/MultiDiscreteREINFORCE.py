# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).
# The PyTorch code was developed by Sheelabhadra Dey (sheelabhadra@tamu.edu).

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np

from torch.optim import Adam
from Solvers.Abstract_Solver import AbstractSolver, Statistics
from lib import plotting
from statistics import mean



class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_assets, n_actions_per_asset, hidden_sizes):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes
        self.layers = nn.ModuleList()
        # Shared layers
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        
        self.n_assets = n_assets
        self.n_actions_per_asset = n_actions_per_asset

        # Policy head: Outputs logits for all assets
        # Size: n_assets * n_actions_per_asset
        self.policy_head = nn.Linear(hidden_sizes[-1], n_assets * n_actions_per_asset)
        
        # Baseline head layers
        self.baseline_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, obs):
        x = torch.cat([obs], dim=-1)
        for layer in self.layers:
            x = F.relu(layer(x))
            
        # Policy: Reshape to (Batch, n_assets, n_actions) and apply Softmax per asset
        logits = self.policy_head(x)
        logits = logits.view(-1, self.n_assets, self.n_actions_per_asset)
        probs = F.softmax(logits, dim=-1)
        
        # Baseline
        baseline = self.baseline_head(x)

        # Remove batch dim if input was single observation
        if len(obs.shape) == 1:
            probs = probs.squeeze(0)
            baseline = baseline.squeeze(0)

        return probs, torch.squeeze(baseline, -1)


class MultiDiscreteREINFORCE(AbstractSolver):
    """
    Key changes 
    PolicyNet: Now outputs a tensor of shape (Batch, n_assets, n_actions) and applies Softmax along the last dimension to ensure each asset has a valid probability distribution.
    
    Select Action: Uses torch.distributions.Categorical to sample actions for all assets simultaneously. 
    It sums the log_prob of each asset to get the total log-probability of the joint action.
    
    PG Loss: Adjusted to take the pre-calculated log_prob directly (for numerical stability) instead of taking the log of a tiny probability product.
    """
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)
        
        # Detect MultiDiscrete dimensions
        if hasattr(env.action_space, 'nvec'):
            self.nvec = env.action_space.nvec
            n_assets = len(self.nvec)
            n_actions = int(self.nvec[0])
        else:
            # Fallback for standard Discrete (treat as 1 asset)
            n_assets = 1
            n_actions = env.action_space.n

        # Create the policy network
        self.model = PolicyNet(
            obs_dim=env.observation_space.shape[0], 
            n_assets=n_assets,
            n_actions_per_asset=n_actions,
            hidden_sizes=self.options.layers
        )
        self.optimizer = Adam(self.model.parameters(), lr=self.options.alpha)
        self.entropy_pct = options.entropy_pct

    def create_greedy_policy(self):
        """
        Creates a greedy policy.
        Returns the action with highest probability for each asset.
        """
        def policy_fn(state):
            state = torch.as_tensor(state, dtype=torch.float32)
            probs, _ = self.model(state)
            # Argmax along the last dimension (actions)
            action = torch.argmax(probs, dim=-1)
            return action.detach().numpy()

        return policy_fn

    def compute_returns(self, rewards, gamma):
        """
        Compute the returns along an episode.
        """
        rewards = torch.tensor(rewards, dtype=torch.float32)
        returns = torch.zeros_like(rewards)
        G = 0
        for r in reversed(range(len(rewards))):
            G = rewards[r] + gamma * G
            returns[r] = G
        return returns.tolist()

    def select_action(self, state, training = True):
        """
        Selects an action given state using MultiDiscrete logic.

        Returns:
            The selected action (numpy array of ints)
            The log_probability sum of the joint action (tensor)
            The baseline value (tensor)
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        probs, baseline = self.model(state)

        # Create Independent Categorical Distributions for each asset
        # probs shape: (n_assets, n_actions)
        dist = distributions.Categorical(probs)

        # Sample actions
        # action shape: (n_assets,)
        action = dist.sample()

        # Calculate Log Probability
        # log_prob shape: (n_assets,)
        # We sum them because joint probability is product of independent probs
        # log(P1 * P2 * ...) = log(P1) + log(P2) + ...
        log_prob_sum = dist.log_prob(action).sum()
        
        # Entropy: Sum across assets (CRITICAL for Small Data exploration)
        entropy_sum = dist.entropy().sum()

        if training:
            return action.detach().numpy(), log_prob_sum, baseline, entropy_sum
        else:
            return action.detach().numpy()

    def update_model(self, rewards, log_probs, baselines, entropies):
        """
        Performs model update.
        """
        returns = torch.as_tensor(
            self.compute_returns(rewards, self.options.gamma), dtype=torch.float32
        )
        
        # Stack list of tensors into a single tensor
        log_probs = torch.stack(log_probs)
        baselines = torch.stack(baselines)
        entropies = torch.stack(entropies)

        # Compute advantage (delta)
        # Normalize returns for stability (Optional but recommended for REINFORCE)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        deltas = returns - baselines

        # Compute loss
        pg_loss = self.pg_loss(deltas.detach(), log_probs).mean()
        value_loss = F.smooth_l1_loss(returns.detach(), baselines)

        # Entropy Regularization (Maximize entropy -> Negative Loss)
        # Hardcoded 0.01 coefficient if not provided
        entropy_loss = -self.entropy_pct * entropies.mean()
        
        loss = pg_loss + value_loss + entropy_loss

        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()

    def train_episode(self):
        state, _ = self.env.reset()
        rewards = []  # Reward per step
        log_probs = []  # Log Action probability
        baselines = []  # Value function
        entropies = []
        
        for _ in range(self.options.steps):
            
            action, log_prob, baseline, entropy = self.select_action(state)
            
            next_state, reward, done, _, _ = self.env.step(action)
            
            # Update Statistics for the Runner
            
            self.statistics[Statistics.Rewards.value] += reward
            self.statistics[Statistics.Steps.value] += 1
            
            rewards.append(reward)
            log_probs.append(log_prob)
            baselines.append(baseline)
            entropies.append(entropy)
            
            state = next_state
            if done:
                break
                
        self.update_model(rewards, log_probs, baselines, entropies)

    def pg_loss(self, advantage, log_prob):
        """
        The policy gradient loss function.
        
        Adapted for Numerical Stability:
        Since select_action now returns the `log_prob` directly (to avoid 
        underflow from multiplying many small probabilities), we remove 
        the `torch.log()` call here.
        
        Loss = -log_prob * advantage
        """
        return -log_prob * advantage

    def __str__(self):
        return "MultiDiscrete REINFORCE"

    def plot(self, stats, smoothing_window, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)