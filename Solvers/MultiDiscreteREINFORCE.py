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
        # Input 'obs' is expected to be on the correct device already
        x = obs
        if x.dim() == 1:
             x = x.unsqueeze(0)
             
        for layer in self.layers:
            x = F.relu(layer(x))
            
        # Policy: Reshape to (Batch, n_assets, n_actions) and apply Softmax per asset
        logits = self.policy_head(x)
        logits = logits.view(-1, self.n_assets, self.n_actions_per_asset)
        probs = F.softmax(logits, dim=-1)
        
        # Baseline
        baseline = self.baseline_head(x)

        # Remove batch dim if input was single observation (original logic preservation)
        if len(obs.shape) == 1:
            probs = probs.squeeze(0)
            baseline = baseline.squeeze(0)

        return probs, torch.squeeze(baseline, -1)


class MultiDiscreteREINFORCE(AbstractSolver):
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
        ).to(self.device)  # 2. MOVE MODEL TO DEVICE
        
        self.optimizer = Adam(self.model.parameters(), lr=self.options.alpha)
        self.entropy_pct = options.entropy_pct

    def create_greedy_policy(self):
        """
        Creates a greedy policy.
        Returns the action with highest probability for each asset.
        """
        @torch.no_grad()
        def policy_fn(state):
            # 3. MOVE INPUT STATE TO DEVICE
            state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)
            
            # with torch.no_grad(): # No gradients needed for inference
            probs, _ = self.model(state_tensor)
            # Argmax along the last dimension (actions)
            action = torch.argmax(probs, dim=-1)
                
            # 4. MOVE OUTPUT BACK TO CPU FOR NUMPY
            return action.cpu().numpy()

        return policy_fn

    def compute_returns(self, rewards, gamma):
        """
        Compute the returns along an episode.
        """
        # Calculation happens on CPU usually for simple list ops, 
        # but we return a list to be converted to tensor later.
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns


    def select_action(self, state, training=True):
        """
        Selects an action given state using MultiDiscrete logic.
        """
        # 3. MOVE INPUT STATE TO DEVICE
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        
        probs, baseline = self.model(state_tensor)

        # Create Independent Categorical Distributions for each asset
        # probs shape: (n_assets, n_actions)
        dist = distributions.Categorical(probs)

        # Sample actions
        action = dist.sample()

        # Calculate Log Probability
        log_prob_sum = dist.log_prob(action).sum()
        
        # Entropy: Sum across assets
        entropy_sum = dist.entropy().sum()

        if training:
            # Return GPU tensors for log_prob and baseline (needed for backprop)
            # Return CPU numpy for action (needed for Env)
            return action.cpu().detach().numpy(), log_prob_sum, baseline, entropy_sum
        else:
            return action.cpu().detach().numpy()

    def update_model(self, rewards, log_probs, baselines, entropies):
        """
        Performs model update.
        """
        # Compute returns (list of floats)
        returns_list = self.compute_returns(rewards, self.options.gamma)
        
        # 3. MOVE TARGETS TO DEVICE
        returns = torch.tensor(returns_list, dtype=torch.float32).to(self.device)
        
        # Stack list of tensors (These are already on device from select_action)
        log_probs = torch.stack(log_probs)
        baselines = torch.stack(baselines)
        entropies = torch.stack(entropies)

        # Compute advantage (delta)
        # returns and baselines are both on device now
        deltas = returns - baselines

        # Compute loss
        pg_loss = self.pg_loss(deltas.detach(), log_probs).mean()
        value_loss = F.smooth_l1_loss(baselines, returns.detach()) # MSE/L1 between predicted and actual

        # Entropy Regularization
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
        rewards = []  
        log_probs = []  
        baselines = []  
        entropies = []
        
        for _ in range(self.options.steps):
            
            action, log_prob, baseline, entropy = self.select_action(state)
            
            # env.step expects numpy action
            next_state, reward, done, _, _ = self.env.step(action)
            
            self.statistics[Statistics.Rewards.value] += reward
            self.statistics[Statistics.Steps.value] += 1
            
            rewards.append(reward)
            log_probs.append(log_prob)   # Tensor on GPU
            baselines.append(baseline)   # Tensor on GPU
            entropies.append(entropy)    # Tensor on GPU
            
            state = next_state
            if done:
                break
                
        self.update_model(rewards, log_probs, baselines, entropies)

    def pg_loss(self, advantage, log_prob):
        return -log_prob * advantage

    def __str__(self):
        return "MultiDiscrete REINFORCE"

    def plot(self, stats, smoothing_window, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)