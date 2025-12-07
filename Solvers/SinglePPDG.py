import random
from copy import deepcopy
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.distributions.normal import Normal
from Solvers.Abstract_Solver import AbstractSolver, Statistics



# Global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DDPG using device: {device}")

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim + act_dim] + hidden_sizes + [1]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs, act):
        # ✅ FIXED: No broken torch.cat([obs])
        x = torch.cat([obs, act], dim=-1)
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x).squeeze(-1)

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, act_lim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.act_lim = act_lim
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs):
        x = obs
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.act_lim * torch.tanh(self.layers[-1](x))

class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, act_lim, hidden_sizes):
        super().__init__()
        self.q = QNetwork(obs_dim, act_dim, hidden_sizes)
        self.pi = PolicyNetwork(obs_dim, act_dim, act_lim, hidden_sizes)

class DDPG(AbstractSolver):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)

        # ✅ GPU: Move models to device
        self.actor_critic = ActorCriticNetwork(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            float(env.action_space.high[0]),
            self.options.layers,
        ).to(device)

        self.target_actor_critic = deepcopy(self.actor_critic).to(device)

        self.optimizer_q = Adam(self.actor_critic.q.parameters(), lr=self.options.alpha)
        self.optimizer_pi = Adam(self.actor_critic.pi.parameters(), lr=self.options.alpha)

        for param in self.target_actor_critic.parameters():
            param.requires_grad = False

        self.replay_memory = deque(maxlen=options.replay_memory_size)
        self.noise_scale = options.noise_scale

    @torch.no_grad()
    def update_target_networks(self, tau=0.995):
        # ✅ Polyak averaging works on GPU tensors automatically
        for param, param_targ in zip(self.actor_critic.parameters(), self.target_actor_critic.parameters()):
            param_targ.data.mul_(tau)
            param_targ.data.add_((1 - tau) * param.data)

    def create_greedy_policy(self):
        @torch.no_grad()
        def policy_fn(state):
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            return self.actor_critic.pi(state_t).cpu().numpy()[0]
        return policy_fn

    @torch.no_grad()
    def select_action(self, state, training=True):
        # ✅ GPU-ready action selection
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        mu = self.actor_critic.pi(state_t)

        if training:
            noise = torch.randn_like(mu) * self.noise_scale
            action = mu + noise
        else:
            action = mu

        # Clip and return numpy
        limit = self.env.action_space.high[0]
        return torch.clamp(action, -limit, limit).cpu().numpy()[0]

    def compute_target_values(self, next_states, rewards, dones):
        # ✅ Fully GPU computation
        next_actions = self.target_actor_critic.pi(next_states)
        target_q = self.target_actor_critic.q(next_states, next_actions)
        return rewards + self.options.gamma * (1 - dones) * target_q

    def replay(self):
        if len(self.replay_memory) <= self.options.batch_size:
            return

        minibatch = random.sample(self.replay_memory, self.options.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # ✅ GPU Batching
        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.array(actions), dtype=torch.float32, device=device)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=device)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones = torch.as_tensor(np.array(dones), dtype=torch.float32, device=device)

        # Update Q
        current_q = self.actor_critic.q(states, actions)
        with torch.no_grad():
            target_q = self.compute_target_values(next_states, rewards, dones)

        q_loss = F.mse_loss(current_q, target_q)
        self.optimizer_q.zero_grad()
        q_loss.backward()
        self.optimizer_q.step()

        # Update Policy (Policy Gradient)
        # Freeze Q-network to save gradients
        for p in self.actor_critic.q.parameters():
            p.requires_grad = False

        pi_loss = -self.actor_critic.q(states, self.actor_critic.pi(states)).mean()
        self.optimizer_pi.zero_grad()
        pi_loss.backward()
        self.optimizer_pi.step()

        # Unfreeze Q
        for p in self.actor_critic.q.parameters():
            p.requires_grad = True

    def memorize(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def train_episode(self):
        state, _ = self.env.reset()
        for _ in range(self.options.steps):
            action = self.select_action(state, training=True)

            # Step + interface adaptation
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.memorize(state, action, reward, next_state, done)

            # Update after every step (standard DDPG)
            self.replay()
            self.update_target_networks()

            state = next_state
            if done:
                break

    def __str__(self):
        return "Single DDPG"
