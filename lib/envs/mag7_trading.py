import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MAG7TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    name = "MAG7TradingEnv"

    def __init__(self, prices, indicators, max_k, initial_cash, random_start=True, min_steps = 50, continous=True):
        super().__init__()
        self.prices = prices
        self.indicators = indicators
        self.T, self.n_assets, self.n_price_feats = prices.shape
        _, _, self.n_ind_feats = indicators.shape

        self.max_k = max_k
        self.initial_cash = initial_cash
        self.random_start = random_start 
        self.base_prices = None
        self.min_steps = min(max(min_steps, 1), self.T)
        self.continuous = continous

        obs_dim = 2 + self.n_assets + self.n_assets * (self.n_price_feats + self.n_ind_feats)
        
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        # )
        low = np.full(obs_dim, -np.inf, dtype=np.float32)
        high = np.full(obs_dim, np.inf, dtype=np.float32)
        
        # Set bounds for normalized positions
        pos_start = 2
        pos_end = pos_start + self.n_assets
        low[pos_start:pos_end] = -1.0
        high[pos_start:pos_end] = 1.0

        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32
        )
        
        if self.continuous:
            # DDPG Mode: Continuous output [-k, k]
            # We use float32 to be compatible with DDPG/TD3
            self.action_space = spaces.Box(
                low=-float(max_k), high=float(max_k), shape=(self.n_assets,), dtype=np.float32
            )
        else:
            # DQN Mode: MultiDiscrete output [0, 2k]
            # Each asset has 2*k + 1 possible actions
            self.action_space = spaces.MultiDiscrete([2 * max_k + 1] * self.n_assets)

    # def decode_action(self, action_vec):
    #     return np.rint(action_vec).astype(int) - self.max_k

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        if self.random_start: #Randomly start at timestep t to prevent overfitting
            # We can start anywhere from index 0 to (Total - min_steps)
            self.t = np.random.randint(0, self.T - self.min_steps)
        else:
            # Test Mode: Always start at Day 0 of this dataset
            self.t = 0
        # self.t = 0
            
        self.cash = self.initial_cash
        self.positions = np.zeros(self.n_assets, dtype=np.int32)
        
        self.base_prices = self.prices[self.t, :, 3] 
        self.base_prices = np.where(self.base_prices == 0, 1.0, self.base_prices)

        self.update_portfolio_value()
        return self.get_obs(), {}

    def step(self, action):
        # FIX: Explicitly handle PyTorch Tensors.
        # If action is a Tensor (requires_grad=True or False), detach and convert to numpy.
        if hasattr(action, 'detach'):
            action = action.detach().cpu().numpy()
        
        # Handle scalar tensors (0-d arrays)
        if np.isscalar(action) and hasattr(action, 'item'):
             action = action.item()

        
        if self.continuous:
            # DDPG: Action is float in [-k, k].
            # 1. Clip to ensure bounds (DDPG noise might push it outside)
            # 2. Round to nearest integer to get discrete stock units
            clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
            a = np.rint(clipped_action).astype(int)
        else:
            # DQN: Action is integer index in [0, 2k]
            # Convert to [-k, k] by shifting
            a = np.array(action, dtype=np.int32) - self.max_k
        

        prices_t = self.prices[self.t, :, 3] # Get Low price for all stock at time t

        for i in range(self.n_assets):
            desired = int(a[i])
            if desired == 0: continue

            price = prices_t[i]
            if desired > 0: # Buy
                max_buy = int(self.cash // price)
                size = min(desired, max_buy)
            else: # Sell
                max_sell = self.positions[i]
                size = -min(-desired, max_sell)

            if size != 0:
                self.cash -= size * price
                self.positions[i] += size

        old_val = self.portfolio_value
        self.update_portfolio_value()
        
        # Returns for Reward
        agent_ret = 0.0
        if old_val > 0:
            agent_ret = (self.portfolio_value - old_val) / old_val

        # Market Return (Average of assets)
        avg_t = np.mean(self.prices[self.t, :, 3])
        avg_next = np.mean(self.prices[self.t+1, :, 3]) if self.t+1 < self.T else avg_t
        mkt_ret = (avg_next - avg_t) / avg_t

        # Differential Reward (Alpha)
        reward = (agent_ret - mkt_ret) * 100.0
        reward = np.clip(reward, -10.0, 10.0)

        self.t += 1
        terminated = self.t >= self.T - 1
        truncated = False

        return self.get_obs(), reward, terminated, truncated, {
            "portfolio_value": float(self.portfolio_value)
        }

    def update_portfolio_value(self):
        prices_t = self.prices[self.t, :, 3]
        self.portfolio_value = self.cash + float(np.dot(self.positions, prices_t))

    def get_obs(self):
        prices_t = self.prices[self.t]
        inds_t = self.indicators[self.t]

        cash_norm = self.cash / self.initial_cash
        value_norm = self.portfolio_value / self.initial_cash
        pos_norm = self.positions / self.max_k
        
        prices_norm = prices_t / self.base_prices[:, None]

        obs = np.concatenate([
            np.array([cash_norm, value_norm], dtype=np.float32),
            pos_norm.astype(np.float32),
            prices_norm.reshape(-1).astype(np.float32),
            inds_t.reshape(-1).astype(np.float32),
        ])
        return obs