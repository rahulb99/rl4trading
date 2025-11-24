import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MAG7TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    name = "Mag7Trading"

    def __init__(self, prices, indicators, max_k=100, initial_cash=1e6):
        """
        prices: (T, 7, 5) [O,H,L,C,V]
        indicators: (T, 7, K)
        """
        super().__init__()
        self.prices = prices
        self.indicators = indicators
        self.T, self.n_assets, self.n_price_feats = prices.shape
        _, _, self.n_ind_feats = indicators.shape

        self.max_k = max_k
        self.initial_cash = initial_cash

        # --- IMPROVEMENT: Normalization ---
        # We will normalize prices relative to the price at t=0 of the episode
        self.base_prices = None 

        obs_dim = 2 + self.n_assets + self.n_assets * (self.n_price_feats + self.n_ind_feats)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-float(max_k), high=float(max_k), shape=(self.n_assets,), dtype=np.float32
        )

    def _decode_action(self, action_vec):
        return np.rint(action_vec).astype(int)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- IMPROVEMENT: Random Start for Training ---
        # Instead of always starting at day 0, start at a random day
        # so the agent learns different market conditions.
        # We leave 250 steps margin for an episode.
        if self.T > 300: 
            self.t = np.random.randint(0, self.T - 250)
        else:
            self.t = 0
            
        self.cash = self.initial_cash
        self.positions = np.zeros(self.n_assets, dtype=np.int32)
        
        # Capture base prices for normalization
        self.base_prices = self.prices[self.t, :, 3] # Close prices at start
        # Avoid division by zero
        self.base_prices = np.where(self.base_prices == 0, 1.0, self.base_prices)

        self._update_portfolio_value()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        a = self._decode_action(action)

        prices_t = self.prices[self.t, :, 3]

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

        old_value = self.portfolio_value
        self._update_portfolio_value()

        self.t += 1
        terminated = self.t >= self.T - 1
        truncated = False

        if old_value > 0:
            reward = (self.portfolio_value - old_value) / old_value
        else:
            reward = 0.0

        # Magnify reward for faster learning (DDPG likes rewards ~1.0, not 0.001)
        reward = reward * 100.0 

        return self._get_obs(), reward, terminated, truncated, {
            "portfolio_value": float(self.portfolio_value)
        }

    def _update_portfolio_value(self):
        prices_t = self.prices[self.t, :, 3]
        self.portfolio_value = self.cash + float(np.dot(self.positions, prices_t))

    def _get_obs(self):
        prices_t = self.prices[self.t]      # (7, 5)
        inds_t = self.indicators[self.t]    # (7, K)

        cash_norm = self.cash / self.initial_cash
        value_norm = self.portfolio_value / self.initial_cash
        pos_norm = self.positions / self.max_k

        # --- IMPROVEMENT: Normalize Prices ---
        # Divide current prices by the base prices (Start of episode)
        # This makes price inputs ~1.0 instead of ~200.0
        # Broadcast base_prices (7,) to (7, 5)
        prices_norm = prices_t / self.base_prices[:, None]

        obs = np.concatenate([
            np.array([cash_norm, value_norm], dtype=np.float32),
            pos_norm.astype(np.float32),
            prices_norm.reshape(-1).astype(np.float32), # Normalized prices
            inds_t.reshape(-1).astype(np.float32),
        ])
        return obs