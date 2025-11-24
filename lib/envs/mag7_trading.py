import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MAG7TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    name = "Gridworld"
    def __init__(self, prices, indicators, max_k=100, initial_cash=1e6):
        """
        prices: np.array shape (T, 7, 5)  # [O,H,L,C,V] per asset
        indicators: np.array shape (T, 7, K)  # e.g. [RSI, MACD, MACD_signal, MACD_hist]
        """
        super().__init__()
        self.prices = prices
        self.indicators = indicators
        self.T, self.n_assets, self.n_price_feats = prices.shape
        _, _, self.n_ind_feats = indicators.shape

        self.max_k = max_k
        self.initial_cash = initial_cash

        # Observation Space:
        # 1. Normalized Cash (1)
        # 2. Normalized Portfolio Value (1)
        # 3. Normalized Positions (7)
        # 4. Flattened Prices for current step (7 * 5)
        # 5. Flattened Indicators for current step (7 * K)
        obs_dim = 2 + self.n_assets + self.n_assets * (self.n_price_feats + self.n_ind_feats)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action Space:
        # Continuous values between -max_k and +max_k for each of the 7 assets.
        # Negative = Sell, Positive = Buy.
        self.action_space = spaces.Box(
            low=-float(max_k), high=float(max_k), shape=(self.n_assets,), dtype=np.float32
        )

    def _decode_action(self, action_vec):
        """
        Converts continuous DDPG output to integer trade quantities.
        """
        # Round to nearest integer to get discrete number of shares
        return np.rint(action_vec).astype(int)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.cash = self.initial_cash
        self.positions = np.zeros(self.n_assets, dtype=np.int32)
        self._update_portfolio_value()
        obs = self._get_obs()
        return obs, {}  # Return obs and empty info dict

    def step(self, action):
        # 1. Clip action to valid range [-max_k, +max_k]
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 2. Decode to integer quantities
        a = self._decode_action(action)  

        prices_t = self.prices[self.t, :, 3]  # Current Close prices, shape (7,)

        # 3. Execute trades asset-by-asset
        for i in range(self.n_assets):
            desired = int(a[i])
            if desired == 0:
                continue

            price = prices_t[i]

            if desired > 0:
                # BUY logic
                max_buy = int(self.cash // price)
                size = min(desired, max_buy)
            else:
                # SELL logic
                max_sell = self.positions[i]
                size = -min(-desired, max_sell)

            if size != 0:
                self.cash -= size * price
                self.positions[i] += size

        # 4. Update Value & Time
        old_value = self.portfolio_value
        self._update_portfolio_value()

        self.t += 1
        terminated = self.t >= self.T - 1
        truncated = False

        # 5. Calculate Reward (Normalized Return)
        if old_value > 0:
            reward = (self.portfolio_value - old_value) / old_value
        else:
            reward = 0.0

        obs = self._get_obs()
        info = {
            "portfolio_value": float(self.portfolio_value),
            "step_return": float(reward),
        }
        
        # Return 5 values for gymnasium compatibility
        return obs, reward, terminated, truncated, info

    def _update_portfolio_value(self):
        prices_t = self.prices[self.t, :, 3] # Close prices
        self.portfolio_value = self.cash + float(np.dot(self.positions, prices_t))

    def _get_obs(self):
        prices_t = self.prices[self.t]      # (7, 5)
        inds_t = self.indicators[self.t]    # (7, K)

        # Normalize components for neural network stability
        cash_norm = self.cash / self.initial_cash
        value_norm = self.portfolio_value / self.initial_cash
        pos_norm = self.positions / self.max_k

        obs = np.concatenate([
            np.array([cash_norm, value_norm], dtype=np.float32),
            pos_norm.astype(np.float32),
            prices_t.reshape(-1).astype(np.float32),
            inds_t.reshape(-1).astype(np.float32),
        ])
        return obs