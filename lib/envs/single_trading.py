import gymnasium as gym
from gymnasium import spaces
import numpy as np

action_space_modes = ["continuous", "multidiscrete", "discrete"]


class SingleStockTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, prices, indicators, max_k, initial_cash=1e5):
        super().__init__()
        self.prices = prices      # np.array shape (T, 5) -> [O,H,L,C,V]
        self.indicators = indicators  # np.array shape (T, K) -> [MACD, RSI, ...]
        self.max_k = max_k
        self.initial_cash = initial_cash

        obs_dim = 3 + self.prices.shape[1] + self.indicators.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(2 * max_k + 1)  # {-k,...,0,...,k}
        self.name = "SingleStockTradingEnv"
        
        if self.action_space_mode == action_space_modes[0]:
            # DDPG Mode: Continuous output [-k, k]
            # We use float32 to be compatible with DDPG/TD3
            self.action_space = spaces.Box(
                low=-float(max_k), high=float(max_k), shape=(self.n_assets,), dtype=np.float32
            )
        elif self.action_space_mode == action_space_modes[1]:
            # REINFORCE & A2C Mode: MultiDiscrete output [0, 2k]
            # Each asset has 2*k + 1 possible actions
            self.action_space = spaces.MultiDiscrete([2 * max_k + 1] * self.n_assets)
        elif self.action_space_mode == action_space_modes[2]:
            # DQN Mode (Flattened): One integer represents a combination of all assets
            # WARNING: Size is (2k+1)^N. Keep max_k small!
            self.discrete_tuple = tuple([2 * max_k + 1] * self.n_assets) # Used for decoding discrete actions
            self.action_space = spaces.Discrete((2 * max_k + 1) ** self.n_assets)
        else:
            raise ValueError("Invalid action space mode")

    # def _decode_action(self, action_idx):
    #   # maps 0..2k -> -k..k
    #   a = action_idx - self.max_k
    #   return int(a)

    def reset(self, seed=None, options=None):
      super().reset(seed=seed)
      self.t = 0
      self.cash = self.initial_cash
      self.position = 0  # shares
      self._update_portfolio_value()
      obs = self._get_obs()
      return obs, {}

    def step(self, action_idx):
        a = self._decode_action(action_idx)

        price = self.prices[self.t, 3]  # close price

        # max buy/sell given constraints
        max_buy = int(self.cash // price)
        max_sell = self.position

        if a > 0:
            size = min(a, max_buy)
        elif a < 0:
            size = -min(-a, max_sell)
        else:
            size = 0

        # execute trade
        self.cash -= size * price
        self.position += size

        old_value = self.portfolio_value
        self._update_portfolio_value()

        self.t += 1
        terminated = self.t >= len(self.prices) - 1
        truncated = False  # or limit max_steps

        if old_value > 0:
          reward = (self.portfolio_value - old_value) / old_value  # r = (v' - v)/v
        else:
          reward = 0.0

        obs = self._get_obs() if not terminated else np.zeros_like(self._get_obs())
        info = {"portfolio_value": self.portfolio_value}

        return obs, reward, terminated, truncated, info

    def _update_portfolio_value(self):
        price = self.prices[self.t, 3]
        self.portfolio_value = self.cash + self.position * price

    def _get_obs(self):
      """
      Normalized observation: portfolio state + normalized market features.

      Returns shape: (obs_dim,) where obs_dim = 3 + 5 + K
      """
      # Normalization scales (computed once at init for consistency)
      if not hasattr(self, 'price_scale'):
          self.price_scale = self.prices[0, 3]  # First close price
          self.vol_scale = np.median(self.prices[:, 4])  # Median volume
          # Clip to avoid div-by-zero/insane values
          self.price_scale = max(self.price_scale, 1.0)
          self.vol_scale = max(self.vol_scale, 1.0)

      # Portfolio features
      cash_norm = self.cash / self.initial_cash

      # Fix: Normalize position by estimated max shares (initial cash / initial price)
      # This prevents the feature from exploding > 1.0 when fully invested
      max_possible_shares = self.initial_cash / self.price_scale
      pos_norm = self.position / max_possible_shares

      val_norm = self.portfolio_value / self.initial_cash

      portfolio_feats = np.array([cash_norm, pos_norm, val_norm], dtype=np.float32)

      # Raw price features at time t
      raw_price_feats = self.prices[self.t]  # [Open, High, Low, Close, Volume]

      # Normalize OHLC prices by first close (puts prices in [0.5, 2.0] range typically)
      ohlc_norm = raw_price_feats[:4] / self.price_scale

      # Log-normalize volume (puts volume in [-2, 4] range typically)
      vol_raw = raw_price_feats[4]
      vol_norm = np.log(vol_raw / self.vol_scale + 1.0)

      # Combined normalized price features
      price_feats_norm = np.concatenate([ohlc_norm, [vol_norm]], dtype=np.float32)

      # Raw technical indicators (assumed pre-normalized or small magnitude)
      ind_feats = self.indicators[self.t].astype(np.float32)

      # NaN safety net (in case indicators have warmup NaNs)
      ind_feats = np.nan_to_num(ind_feats, nan=0.0)

      # Full observation vector
      obs = np.concatenate([
          portfolio_feats,      # shape (3,)
          price_feats_norm,     # shape (5,)
          ind_feats             # shape (K,)
      ], dtype=np.float32)

      return obs

    def __name__(self):
        return "SingleStockTradingEnv"
