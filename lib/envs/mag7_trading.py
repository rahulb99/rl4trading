import gymnasium as gym
from gymnasium import spaces
import numpy as np

action_space_modes = ["continuous", "multidiscrete", "discrete"]

class MAG7TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    name = "MAG7TradingEnv"

    def __init__(self, prices, indicators, max_k, initial_cash, random_start=True, min_steps = 50, action_space_mode="multidiscrete", penalty = 0.01):
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
        self.action_space_mode = action_space_mode
        self.penalty = penalty

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
        
        if self.action_space_mode == action_space_modes[0]:
            # DDPG: Action is float in [-k, k].
            # 1. Clip to ensure bounds (DDPG noise might push it outside)
            # 2. Round to nearest integer to get discrete stock units
            # clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
            a = np.rint(action).astype(int)
        elif self.action_space_mode == action_space_modes[1]:
            # REINFORCE & A2C: Action is integer index in [0, 2k]
            # Convert to [-k, k] by shifting
            a = np.array(action, dtype=np.int32) - self.max_k
        elif self.action_space_mode == action_space_modes[2]:
            # DQN (Flattened): Action is a single integer index (0 to Total-1)
            # We unravel this index back into a vector of shape (n_assets,)
            # where each element is in [0, 2k]
            unraveled_indices = np.unravel_index(int(action), self.discrete_tuple)
            a = np.array(unraveled_indices, dtype=np.int32) - self.max_k
        else:
            raise ValueError("Invalid action space mode")

        a = np.clip(a, -self.max_k, self.max_k)
        
        # If running single ticker, 'a' might be a scalar or 1D array. 
        # We need to ensure it is iterable for the loop below.
        # if self.n_assets == 1:
        #     a = np.atleast_1d(a)
                
        prices_t = self.prices[self.t, :, 3] # Get Low price for all stock at time t
        
        # penalties = 0.0
        trade_happened = False
        
        # PASS 1: EXECUTE ALL SELLS
        for i in range(self.n_assets):
            desired = int(a[i])
            if desired < 0: # SELL
                price = prices_t[i]
                max_sell = self.positions[i]
                
                # if max_sell == 0:
                #     penalties -= self.penalty
                
                size = min(-desired, max_sell) 
                
                if size != 0:
                    self.cash += size * price
                    self.positions[i] -= size
                    trade_happened = True

        # PASS 2: EXECUTE ALL BUYS
        for i in range(self.n_assets):
            desired = int(a[i])
            if desired > 0: # BUY
                price = prices_t[i]
                max_buy = int(self.cash // price)
                size = min(desired, max_buy)
                
                # if size == 0 and desired > 0:
                #    print(f"DEBUG: Wanted to buy {desired} of Asset {i} at ${price:.2f}, but max_buy is {max_buy}. Cash: ${self.cash:.2f}")
                # if max_buy == 0:
                #      penalties -= self.penalty
                
                if size != 0:
                    self.cash -= size * price
                    self.positions[i] += size
                    trade_happened = True

        old_val = self.portfolio_value
        
        # 2. Increment Time Step (Move to T+1)
        self.t += 1
        
        # 3. Check Termination
        terminated = self.t >= self.T - 1
        truncated = False
        
        # 4. Update Portfolio Value using NEW prices (Prices at T+1)
        if not terminated:
            self.update_portfolio_value()
            
        # 5. Calculate Reward (Change in value due to market movement)
        raw_pnl = self.portfolio_value - old_val
        
        # # trade_happened = raw_pnl != 0
        
        agent_ret = 0.0
        if self.portfolio_value > 0:
            agent_ret = raw_pnl / self.portfolio_value
        
        reward = agent_ret
        
        # avg_t = np.mean(self.prices[self.t, :, 3])
        # avg_next = np.mean(self.prices[self.t+1, :, 3]) if self.t+1 < self.T else avg_t
        # mkt_ret = (avg_next - avg_t) / avg_t
        
        # reward = (agent_ret - mkt_ret) * 100.0
        # reward = np.clip(reward, -10.0, 10.0)
                
        # 3. HARD CLIP (The Safety Net)
        # Prevents the 100,000 reward from exploding the gradients
        # reward = np.clip(reward, -10.0, 10.0)
        # if reward == 0:
        #     print(action)
        
        # 4. The Penalty (Must be smaller than the clipped reward)
        # If reward is +/- 0.5, penalty should be 0.01
        
        # Check Cash Penalty (Force Trading)
        # Lower this from -0.01 to -0.001 so it's not fatal
        if not trade_happened:
            # If we are holding mostly cash (>90%), apply penalty
            if (self.cash / self.portfolio_value) > 0.9:
                reward -= self.penalty  # Force it to Buy
            # print(action)
        
        # cash_ratio = self.cash / self.portfolio_value
        # if cash_ratio > 0.5:
        #     # -0.01 per step is painful enough to force buying
        #     reward -= 0.01
        
        # reward = np.clip(reward, -1.0, 1.0)

        # self.t += 1
        # terminated = self.t >= self.T - 1
        # truncated = False
        # print("Reward: {}".format(reward))

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