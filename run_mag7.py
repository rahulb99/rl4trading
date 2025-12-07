# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).

import gymnasium as gym
import optparse
import sys
import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

gym.logger.set_level(40)
if "../" not in sys.path: sys.path.append("../")

from lib import plotting
from Solvers.Abstract_Solver import AbstractSolver, Statistics
import Solvers.Available_solvers as avs
from lib.envs.mag7_trading import MAG7TradingEnv

# ==========================================
# 1. DATA LOADING & SPLITTING
# ==========================================
def compute_indicators(df):
    close = df['Close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    # macd = (exp1 - exp2) / close

    return pd.DataFrame({'RSI': rsi/100.0, 
                         'MACD': macd
                          }).fillna(0)

def load_data_raw(data_source="data", selected_ticker="all"):
    """
    Loads data. 
    - If data_source is a file: loads that specific file.
    - If data_source is a dir:
        - If selected_ticker is "all": loads the default Mag7 list.
        - If selected_ticker is "XYZ": loads only XYZ.csv from that dir.
    """
    price_feats = ['Open', 'High', 'Low', 'Close', 'Volume']
    dfs = {}
    
    # 1. Determine Tickers and Paths
    if os.path.isfile(data_source):
        print(f"Loading single file target: {os.path.abspath(data_source)}...")
        ticker_name = os.path.splitext(os.path.basename(data_source))[0]
        tickers = [ticker_name]
        file_paths = [data_source]
        
    elif os.path.isdir(data_source):
        print(f"Loading from directory: {os.path.abspath(data_source)}")
        
        if selected_ticker == "all":
            print("Selection: ALL default tickers")
            tickers = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']
        else:
            print(f"Selection: Single ticker '{selected_ticker}'")
            tickers = [selected_ticker]
            
        file_paths = [os.path.join(data_source, f"{t}.csv") for t in tickers]
        
    else:
        raise FileNotFoundError(f"Input {data_source} is not a valid file or directory.")

    common_index = None

    # 2. Load Loop
    for t, path in zip(tickers, file_paths):
        if not os.path.exists(path): 
            raise FileNotFoundError(f"Missing file for ticker {t} at: {path}")
        
        df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
        for col in price_feats: 
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        dfs[t] = df
        
        if common_index is None: 
            common_index = df.index
        else: 
            common_index = common_index.intersection(df.index)

    common_index = common_index.sort_values()
    print(f"Loaded {len(tickers)} asset(s). Total Common Days: {len(common_index)}")

    all_prices, all_inds = [], []
    for t in tickers:
        df = dfs[t].reindex(common_index).fillna(method='ffill')
        all_prices.append(df[price_feats].values)
        all_inds.append(compute_indicators(df).values)

    # (Time, Assets, Feats)
    prices = np.stack(all_prices, axis=0).transpose(1, 0, 2).astype(np.float32)
    inds = np.stack(all_inds, axis=0).transpose(1, 0, 2).astype(np.float32)
    return prices, inds

# Cache data so we don't reload 4 times
CACHED_PRICES = None
CACHED_INDS = None

def getEnv(options, split='train', split_rate=0.8):
    """
    split: 'train' (first 80%) or 'test' (last 20%)
    """
    global CACHED_PRICES, CACHED_INDS
    
    if CACHED_PRICES is None:
        # Pass the ticker option to the loader
        CACHED_PRICES, CACHED_INDS = load_data_raw(options.data_dir, options.ticker)
        
    T = len(CACHED_PRICES)
    split_idx = int(T * split_rate) # 80/20 Split
    
    if split == 'train':
        prices = CACHED_PRICES[:split_idx]
        indicators = CACHED_INDS[:split_idx]
        random_start = options.random_start
        print(f"Creating TRAIN Env: {len(prices)} days.")
    elif split == 'test':
        prices = CACHED_PRICES[split_idx:]
        indicators = CACHED_INDS[split_idx:]
        random_start=False
        print(f"Creating TEST Env: {len(prices)} days.")
    else:
        raise Exception("Invalid split")
    
    return MAG7TradingEnv(prices=prices, 
                          indicators=indicators, 
                          max_k=options.max_k,
                          initial_cash=options.initial_cash,
                          random_start=random_start,
                          min_steps=options.steps,
                          action_space_mode=options.action_space_mode)

# ==========================================
# 2. PLOTTING & BACKTEST LOGIC
# ==========================================

def plot_training_curve(stats, model= "ML", outfile="out"):
    rewards = pd.Series(stats.episode_rewards)
    window_size = max(10, int(len(rewards) * 0.1))
    rolling_avg = rewards.rolling(window=window_size, min_periods=1).mean()

    # Save Training Data to CSV
    df_training = pd.DataFrame({
        "Episode": range(1, len(rewards) + 1),
        "Reward": rewards,
        f"Moving_Avg_Window_{window_size}": rolling_avg
    })

    if not os.path.exists("Results"): 
        os.makedirs("Results")

    csv_path = os.path.join("Results", f"{outfile}_training.csv")
    df_training.to_csv(csv_path, index=False)
    print(f"Training data saved to {csv_path}")

    # Generate the Plot
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Return', alpha=0.3, color='blue')
    plt.plot(rolling_avg, label=f'Avg Return (MA-{window_size})', color='red', linewidth=2)
    
    plt.title("Mag7 {} Return".format(model))
    plt.xlabel("Episode")
    plt.ylabel("Return ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    chart_path = os.path.join("Results", f"{outfile}_training.png")
    plt.savefig(chart_path)
    print(f"Training chart saved to {chart_path}")
    if not options.disable_plots:
        plt.show()
    else:
        plt.clf()

def run_backtest(env, solver, options):
    """Runs a single full pass over the Test Data."""
    print("\n>>> STARTING BACKTEST ON 20% TEST DATA <<<")
    state, _ = env.reset()
    done = False
    
    agent_vals = []
    market_vals = []
    
    initial_cash = env.initial_cash
    start_prices = env.prices[0, :, 3]
    start_prices = np.where(start_prices == 0, 1e-9, start_prices)
    
    # Dynamically calculate shares based on number of assets found
    num_assets = len(start_prices)
    shares = (initial_cash / float(num_assets)) / start_prices
    
    while not done:
        action = solver.select_action(state, training = False) 
        state, reward, term, trunc, info = env.step(action)
        done = term or trunc
        
        agent_vals.append(info['portfolio_value'])
        
        curr_prices = env.prices[env.t, :, 3]
        market_vals.append(np.sum(shares * curr_prices))
    
    if options.outfile:
        results_dir = "Results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        df_results = pd.DataFrame({
            "Time_Step": range(len(agent_vals)),
            "Agent_Value": agent_vals,
            "Market_Value": market_vals
        })
        
        fname = options.outfile
        full_path = os.path.join(results_dir, fname)
        data_path = full_path + '.csv'
            
        df_results.to_csv(data_path, index=False)
        print(f"Saved test results saved to {data_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(agent_vals, label="Agent Portfolio", color='blue')
    plt.plot(market_vals, label="Buy & Hold Portfolio", color='orange', linestyle='--')
    plt.title("Mag7 Trading test on {} for {} days".format(str(solver), len(agent_vals)+1))
    plt.xlabel("Time Steps (Days)")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    
    chart_path = full_path + '.png'
    plt.savefig(chart_path)
    
    print(f"Saved test results saved to {data_path}")
    print(f"Saved chart saved to {chart_path}")
    
    agent_ret = (agent_vals[-1] - initial_cash) / initial_cash
    mkt_ret = (market_vals[-1] - initial_cash) / initial_cash
    print(f"Final Agent Return: {agent_ret:.2%}")
    print(f"Final Market Return: {mkt_ret:.2%}")
    
    if not options.disable_plots:
        plt.show()
    else:
        plt.clf()

def build_parser():
    parser = optparse.OptionParser()
    parser.add_option("-s", "--solver", dest="solver", default="random")
    parser.add_option("-o", "--outfile", dest="outfile", default="out")
    parser.add_option("-e", "--episodes", type="int", dest="episodes", default=100)
    parser.add_option("-t", "--steps", type="int", dest="steps", default=1000)
    parser.add_option("-l", "--layers", dest="layers", default="[64,64]")
    parser.add_option("-a", "--alpha", type="float", dest="alpha", default=0.001)
    parser.add_option("-r", "--seed", type="int", dest="seed", default=42)
    parser.add_option("-g", "--gamma", type="float", dest="gamma", default=0.99)
    parser.add_option("-p", "--epsilon", type="float", dest="epsilon", default=0.1)
    parser.add_option("-P", "--final_epsilon", type="float", dest="epsilon_end", default=0.1)
    parser.add_option("-c", "--decay", type="float", dest="epsilon_decay", default=0.99)
    parser.add_option("-n", "--noise_scale", type="float", dest="noise_scale", default=10)
    parser.add_option("-i", "--initial_cash", type="int", dest="initial_cash", default=1e6)
    parser.add_option("-k", "--max_k", type="int", dest="max_k", default=100)
    parser.add_option("-y", "--entropy_pct", type="float", dest="entropy_pct", default=0.01)
    parser.add_option("-d", "--data_dir", dest="data_dir", default="data")
    parser.add_option("-A", "--action_space_mode", dest="action_space_mode", default="multidiscrete")
    
    # [NEW] Ticker selection option
    parser.add_option("-T", "--ticker", dest="ticker", default="all",
                      help="Specific ticker to select (e.g., 'AAPL') or 'all' for default set.")

    parser.add_option("-m", "--replay", type="int", dest="replay_memory_size", default=100000)
    parser.add_option("-N", "--update", type="int", dest="update_target_estimator_every", default=1000)
    parser.add_option("-b", "--batch_size", type="int", dest="batch_size", default=64)
    parser.add_option("-x", "--experiment_dir", dest="experiment_dir", default="Experiments")
    
    parser.add_option("--no-plots", action="store_true", dest="disable_plots", default=False)
    parser.add_option("--rand-start", action="store_true", dest="random_start", default=False)
    parser.add_option("--cont", action="store_true", dest="continous", default=False)
    parser.add_option("--gpu", action="store_true", dest="gpu", default=False)
    return parser

def parse_list(s):
    if s.startswith("["): s = s[1:-1]
    return [int(n) for n in s.split(",") if n.strip()]

def main(options):
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)

    # 1. Create Environments
    train_env = getEnv(split='train', options=options)
    test_env = getEnv(split='test', options=options)
    
    # 2. Setup Solver
    try: options.layers = parse_list(options.layers)
    except: pass
    solver = avs.get_solver_class(options.solver)(train_env, test_env, options)
    
    # 3. Training Loop
    print(f"\n>>> TRAINING ON 80% DATA ({options.episodes} Episodes) <<<")
    stats = plotting.EpisodeStats(episode_lengths=[], episode_rewards=[])
    
    for i in range(options.episodes):
        solver.init_stats()
        train_env.reset(seed=None) # Random start
        solver.train_episode()
        
        # Logging
        rew = solver.statistics[Statistics.Rewards.value]
        stats.episode_rewards.append(rew)
        stats.episode_lengths.append(solver.statistics[Statistics.Steps.value])
        
        print(f"Episode {i+1}: Reward {rew:.4f}")

    # 4. Plot Training Stats (Return & Return Avg)
    plot_training_curve(stats, model=str(solver), outfile=options.outfile)

    # 5. Backtest Phase
    run_backtest(test_env, solver, options)
        
    return {"stats": stats, "solver": solver}

if __name__ == "__main__":
    parser = build_parser()
    (options, args) = parser.parse_args()
    main(options)