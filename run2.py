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

gym.logger.set_level(40)

if "../" not in sys.path:
    sys.path.append("../")

from lib import plotting
from Solvers.Abstract_Solver import AbstractSolver, Statistics
import Solvers.Available_solvers as avs
from lib.envs.gridworld import GridworldEnv
from lib.envs.blackjack import BlackjackEnv
from lib.envs.cliff_walking import CliffWalkingEnv
from lib.envs.windy_gridworld import WindyGridworldEnv
from lib.envs.mag7_trading import MAG7TradingEnv 

import matplotlib
import matplotlib.pyplot as plt


# ==========================================
# DATA LOADING UTILITIES
# ==========================================
def compute_indicators(df):
    close = df['Close']
    
    # 1. RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 2. MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    # 3. SMA Ratio
    sma20 = close.rolling(window=20).mean()
    sma_ratio = close / sma20

    indicators = pd.DataFrame({
        'RSI': rsi / 100.0,
        'MACD': macd,
        'Signal': signal,
        'SMA_Ratio': sma_ratio
    })
    return indicators.fillna(0)

def load_data(data_dir="data"):
    tickers = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']
    price_feats = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    all_prices = []
    all_inds = []
    common_index = None

    print(f"Loading data from {os.path.abspath(data_dir)}...")

    dfs = {}
    for t in tickers:
        path = os.path.join(data_dir, f"{t}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find {path}")
        
        df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
        
        # Coerce to numeric
        for col in price_feats:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        dfs[t] = df
        
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)

    common_index = common_index.sort_values()
    print(f"Found {len(common_index)} common trading days.")

    for t in tickers:
        df = dfs[t].reindex(common_index).fillna(method='ffill')
        prices_df = df[price_feats]
        inds_df = compute_indicators(df)
        all_prices.append(prices_df.values)
        all_inds.append(inds_df.values)

    prices_array = np.stack(all_prices, axis=0).transpose(1, 0, 2)
    inds_array = np.stack(all_inds, axis=0).transpose(1, 0, 2)
    
    return prices_array.astype(np.float32), inds_array.astype(np.float32)


# ==========================================
# MAIN PARSER & LOGIC
# ==========================================

def build_parser():
    parser = optparse.OptionParser(
        description="Run a specified RL algorithm on a specified domain."
    )
    parser.add_option("-s", "--solver", dest="solver", type="string", default="random", help="Solver from " + str(avs.solvers))
    parser.add_option("-d", "--domain", dest="domain", type="string", default="Mag7Trading", help="Domain name")
    parser.add_option("-o", "--outfile", dest="outfile", default="out", help="Write results to FILE")
    parser.add_option("-x", "--experiment_dir", dest="experiment_dir", default="Experiments", help="Directory to save Tensorflow summaries in")
    parser.add_option("-e", "--episodes", type="int", dest="episodes", default=500, help="Number of episodes")
    parser.add_option("-t", "--steps", type="int", dest="steps", default=10000, help="Maximal number of steps per episode")
    parser.add_option("-l", "--layers", dest="layers", type="string", default="[64,64]", help='size of hidden layers')
    parser.add_option("-a", "--alpha", dest="alpha", type="float", default=0.5, help="Learning rate")
    parser.add_option("-r", "--seed", type="int", dest="seed", default=random.randint(0, 9999999999), help="Seed integer")
    parser.add_option("-g", "--gamma", dest="gamma", type="float", default=1.00, help="Discount factor")
    parser.add_option("-p", "--epsilon", dest="epsilon", type="float", default=0.1, help="Initial epsilon")
    parser.add_option("-P", "--final_epsilon", dest="epsilon_end", type="float", default=0.1, help="Final epsilon")
    parser.add_option("-c", "--decay", dest="epsilon_decay", type="float", default=0.99, help="Epsilon decay")
    parser.add_option("-m", "--replay", type="int", dest="replay_memory_size", default=500000, help="Replay memory size")
    parser.add_option("-N", "--update", type="int", dest="update_target_estimator_every", default=10000, help="Copy params every N steps")
    parser.add_option("-b", "--batch_size", type="int", dest="batch_size", default=32, help="Batch size")
    parser.add_option("--no-plots", help="Disable plots", dest="disable_plots", default=False, action="store_true")
    return parser


def readCommand(argv):
    parser = build_parser()
    (options, args) = parser.parse_args(argv)
    return options


def getEnv(domain, render_mode=""):
    d_lower = domain.lower()
    if d_lower == "blackjack": return BlackjackEnv()
    elif d_lower == "gridworld": return GridworldEnv()
    elif d_lower == "cliffwalking": return CliffWalkingEnv()
    elif d_lower == "windygridworld": return WindyGridworldEnv()
    elif d_lower in ["mag7trading", "mag7tra", "mag7"]:
        try:
            prices, indicators = load_data(data_dir="data")
            return MAG7TradingEnv(prices, indicators)
        except Exception as e:
            print(f"Error initializing MAG7TradingEnv: {e}")
            sys.exit(1)
    else:
        try: return gym.make(domain, render_mode=render_mode)
        except: assert False, f"Domain '{domain}' must be a valid Gym environment"


def parse_list(string):
    string = string.strip()
    if string.startswith("[") and string.endswith("]"): string = string[1:-1]
    parts = string.split(",")
    l = []
    for n in parts:
        if n.strip(): l.append(int(n.strip()))
    return l

def calculate_buy_and_hold(env, start_t, end_t, initial_cash):
    """
    Calculates the return of a Buy & Hold strategy for the specific time period.
    Strategy: Split initial cash equally among all 7 assets at start_t. Hold until end_t.
    """
    try:
        # Get prices at start and end (using Close price, index 3)
        # Shape: (7,)
        start_prices = env.prices[start_t, :, 3]
        end_prices = env.prices[end_t, :, 3]
        
        # Avoid division by zero
        start_prices = np.where(start_prices == 0, 1e-9, start_prices)

        # Allocate cash equally
        cash_per_asset = initial_cash / 7.0
        
        # Calculate shares bought (fractional shares allowed for simplicity of baseline)
        shares = cash_per_asset / start_prices
        
        # value at end
        final_value = np.sum(shares * end_prices)
        
        # Return percentage (e.g., 0.05 for 5%)
        return (final_value - initial_cash) / initial_cash
    except:
        return 0.0

render = False
def on_press(key):
    from pynput import keyboard
    if key == keyboard.Key.esc: return False
    try: k = key.char
    except: k = key.name
    if k in ["^"]:
        global render
        render = True

def plot_comparison(agent_returns, market_returns, window=10):
    """Plots Agent vs Market comparison."""
    if len(agent_returns) < window: return

    # Smoothing
    agent_smooth = pd.Series(agent_returns).rolling(window=window).mean()
    market_smooth = pd.Series(market_returns).rolling(window=window).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(agent_smooth, label="DDPG Agent Return", color='blue')
    plt.plot(market_smooth, label="Market (Buy & Hold) Return", color='orange', linestyle='--')
    plt.title("Agent vs Market Performance (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Return (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

def main(options):
    resultdir = os.path.abspath("./Results/")
    if not os.path.exists(resultdir): os.makedirs(resultdir)
    
    random.seed(options.seed)
    
    # Initialize Environment
    env = getEnv(options.domain)
    env._max_episode_steps = options.steps + 1
    eval_env = getEnv(options.domain, render_mode="human")
    
    print(f"\n---------- {options.domain} ----------")
    try: options.layers = parse_list(options.layers)
    except: pass
    
    solver = avs.get_solver_class(options.solver)(env, eval_env, options)
    
    # Trackers
    stats = plotting.EpisodeStats(episode_lengths=[], episode_rewards=[])
    market_returns_history = []
    agent_returns_history = [] # We'll track raw portfolio return, not just "reward"

    if not options.disable_plots:
        plt.ion()
        from pynput import keyboard
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    with open(os.path.join(resultdir, options.outfile + ".csv"), "w+") as result_file:
        result_file.write(AbstractSolver.get_out_header() + ",MarketReturn\n")
        
        for i_episode in range(options.episodes):
            solver.init_stats()
            solver.statistics[Statistics.Episode.value] += 1
            
            # 1. Reset Env (Random Start)
            env.reset(seed=None) 
            start_t = env.t  # Record start time
            
            # 2. Train Episode
            solver.train_episode()
            end_t = env.t    # Record end time
            
            # 3. Calculate Comparisons
            # Agent Return (Unnormalized): (Current Portfolio - Initial) / Initial
            agent_ret = (env.portfolio_value - env.initial_cash) / env.initial_cash
            
            # Market Return (Buy & Hold) for the same period
            market_ret = calculate_buy_and_hold(env, start_t, end_t, env.initial_cash)
            
            # Log
            agent_returns_history.append(agent_ret * 100) # Convert to %
            market_returns_history.append(market_ret * 100)
            
            stats.episode_rewards.append(solver.statistics[Statistics.Rewards.value])
            stats.episode_lengths.append(solver.statistics[Statistics.Steps.value])
            
            print(f"Episode {i_episode+1}: Agent Reward {solver.statistics[Statistics.Rewards.value]:.4f} | Agent Return: {agent_ret*100:.2f}% | Market Return: {market_ret*100:.2f}%")
            
            # Decay epsilon
            if options.epsilon > options.epsilon_end:
                options.epsilon *= options.epsilon_decay

            global render
            if render and not options.disable_plots:
                solver.run_greedy()
                render = False
            
            # Plot intermediate
            if not options.disable_plots and i_episode > 0 and i_episode % 10 == 0:
                 # Minimal plot update to avoid blocking
                 pass

    # Final Plot
    if not options.disable_plots:
        plot_comparison(agent_returns_history, market_returns_history)
        # Also plot the standard stats
        # solver.plot(stats, int(0.1 * options.episodes), True)
        
    return {"stats": stats, "solver": solver}


if __name__ == "__main__":
    options = readCommand(sys.argv)
    main(options)