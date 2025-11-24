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
import pandas as pd # Added for data loading
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


def build_parser():
    parser = optparse.OptionParser(
        description="Run a specified RL algorithm on a specified domain."
    )
    parser.add_option(
        "-s",
        "--solver",
        dest="solver",
        type="string",
        default="random",
        help="Solver from " + str(avs.solvers),
    )
    parser.add_option(
        "-d",
        "--domain",
        dest="domain",
        type="string",
        default="Gridworld",
        help="Domain from OpenAI Gym",
    )
    parser.add_option(
        "-o",
        "--outfile",
        dest="outfile",
        default="out",
        help="Write results to FILE",
        metavar="FILE",
    )
    parser.add_option(
        "-x",
        "--experiment_dir",
        dest="experiment_dir",
        default="Experiments",
        help="Directory to save Tensorflow summaries in",
        metavar="FILE",
    )
    parser.add_option(
        "-e",
        "--episodes",
        type="int",
        dest="episodes",
        default=500,
        help="Number of episodes for training",
    )
    parser.add_option(
        "-t",
        "--steps",
        type="int",
        dest="steps",
        default=10000,
        help="Maximal number of steps per episode",
    )
    parser.add_option(
        "-l",
        "--layers",
        dest="layers",
        type="string",
        default="[24,24]",
        help='size of hidden layers in a Deep neural net. e.g., "[10,15]" creates a net where the'
        "Input layer is connected to a layer of size 10 that is connected to a layer of size 15"
        " that is connected to the output",
    )
    parser.add_option(
        "-a",
        "--alpha",
        dest="alpha",
        type="float",
        default=0.5,
        help="The learning rate (alpha) for updating state/action values",
    )
    parser.add_option(
        "-r",
        "--seed",
        type="int",
        dest="seed",
        default=random.randint(0, 9999999999),
        help="Seed integer for random stream",
    )
    parser.add_option(
        "-g",
        "--gamma",
        dest="gamma",
        type="float",
        default=1.00,
        help="The discount factor (gamma)",
    )
    parser.add_option(
        "-p",
        "--epsilon",
        dest="epsilon",
        type="float",
        default=0.1,
        help="Initial epsilon for epsilon greedy policies (might decay over time)",
    )
    parser.add_option(
        "-P",
        "--final_epsilon",
        dest="epsilon_end",
        type="float",
        default=0.1,
        help="The final minimum value of epsilon after decaying is done",
    )
    parser.add_option(
        "-c",
        "--decay",
        dest="epsilon_decay",
        type="float",
        default=0.99,
        help="Epsilon decay factor",
    )
    parser.add_option(
        "-m",
        "--replay",
        type="int",
        dest="replay_memory_size",
        default=500000,
        help="Size of the replay memory",
    )
    parser.add_option(
        "-N",
        "--update",
        type="int",
        dest="update_target_estimator_every",
        default=10000,
        help="Copy parameters from the Q estimator to the target estimator every N steps.",
    )
    parser.add_option(
        "-b",
        "--batch_size",
        type="int",
        dest="batch_size",
        default=32,
        help="Size of batches to sample from the replay memory",
    )
    parser.add_option(
        "--no-plots",
        help="Option to disable plots if the solver results any",
        dest="disable_plots",
        default=False,
        action="store_true",
    )
    return parser


def readCommand(argv):
    parser = build_parser()
    (options, args) = parser.parse_args(argv)
    return options


def getEnv(domain, render_mode=""):
    if domain == "Blackjack":
        return BlackjackEnv()
    elif domain == "Gridworld":
        return GridworldEnv()
    elif domain == "CliffWalking":
        return CliffWalkingEnv()
    elif domain == "WindyGridworld":
        return WindyGridworldEnv()
    elif domain == "Mag7Trading":
        # Load data dynamically when this domain is requested
        try:
            # Assumes 'data' folder is in the same directory where run.py is executed
            prices, indicators = load_data(data_dir="data")
            return MAG7TradingEnv(prices, indicators)
        except Exception as e:
            print(f"Error initializing MAG7TradingEnv: {e}")
            print("Ensure the 'data' directory exists and contains the CSV files.")
            sys.exit(1)
    else:
        try:
            return gym.make(domain, render_mode=render_mode)
        except:
            assert False, "Domain must be a valid (and installed) Gym environment"


def parse_list(string):
    string.strip()
    string = string[1:-1].split(",")  # Change "[0,1,2,3]" to '0', '1', '2', '3'
    l = []
    for n in string:
        l.append(int(n))
    return l


render = False


def on_press(key):
    from pynput import keyboard
    if key == keyboard.Key.esc:
        return False  # stop listener
    try:
        k = key.char  # single char keys
    except:
        k = key.name  # other keys
    if k in ["^"]:
        print(f"Key pressed: {k}")
        global render
        render = True


# ==========================================
# DATA LOADING UTILITIES (Added)
# ==========================================
def compute_indicators(df):
    """
    Computes technical indicators for a single asset dataframe.
    """
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
    """
    Loads MAG7 data from CSVs in the specified directory.
    """
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
        
        # Read CSV (Header row 0, Date index)
        df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
        
        # Coerce to numeric to handle potential sub-headers
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

    # Shape: (Time, Assets, Feats)
    prices_array = np.stack(all_prices, axis=0).transpose(1, 0, 2)
    inds_array = np.stack(all_inds, axis=0).transpose(1, 0, 2)
    
    return prices_array.astype(np.float32), inds_array.astype(np.float32)


def main(options):
    resultdir = "Results/"

    resultdir = os.path.abspath(f"./{resultdir}")
    options.experiment_dir = os.path.abspath(f"./{options.experiment_dir}")

    # Create result file if one doesn't exist
    print(os.path.join(resultdir, options.outfile + ".csv"))
    if not os.path.exists(os.path.join(resultdir, options.outfile + ".csv")):
        with open(
            os.path.join(resultdir, options.outfile + ".csv"), "w+"
        ) as result_file:
            result_file.write(AbstractSolver.get_out_header())

    random.seed(options.seed)
    env = getEnv(options.domain)
    env._max_episode_steps = options.steps + 1  # suppress truncation
    # if options.domain == "FlappyBird-v0":
    #     eval_env = env
    # else:
    eval_env = getEnv(options.domain, render_mode="human")
    print(f"\n---------- {options.domain} ----------")
    print(f"Domain state space is {env.observation_space}")
    print(f"Domain action space is {env.action_space}")
    print("-" * (len(options.domain) + 22) + "\n")
    try:
        options.layers = parse_list(options.layers)
    except ValueError:
        raise Exception(
            "layers argument doesnt follow int array conventions i.e., [<int>,<int>,<int>,...]"
        )
    except:
        pass
    solver = avs.get_solver_class(options.solver)(env, eval_env, options)

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(episode_lengths=[], episode_rewards=[])

    plt.ion()
    if not options.disable_plots:
        # Detects key press for rendering
        from pynput import keyboard
        listener = keyboard.Listener(on_press=on_press)
        listener.start()  # start listening on a separate thread


    with open(os.path.join(resultdir, options.outfile + ".csv"), "a+") as result_file:
        result_file.write("\n")
        for i_episode in range(options.episodes):
            solver.init_stats()
            solver.statistics[Statistics.Episode.value] += 1
            env.reset(seed=123)
            solver.train_episode()
            result_file.write(solver.get_stat() + "\n")
            # Decay epsilon
            if options.epsilon > options.epsilon_end:
                options.epsilon *= options.epsilon_decay
            # Update statistics
            stats.episode_rewards.append(solver.statistics[Statistics.Rewards.value])
            stats.episode_lengths.append(solver.statistics[Statistics.Steps.value])
            print(
                f"Episode {i_episode+1}: Reward {solver.statistics[Statistics.Rewards.value]}, Steps {solver.statistics[Statistics.Steps.value]}"
            )
            global render
            if render and not options.disable_plots:
                solver.run_greedy()
                render = False
            if (
                options.solver
                in ["ql", "sarsa", "aql", "dqn", "reinforce", "a2c", "ddpg"]
                and not options.disable_plots
            ):
                solver.plot(stats, int(0.1 * options.episodes), False)

    if not options.disable_plots:
        solver.run_greedy()
        solver.plot(stats, int(0.1 * options.episodes), True)
        if options.solver == "aql" and "MountainCar-v0" in str(env):
            solver.plot_q_function()
        solver.close()
    plt.ioff()

    return {"stats": stats, "solver": solver}


if __name__ == "__main__":
    options = readCommand(sys.argv)
    main(options)
