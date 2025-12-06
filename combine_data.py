import pandas as pd
import os
import matplotlib.pyplot as plt


def plot_training_performance(file_path, initial_cash = 1000000):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # 1. Load Data
    df = pd.read_csv(file_path)
    
    # Check if 'Final_Portfolio_Value' exists (from your new code)
    # If not, fallback to 'Reward' but warn the user
    if 'Final_Portfolio_Value' in df.columns:
        metric = 'Final_Portfolio_Value'
        ylabel = 'Portfolio Value ($)'
        title_metric = 'Wealth'
        baseline = initial_cash
    else:
        metric = 'Reward'
        ylabel = 'Reward Points'
        title_metric = 'Reward'
        baseline = 0
        print("Warning: 'Final_Portfolio_Value' not found. Plotting 'Reward' instead.")

    # 2. Calculate Statistics
    # EMA (Exponential Moving Average) - Reacts faster than Simple Moving Average
    span = max(10, int(len(df) * 0.05)) # Dynamic window (5% of total episodes)
    df['EMA'] = df[metric].ewm(span=span, adjust=False).mean()
    
    # Rolling Standard Deviation (Volatility)
    df['Std'] = df[metric].rolling(window=span).std()

    # 3. Plotting
    plt.figure(figsize=(12, 6))

    # A. The Baseline (Breakeven point)
    plt.axhline(y=baseline, color='black', linestyle='--', alpha=0.5, label='Baseline')

    # B. The Raw Data (Faint dots) to show the "real" chaos
    plt.scatter(df['Episode'], df[metric], color='gray', alpha=0.1, s=10, label='Raw Episode Result')

    # C. The Trend Line (EMA)
    plt.plot(df['Episode'], df['EMA'], color='#1f77b4', linewidth=2, label=f'Trend (EMA-{span})')

    # D. The Volatility Zone (Shading)
    plt.fill_between(
        df['Episode'], 
        df['EMA'] - df['Std'], 
        df['EMA'] + df['Std'], 
        color='#1f77b4', alpha=0.15, label='Volatility Range'
    )

    plt.title(f"Training Progress: Agent {title_metric} over Time")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_img = file_path.replace('.csv', '_plot.png')
    plt.savefig(output_img)
    print(f"Plot saved to: {output_img}")
    plt.show()

# --- Configuration ---
# Update this path to the folder where your CSV files are stored
# Example: folder_path = r"C:\Users\Name\Documents\TradingData"
folder_path = os.path.abspath("Results") # "." means current directory

# # List of your specific files
# file_list = [
#     "mag7_ddqn.csv",
#     "mag7_dqn.csv",
#     "mag7_mda2c.csv",
#     "mag7_mdreinforce.csv",
#     "mag7_ddpg.csv"
# ]
# merge_cols = ['Time_Step', 'Agent_Value']
# preserve_cols = {
#     'Time_Step': 'Time_Step',
#     'Market_Value': 'Buy_n_Hold'
#                 }
# outfile_name = "mag7_results.csv"

# List of your specific files
file_list = [
    "mag7_ddqn_training.csv",
    "mag7_dqn_training.csv",
    "mag7_mda2c_training.csv",
    "mag7_mdreinforce_training.csv",
    "mag7_ddpg_training.csv"
]
merge_cols = ['Episode', 'Reward']
preserve_cols = {
    'Episode': 'Episode'
                }
outfile_name = "mag7_results_training.csv"



presevere_cols_lst = list(preserve_cols.keys())
merge_on = [item for item in merge_cols if item in presevere_cols_lst]
rename_col = [item for item in merge_cols if item not in presevere_cols_lst]

# Output filename
output_file = os.path.join(folder_path, outfile_name)
# ---------------------

# 1. Setup the baseline using the first file
first_file_path = os.path.join(folder_path, file_list[0])
base_df = pd.read_csv(first_file_path)

# Initialize combined DataFrame with Time_Step and Market_Value (baseline)
combined_df = base_df[presevere_cols_lst].copy()
combined_df = combined_df.rename(columns=preserve_cols)

# 2. Loop through files to merge Agent_Values
for filename in file_list:
    file_path = os.path.join(folder_path, filename)
    
    # Check if file exists to avoid errors
    if os.path.exists(file_path):
        current_df = pd.read_csv(file_path)
        
        # Extract algorithm name from filename (e.g., "mag7_ddqn.csv" -> "ddqn")
        # You can adjust the split logic if your naming convention changes
        algo_name = filename.split('_')[1].replace('.csv', '').upper()
        
        # Merge the Agent_Value column, renaming it to the algo name
        # We merge on 'Time_Step' to ensure data stays aligned even if rows are missing
        combined_df = pd.merge(
            combined_df, 
            current_df[merge_cols].rename(columns={rename_col[0]: algo_name}), 
            on=merge_on, 
            how='left'
        )
    else:
        print(f"Warning: File not found - {file_path}")

# 3. Save and View
print(f"Combined data saved to: {output_file}")
print(combined_df.head())

combined_df.to_csv(output_file, index=False)