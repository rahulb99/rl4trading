import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Path to the CSV file containing merged results
csv_file_path = "Results/mag7_results_training.csv" 

# Smoothing factor (Window Size). 
# This is now the 'N' in N-day Simple Moving Average.
smoothing_span = 500 

# Set this to 1000000 if plotting Portfolio Value, or 0 if plotting Rewards
baseline_value = 0 
# ---------------------

def plot_multi_algo(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        print("Please ensure you have a CSV with columns: Episode, Algo1, Algo2, ...")
        return

    # 1. Read Data
    df = pd.read_csv(file_path)
    
    # Try to set index to 'Episode' or 'Time_Step' if they exist
    if 'Episode' in df.columns:
        df = df.set_index('Episode')
    elif 'Time_Step' in df.columns:
        df = df.set_index('Time_Step')
    
    # 2. Setup Plot
    plt.figure(figsize=(14, 7))
    
    # Color palette for distinct lines
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    color_idx = 0

    # Dictionary to store the smoothed data for the CSV export
    smoothed_data = {}

    # 3. Iterate through every column (Algorithm)
    for algo_name in df.columns:
        # Get the data series
        data = df[algo_name]
        
        # --- CHANGED: Calculate SMA (Simple Moving Average) ---
        # min_periods=1 ensures we get a value even before the window is full
        sma = data.rolling(window=smoothing_span, min_periods=1).mean()
        
        # Store smoothed data for CSV saving later
        smoothed_data[algo_name] = sma
        
        # Pick color
        c = colors[color_idx % len(colors)]
        
        # A. Plot Raw Data (Very faint "shadow" to show variance)
        plt.plot(data.index, data, color=c, alpha=0.1, linewidth=1)
        
        # B. Plot Smoothed Trend (Thick, solid line)
        plt.plot(data.index, sma, color=c, linewidth=2.5, label=algo_name)
        
        color_idx += 1

    # 4. Add Baseline (Comparison Point)
    plt.axhline(y=baseline_value, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')

    # 5. Styling
    plt.title(f"Algorithm Comparison (SMA Window={smoothing_span})", fontsize=16)
    plt.xlabel("Episode / Time Step", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # --- SAVE IMAGE ---
    output_img = file_path.replace('.csv', '_comparison_sma.png')
    plt.savefig(output_img, dpi=300)
    print(f"Comparison chart saved to: {output_img}")
    
    # --- SAVE CSV ---
    # Create a DataFrame from the accumulated smoothed data
    smoothed_df = pd.DataFrame(smoothed_data, index=df.index)
    
    # Construct filename (e.g., mag7_results_training_smoothed_sma.csv)
    output_csv = file_path.replace('.csv', '_smoothed_sma.csv')
    
    # Save to CSV
    smoothed_df.to_csv(output_csv)
    print(f"Smoothed SMA data saved to: {output_csv}")

    plt.show()

if __name__ == "__main__":
    plot_multi_algo(csv_file_path)