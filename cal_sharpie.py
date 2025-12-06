import pandas as pd
import numpy as np
import os

# --- Configuration ---
input_file = "Results/mag7_combined_results.csv"  
output_file = "Results/mag7_performance_metrics.csv"

# The baseline cash you started with
INITIAL_CASH = 1000000 

# Risk-Free Rate (Annualized 4.5% for Sharpe)
RISK_FREE_RATE_ANNUAL = 3.75/100

# Trading Days (252 for stocks, 365 for crypto)
TRADING_DAYS = 252 
# ---------------------

def calculate_metrics(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # 1. Load Data
    df = pd.read_csv(file_path)
    
    # Set index if needed
    if 'Time_Step' in df.columns:
        df = df.set_index('Time_Step')

    # 2. Calculate Daily Returns (for Sharpe/Volatility)
    returns_df = df.pct_change().dropna()

    metrics_data = []

    print(f"{'Model':<15} | {'Total Return Pct':<15} | {'Sharpe Ratio':<12} | {'Ann. Return':<12} | {'Ann. Volatility':<15}")
    print("-" * 80)

    # 3. Iterate through each model
    for model in df.columns:
        # Get raw prices and daily returns
        prices = df[model]
        r = returns_df[model]
        
        # --- A. Total Return % (Cumulative) ---
        # Formula: (Final Value - Initial Cash) / Initial Cash
        final_value = prices.iloc[-1]
        total_return = (final_value - INITIAL_CASH) / INITIAL_CASH

        # --- B. Annualized Return ---
        # Mean daily return * 252 (Standard industry approximation)
        avg_return = r.mean() * TRADING_DAYS
        
        # --- C. Annualized Volatility ---
        volatility = r.std() * np.sqrt(TRADING_DAYS)
        
        # --- D. Sharpe Ratio ---
        if volatility == 0:
            sharpe = 0
        else:
            sharpe = (avg_return - RISK_FREE_RATE_ANNUAL) / volatility
        
        # Store Data
        metrics_data.append({
            "Model": model,
            "Total_Return_Pct": total_return,
            "Sharpe_Ratio": sharpe,
            "Annualized_Return": avg_return,
            "Annualized_Volatility": volatility,
            "Final_Value": final_value
        })

        # Print
        print(f"{model:<15} | {total_return:>15.2%} | {sharpe:>12.4f} | {avg_return:>12.2%} | {volatility:>15.2%}")

    # 4. Save to CSV
    metrics_df = pd.DataFrame(metrics_data)
    
    # Sort by Total Return (Highest Profit First)
    metrics_df = metrics_df.sort_values(by="Total_Return_Pct", ascending=False)
    
    metrics_df.to_csv(output_file, index=False)
    print(f"\nSuccess! Metrics saved to: {output_file}")

if __name__ == "__main__":
    calculate_metrics(input_file)