import os
import yfinance as yf
import pandas as pd

# 1. Define Tickers and Directory
TICKERS = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']
DATA_DIR = "data2"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

print(f"Downloading data for: {TICKERS}")

# 2. Download and Save
for ticker in TICKERS:
    print(f"Fetching {ticker}...", end=" ")
    
    # Download full history
    df = yf.download(ticker, period="max", interval="1d", progress=False)
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)
    
    # 3. Filter and Reorder columns specifically
    target_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    # Select only these columns to ensure order
    df = df[target_cols]
    
    # 4. Save to CSV with Manual Headers
    save_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    
    with open(save_path, 'w', newline='') as f:
        # Row 1: Date,Close,High,Low,Open,Volume
        # We manually construct this string
        header_1 = "Date," + ",".join(target_cols)
        f.write(header_1 + "\n")
        
        # Row 2: ,AAPL,AAPL,AAPL,AAPL,AAPL
        # The leading comma skips the Index column, aligning tickers with data columns
        header_2 = "," + ",".join([ticker] * len(target_cols))
        f.write(header_2 + "\n")
        
        # Write the dataframe data
        # header=False: Don't write pandas headers (we did it manually)
        # index=True: DO write the Date index (default)
        df.to_csv(f, header=False)
        
    print(f"Saved {len(df)} rows to {save_path}")

print("\nDone! Run your training script again.")