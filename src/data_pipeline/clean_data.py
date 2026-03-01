import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.settings import RAW_DATA_DIR, TARGET_SYMBOLS, TIMEFRAME

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw OHLCV dataframe.
    """
    if df.empty:
        return df

    # 1. Drop duplicates based on timestamp index
    df = df[~df.index.duplicated(keep='first')]

    # 2. Forward fill missing values up to a limit
    # We don't want to forward fill too much, as stale data is bad for live trading
    df = df.ffill(limit=3) 

    # 3. Drop remaining NaNs
    df = df.dropna()

    # 4. Remove obvious outliers (e.g., volume == 0 usually means trading halt or bad data)
    # Be careful not to drop legitimate zero-volume candles in low liquidity assets,
    # but for Top US stocks, 0 volume is suspicious during market hours.
    # Note: Indices like ^NSEI often report 0 volume, so we should NOT drop them.
    # df = df[df['volume'] > 0]
    
    return df

def clean_all_raw_data():
    """Reads all raw data, cleans it, and saves it."""
    processed_dir = RAW_DATA_DIR.parent / 'processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    for symbol in TARGET_SYMBOLS:
        raw_file = RAW_DATA_DIR / f"{symbol}_{TIMEFRAME}_raw.csv"
        if raw_file.exists():
            print(f"Cleaning {symbol}...")
            df = pd.read_csv(raw_file, index_col=0, parse_dates=True)
            cleaned_df = clean_dataframe(df)
            
            processed_file = processed_dir / f"{symbol}_{TIMEFRAME}_cleaned.csv"
            cleaned_df.to_csv(processed_file)
            print(f"Saved cleaned data to {processed_file}. Kept {len(cleaned_df)}/{len(df)} rows.")
        else:
            print(f"Warning: Raw data file {raw_file} not found.")

if __name__ == "__main__":
    clean_all_raw_data()
