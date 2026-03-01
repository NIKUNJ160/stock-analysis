import yfinance as yf
import pandas as pd
import sys
import os

# Add the root directory to sys.path to resolve imports properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.settings import TARGET_SYMBOLS, TIMEFRAME, HISTORY_DAYS, RAW_DATA_DIR

def fetch_historical_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """
    Downloads historical OHLCV data for a given symbol.
    """
    print(f"Fetching {period} of {interval} data for {symbol}...")
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            print(f"Warning: No data returned for {symbol}.")
            return df
            
        # Clean up column names (lowercase) and drop timezone awareness for simplicity
        df.columns = [col.lower() for col in df.columns]
        df.index = df.index.tz_localize(None)
        
        # Keep only standard OHLCV columns
        cols_to_keep = ['open', 'high', 'low', 'close', 'volume']
        df = df[[col for col in cols_to_keep if col in df.columns]]
        
        # Save to raw directory
        filename = RAW_DATA_DIR / f"{symbol}_{interval}_raw.csv"
        df.to_csv(filename)
        print(f"Saved {len(df)} rows to {filename}")
        
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def fetch_all():
    """Fetches data for all target symbols."""
    period_str = f"{HISTORY_DAYS}d"
    
    for symbol in TARGET_SYMBOLS:
        fetch_historical_data(symbol, TIMEFRAME, period_str)

if __name__ == "__main__":
    fetch_all()
