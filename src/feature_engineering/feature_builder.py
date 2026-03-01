import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.feature_engineering.indicators import add_all_indicators
from src.feature_engineering.candlestick_features import add_candlestick_features, add_rolling_features

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main pipeline to transform raw OHLCV data into a feature-rich dataset.
    This function will be used BOTH for historical training and live inference.
    """
    if df.empty:
        return df
        
    df = df.copy()
    
    # 1. Add Technical Indicators
    df = add_all_indicators(df)
    
    # 2. Add Candlestick Shapes
    df = add_candlestick_features(df)
    
    # 3. Add Rolling Statistics
    df = add_rolling_features(df)
    
    # 4. Drop rows with NaNs created by rolling windows/indicators
    # For example, RSI with window 14 needs 14 rows, so the first 13 rows will be NaN.
    # In live trading, we'll keep a buffer of recent candles to avoid this.
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    # Test script
    from config.settings import TARGET_SYMBOLS, TIMEFRAME, RAW_DATA_DIR
    processed_dir = RAW_DATA_DIR.parent / 'processed'
    
    for symbol in TARGET_SYMBOLS:
        clean_file = processed_dir / f"{symbol}_{TIMEFRAME}_cleaned.csv"
        if clean_file.exists():
            print(f"Building features for {symbol}...")
            df = pd.read_csv(clean_file, index_col=0, parse_dates=True)
            featured_df = build_features(df)
            
            features_dir = RAW_DATA_DIR.parent / 'features'
            os.makedirs(features_dir, exist_ok=True)
            
            out_file = features_dir / f"{symbol}_{TIMEFRAME}_features.csv"
            featured_df.to_csv(out_file)
            print(f"Saved {len(featured_df)} feature rows to {out_file}")
    
