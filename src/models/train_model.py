import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.settings import TARGET_SYMBOLS, TIMEFRAME, RAW_DATA_DIR, TRAIN_TEST_SPLIT, PREDICT_HORIZON
from src.models.random_forest import train_rf, evaluate_model, save_model

def create_target_labels(df: pd.DataFrame, horizon: int = PREDICT_HORIZON) -> pd.DataFrame:
    """
    Creates target labels for Supervised Learning.
    Target: 1 if the price went up after `horizon` candles, 0 otherwise.
    """
    df = df.copy()
    # Current close vs Future close
    df['future_close'] = df['close'].shift(-horizon)
    
    # 1 if future price > current price + a tiny fee threshold
    df['target'] = np.where(df['future_close'] > df['close'] * 1.0005, 1, 0)
    
    # Drop rows at the end that don't have a future_close yet
    df = df.dropna(subset=['future_close'])
    return df

def run_training_pipeline():
    features_dir = RAW_DATA_DIR.parent / 'features'
    
    for symbol in TARGET_SYMBOLS:
        feature_file = features_dir / f"{symbol}_{TIMEFRAME}_features.csv"
        
        if feature_file.exists():
            print(f"\n========== Starting Training for {symbol} ==========")
            df = pd.read_csv(feature_file, index_col=0, parse_dates=True)
            
            # Create labels
            df = create_target_labels(df)
            
            # Separate Features (X) and Target (y)
            # Remove non-predictive columns to prevent lookahead bias
            drop_cols = ['future_close', 'target'] 
            X = df.drop(columns=drop_cols)
            y = df['target']
            
            # Chronological Train/Test Split (Time Series strictly needs this)
            split_idx = int(len(df) * TRAIN_TEST_SPLIT)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
            
            # Train and Evaluate
            model = train_rf(X_train, y_train, symbol)
            evaluate_model(model, X_test, y_test)
            save_model(model, symbol, TIMEFRAME)
            
        else:
            print(f"Skipping {symbol}: Feature file not found. Did you run feature_builder.py?")

if __name__ == "__main__":
    run_training_pipeline()
