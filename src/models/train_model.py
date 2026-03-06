import pandas as pd
import numpy as np
import os

from config.settings import TARGET_SYMBOLS, TIMEFRAME, RAW_DATA_DIR, TRAIN_TEST_SPLIT, PREDICT_HORIZON
from src.models.random_forest import train_rf, evaluate_model, save_model
from src.utils.logger import get_logger

logger = get_logger("Models.Training")

# Raw OHLCV columns that must be dropped to prevent lookahead bias
RAW_OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']


def create_target_labels(df: pd.DataFrame, horizon: int = PREDICT_HORIZON) -> pd.DataFrame:
    """
    Creates target labels for Supervised Learning.
    Target: 1 if the price went up after `horizon` candles, 0 otherwise.
    """
    df = df.copy()
    df['future_close'] = df['close'].shift(-horizon)
    df['target'] = np.where(df['future_close'] > df['close'] * 1.0005, 1, 0)
    df = df.dropna(subset=['future_close'])
    return df


def run_training_pipeline():
    features_dir = RAW_DATA_DIR.parent / 'features'
    
    for symbol in TARGET_SYMBOLS:
        feature_file = features_dir / f"{symbol}_{TIMEFRAME}_features.csv"
        
        if feature_file.exists():
            logger.info(f"========== Starting Training for {symbol} ==========")
            df = pd.read_csv(feature_file, index_col=0, parse_dates=True)
            
            df = create_target_labels(df)
            
            # Drop non-predictive columns AND raw OHLCV to prevent lookahead bias
            drop_cols = ['future_close', 'target'] + RAW_OHLCV_COLS
            drop_cols = [c for c in drop_cols if c in df.columns]
            X = df.drop(columns=drop_cols)
            y = df['target']
            
            # Chronological Train/Test Split
            split_idx = int(len(df) * TRAIN_TEST_SPLIT)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
            logger.info(f"Feature columns ({len(X.columns)}): {list(X.columns)}")
            
            model = train_rf(X_train, y_train, symbol)
            evaluate_model(model, X_test, y_test)
            save_model(model, symbol, TIMEFRAME)
            
        else:
            logger.warning(f"Skipping {symbol}: Feature file not found. Run feature_builder.py first.")


if __name__ == "__main__":
    run_training_pipeline()
