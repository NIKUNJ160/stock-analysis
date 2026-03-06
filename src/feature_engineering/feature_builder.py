import pandas as pd
import os

from src.feature_engineering.indicators import add_all_indicators
from src.feature_engineering.candlestick_features import add_candlestick_features, add_rolling_features
from src.utils.logger import get_logger

logger = get_logger("FeatureBuilder")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main pipeline to transform raw OHLCV data into a feature-rich dataset.
    This function is used BOTH for historical training and live inference.
    """
    if df.empty:
        return df
        
    df = df.copy()
    
    df = add_all_indicators(df)
    df = add_candlestick_features(df)
    df = add_rolling_features(df)
    df = df.dropna()
    
    return df


if __name__ == "__main__":
    from config.settings import TARGET_SYMBOLS, TIMEFRAME, RAW_DATA_DIR
    
    processed_dir = RAW_DATA_DIR.parent / 'processed'
    
    for symbol in TARGET_SYMBOLS:
        clean_file = processed_dir / f"{symbol}_{TIMEFRAME}_cleaned.csv"
        if clean_file.exists():
            logger.info(f"Building features for {symbol}...")
            df = pd.read_csv(clean_file, index_col=0, parse_dates=True)
            featured_df = build_features(df)
            
            features_dir = RAW_DATA_DIR.parent / 'features'
            os.makedirs(features_dir, exist_ok=True)
            
            out_file = features_dir / f"{symbol}_{TIMEFRAME}_features.csv"
            featured_df.to_csv(out_file)
            logger.info(f"Saved {len(featured_df)} feature rows to {out_file}")
