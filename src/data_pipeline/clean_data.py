import pandas as pd
import numpy as np
import os

from config.settings import RAW_DATA_DIR, TARGET_SYMBOLS, TIMEFRAME
from src.utils.logger import get_logger

logger = get_logger("DataPipeline.Clean")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the raw OHLCV dataframe."""
    if df.empty:
        return df

    df = df[~df.index.duplicated(keep='first')]
    df = df.ffill(limit=3)
    df = df.dropna()
    
    return df


def clean_all_raw_data():
    """Reads all raw data, cleans it, and saves it."""
    processed_dir = RAW_DATA_DIR.parent / 'processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    for symbol in TARGET_SYMBOLS:
        raw_file = RAW_DATA_DIR / f"{symbol}_{TIMEFRAME}_raw.csv"
        if raw_file.exists():
            logger.info(f"Cleaning {symbol}...")
            df = pd.read_csv(raw_file, index_col=0, parse_dates=True)
            cleaned_df = clean_dataframe(df)
            
            processed_file = processed_dir / f"{symbol}_{TIMEFRAME}_cleaned.csv"
            cleaned_df.to_csv(processed_file)
            logger.info(f"Saved cleaned data to {processed_file}. Kept {len(cleaned_df)}/{len(df)} rows.")
        else:
            logger.warning(f"Raw data file {raw_file} not found.")


if __name__ == "__main__":
    clean_all_raw_data()
