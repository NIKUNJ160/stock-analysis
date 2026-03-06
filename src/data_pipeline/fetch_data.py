import yfinance as yf
import pandas as pd

from config.settings import TARGET_SYMBOLS, TIMEFRAME, HISTORY_DAYS, RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger("DataPipeline.Fetch")


def fetch_historical_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Downloads historical OHLCV data for a given symbol."""
    logger.info(f"Fetching {period} of {interval} data for {symbol}...")
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            logger.warning(f"No data returned for {symbol}.")
            return df
            
        df.columns = [col.lower() for col in df.columns]
        if getattr(df.index, 'tz', None) is not None:
            df.index = df.index.tz_convert(None)
        
        cols_to_keep = ['open', 'high', 'low', 'close', 'volume']
        df = df[[col for col in cols_to_keep if col in df.columns]]
        
        filename = RAW_DATA_DIR / f"{symbol}_{interval}_raw.csv"
        df.to_csv(filename)
        logger.info(f"Saved {len(df)} rows to {filename}")
        
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def fetch_all():
    """Fetches data for all target symbols."""
    period_str = f"{HISTORY_DAYS}d"
    for symbol in TARGET_SYMBOLS:
        fetch_historical_data(symbol, TIMEFRAME, period_str)


if __name__ == "__main__":
    fetch_all()
