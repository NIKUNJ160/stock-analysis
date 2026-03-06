import asyncio
import pandas as pd
from pathlib import Path

from config.settings import TARGET_SYMBOLS, TIMEFRAME, RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger("MarketFeed")


async def mock_websocket_feed(market_data_queue: asyncio.Queue, symbol: str, emit_delay: float = 0.5):
    """
    Simulates a live market data feed by playing back a historical CSV file.
    Emits candles one by one to the queue at a specified delay.
    """
    processed_dir = RAW_DATA_DIR.parent / 'processed'
    file_path = processed_dir / f"{symbol}_{TIMEFRAME}_cleaned.csv"
    
    if not file_path.exists():
        logger.error(f"Could not find {file_path}")
        return

    logger.info(f"Loading mock data for {symbol}...")
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    
    for timestamp, row in df.iterrows():
        msg = {
            "type": "new_candle",
            "symbol": symbol,
            "timestamp": timestamp.isoformat(),
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": float(row['volume'])
        }
        
        await market_data_queue.put(msg)
        await asyncio.sleep(emit_delay)

    logger.info(f"Finished playing back mock data for {symbol}.")
