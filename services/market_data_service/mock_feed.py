import asyncio
import pandas as pd
import json
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from config.settings import TARGET_SYMBOLS, TIMEFRAME, RAW_DATA_DIR

async def mock_websocket_feed(market_data_queue: asyncio.Queue, symbol: str, emit_delay: float = 0.5):
    """
    Simulates a live market data feed by playing back a historical CSV file.
    Emits candles one by one to the queue at a specified delay.
    """
    processed_dir = RAW_DATA_DIR.parent / 'processed'
    file_path = processed_dir / f"{symbol}_{TIMEFRAME}_cleaned.csv"
    
    if not file_path.exists():
        print(f"[MarketFeed] Error: Could not find {file_path}")
        return

    # Read the historical data
    print(f"[MarketFeed] Loading mock data for {symbol}...")
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    
    # Simulate feeding data point by point
    for timestamp, row in df.iterrows():
        # Formulate a JSON-like message as if it came from a websocket
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
        await asyncio.sleep(emit_delay) # Simulate time passing between candles

    print(f"[MarketFeed] Finished playing back mock data for {symbol}.")
