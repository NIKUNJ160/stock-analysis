import asyncio
import json
import websockets
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable

from config.settings import TARGET_SYMBOLS, TIMEFRAME
from src.utils.logger import get_logger

logger = get_logger("WebSocketListener")


class WebSocketListener:
    """
    Live WebSocket listener that connects to a market data provider
    and streams real-time tick data into the pipeline.
    
    Supports multiple providers via URL configuration.
    Default uses a simulated local WebSocket for development.
    """
    
    def __init__(self, ws_url: str = "ws://localhost:8765", symbols: list[str] = None):
        self.ws_url = ws_url
        self.symbols = symbols or TARGET_SYMBOLS
        self.is_running = False
        self._connection = None
    
    async def connect(self, market_data_queue: asyncio.Queue):
        """Connect to WebSocket and stream data into the queue."""
        self.is_running = True
        retry_delay = 1
        max_retry = 30
        
        while self.is_running:
            try:
                logger.info(f"Connecting to WebSocket at {self.ws_url}...")
                async with websockets.connect(self.ws_url) as ws:
                    self._connection = ws
                    retry_delay = 1  # Reset on successful connection
                    logger.info("WebSocket connected successfully")
                    
                    # Subscribe to symbols
                    subscribe_msg = json.dumps({
                        "type": "subscribe",
                        "symbols": self.symbols
                    })
                    await ws.send(subscribe_msg)
                    logger.info(f"Subscribed to: {self.symbols}")
                    
                    # Listen for messages
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            if data.get("type") == "tick":
                                await market_data_queue.put(data)
                            elif data.get("type") == "new_candle":
                                await market_data_queue.put(data)
                            elif data.get("type") == "error":
                                logger.warning(f"Server error: {data.get('message')}")
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON received: {message[:100]}")
                            
            except websockets.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
            except ConnectionRefusedError:
                logger.warning(f"Connection refused. Retrying in {retry_delay}s...")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            
            if self.is_running:
                await asyncio.sleep(min(retry_delay, max_retry))
                retry_delay = min(retry_delay * 2, max_retry)
    
    async def disconnect(self):
        """Gracefully disconnect."""
        self.is_running = False
        if self._connection:
            await self._connection.close()
            logger.info("WebSocket disconnected")


class CandleAggregator:
    """
    Aggregates raw tick data into OHLCV candles at the specified timeframe.
    
    Receives individual ticks and produces complete candles when
    the timeframe interval elapses.
    """
    
    TIMEFRAME_SECONDS = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
    }
    
    def __init__(self, timeframe: str = TIMEFRAME):
        if timeframe not in self.TIMEFRAME_SECONDS:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        self.timeframe = timeframe
        self.interval_seconds = int(self.TIMEFRAME_SECONDS.get(timeframe, 300))
        self.current_candles: dict[str, dict] = {}  # symbol -> current building candle
        self._stop_event = asyncio.Event()
        
    def stop(self):
        self._stop_event.set()
        
    def _get_candle_start(self, timestamp: datetime) -> datetime:
        """Round down timestamp to the start of the current candle interval."""
        seconds = int(timestamp.timestamp())
        candle_start = seconds - (seconds % self.interval_seconds)
        return datetime.fromtimestamp(candle_start)
    
    async def process_ticks(self, tick_queue: asyncio.Queue, candle_queue: asyncio.Queue):
        """
        Continuously reads ticks and emits completed candles.
        """
        logger.info(f"Candle Aggregator started ({self.timeframe} candles)")
        
        while not self._stop_event.is_set():
            try:
                tick = await asyncio.wait_for(tick_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            symbol = tick.get("symbol")
            if not symbol or ("price" not in tick and "close" not in tick) or "timestamp" not in tick:
                logger.warning(f"Invalid tick data missing required fields: {tick}")
                tick_queue.task_done()
                continue
                
            price = float(tick.get("price", tick.get("close", 0)))
            volume = float(tick.get("volume", 0))
            ts = datetime.fromisoformat(tick.get("timestamp", datetime.now(tz=timezone.utc).isoformat()))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            
            candle_start = self._get_candle_start(ts)
            
            if symbol not in self.current_candles:
                # Start a new candle
                self.current_candles[symbol] = {
                    "symbol": symbol,
                    "timestamp": candle_start.isoformat(),
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": volume,
                    "candle_start": candle_start,
                    "type": "new_candle"
                }
            else:
                current = self.current_candles[symbol]
                current_start = current["candle_start"]
                
                if candle_start < current_start:
                    logger.debug(f"Late tick for {symbol} ignored")
                    tick_queue.task_done()
                    continue
                elif candle_start > current_start:
                    # New interval — emit the completed candle
                    completed = {k: v for k, v in current.items() if k != "candle_start"}
                    await candle_queue.put(completed)
                    logger.debug(f"Candle completed: {symbol} @ {current['timestamp']}")
                    
                    # Start new candle
                    self.current_candles[symbol] = {
                        "symbol": symbol,
                        "timestamp": candle_start.isoformat(),
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": volume,
                        "candle_start": candle_start,
                        "type": "new_candle"
                    }
                else:
                    # Update current candle
                    current["high"] = max(current["high"], price)
                    current["low"] = min(current["low"], price)
                    current["close"] = price
                    current["volume"] += volume
            
            tick_queue.task_done()
