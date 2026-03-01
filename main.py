import asyncio
import pandas as pd
import sys
import os
import json

from services.market_data_service.mock_feed import mock_websocket_feed
from services.model_service.realtime_predictor import RealtimePredictor
from src.feature_engineering.feature_builder import build_features
from config.settings import TARGET_SYMBOLS

class FeatureService:
    def __init__(self, max_buffer: int = 200):
        # Keeps recent raw candles in memory to compute rolling features like EMA50
        self.raw_buffers = {sym: pd.DataFrame() for sym in TARGET_SYMBOLS}
        self.max_buffer = max_buffer
        
    async def run(self, market_data_queue: asyncio.Queue, feature_queue: asyncio.Queue):
        print("[FeatureService] Started listening for live candles...")
        while True:
            # 1. Get raw candle dictionary from websocket/mock feed
            candle_msg = await market_data_queue.get()
            symbol = candle_msg['symbol']
            
            # Format as DataFrame row
            row = pd.DataFrame([{
                'open': candle_msg['open'],
                'high': candle_msg['high'],
                'low': candle_msg['low'],
                'close': candle_msg['close'],
                'volume': candle_msg['volume']
            }], index=pd.to_datetime([candle_msg['timestamp']]))
            
            # 2. Append to buffer & truncate
            buffer = pd.concat([self.raw_buffers[symbol], row])
            if len(buffer) > self.max_buffer:
                buffer = buffer.iloc[-self.max_buffer:]
            self.raw_buffers[symbol] = buffer
            
            # 3. If we have enough data (e.g., at least 50 candles for 50 EMA)
            if len(buffer) >= 60:
                # 4. Build all features (takes the whole buffer to calculate rolling)
                features_df = build_features(buffer)
                
                # 5. Extract only the VERY LATEST row (the newest candle's features)
                latest_features = features_df.iloc[[-1]] 
                
                # 6. Pass the computed feature vector to the Model Service
                packet = {
                    "symbol": symbol,
                    "timestamp": candle_msg['timestamp'],
                    "features": latest_features
                }
                await feature_queue.put(packet)
            
            market_data_queue.task_done()

async def signal_consumer(signal_queue: asyncio.Queue):
    """
    Acts as the Risk Engine / Execution Engine for now.
    Reads signals and prints them.
    """
    print("[SignalEngine] Started listening for trading signals...")
    latest_signals = {}
    signals_file = os.path.join("data", "latest_signals.json")
    while True:
        signal = await signal_queue.get()
        print(f"\n🚀 SIGNAL GENERATED: {signal['symbol']} @ {signal['timestamp']}")
        print(f"   Model: {signal['model_prediction']}")
        print(f"   Confidence: {signal['confidence']*100:.2f}%")
        
        # Save for dashboard
        latest_signals[signal['symbol']] = {
            "timestamp": signal['timestamp'],
            "prediction": signal['model_prediction'],
            "confidence": f"{signal['confidence']*100:.2f}%"
        }
        with open(signals_file, "w") as f:
            json.dump(latest_signals, f)
            
        signal_queue.task_done()

async def main():
    print("Starting Live Quant Architecture Services...\n")
    
    # 1) Setup Queues (Message Bus)
    market_data_queue = asyncio.Queue()
    feature_queue = asyncio.Queue()
    signal_queue = asyncio.Queue()

    # 2) Initialize State/Models
    feature_service = FeatureService()
    predictor_service = RealtimePredictor()
    
    # 3) Define async tasks representing microservices
    mock_feed_tasks = []
    for symbol in TARGET_SYMBOLS:
        task = asyncio.create_task(
            mock_websocket_feed(market_data_queue, symbol, emit_delay=0.1)
        )
        mock_feed_tasks.append(task)
    
    feature_task = asyncio.create_task(
        feature_service.run(market_data_queue, feature_queue)
    )
    
    predictor_task = asyncio.create_task(
        predictor_service.prediction_loop(feature_queue, signal_queue)
    )
    
    signal_task = asyncio.create_task(
        signal_consumer(signal_queue)
    )
    
    # 4) Wait for all mock feeds to finish playing
    await asyncio.gather(*mock_feed_tasks)
    
    # 5) Wait for all items in the queues to be processed before shutting down
    await market_data_queue.join()
    await feature_queue.join()
    await signal_queue.join()
    
    # Clean up the infinite listener tasks
    feature_task.cancel()
    predictor_task.cancel()
    signal_task.cancel()
    
    print("\n[System] All historical candles processed. Shutting down gracefully.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown signal received.")
