import asyncio
import pandas as pd
import json
import os

from services.market_data_service.mock_feed import mock_websocket_feed
from services.model_service.realtime_predictor import RealtimePredictor
from src.feature_engineering.feature_builder import build_features
from src.signal_engine.signal_generator import SignalGenerator, SignalStrength
from src.signal_engine.signal_filter import SignalFilter, FilterConfig
from src.risk_management.risk_manager import RiskManager, RiskConfig
from src.risk_management.stoploss import StopLossEngine
from src.risk_management.position_sizing import PositionSizer
from services.execution_service.broker_api import PaperTradingBroker
from services.execution_service.order_manager import OrderManager
from infrastructure.redis_cache import RedisCache
from src.utils.logger import get_logger
from src.utils.helpers import safe_write_json
from config.settings import (
    TARGET_SYMBOLS, MAX_CANDLE_BUFFER, MOCK_EMIT_DELAY,
    MAX_CAPITAL, MAX_RISK_PER_TRADE, MAX_OPEN_POSITIONS,
    MAX_EXPOSURE_PER_SYMBOL, MAX_DAILY_LOSS,
    SIGNAL_MIN_CONFIDENCE, SIGNAL_MIN_MODEL_CONFIDENCE,
    SIGNAL_COOLDOWN_SECONDS, SIGNAL_MAX_VOLATILITY_RATIO,
    SL_ATR_MULTIPLIER,
)

logger = get_logger("Main")


class FeatureService:
    def __init__(self, max_buffer: int = MAX_CANDLE_BUFFER):
        self.raw_buffers = {sym: pd.DataFrame() for sym in TARGET_SYMBOLS}
        self.max_buffer = max_buffer
        
    async def run(self, market_data_queue: asyncio.Queue, feature_queue: asyncio.Queue):
        logger.info("FeatureService started listening for live candles...")
        while True:
            candle_msg = await market_data_queue.get()
            symbol = candle_msg['symbol']
            
            row = pd.DataFrame([{
                'open': candle_msg['open'],
                'high': candle_msg['high'],
                'low': candle_msg['low'],
                'close': candle_msg['close'],
                'volume': candle_msg['volume']
            }], index=pd.to_datetime([candle_msg['timestamp']]))
            
            buffer = pd.concat([self.raw_buffers[symbol], row])
            if len(buffer) > self.max_buffer:
                buffer = buffer.iloc[-self.max_buffer:]
            self.raw_buffers[symbol] = buffer
            
            if len(buffer) >= 60:
                try:
                    features_df = build_features(buffer)
                    latest_features = features_df.iloc[[-1]]
                    
                    packet = {
                        "symbol": symbol,
                        "timestamp": candle_msg['timestamp'],
                        "features": latest_features,
                        "close_price": candle_msg['close'],
                    }
                    await feature_queue.put(packet)
                except Exception as e:
                    logger.error(f"Feature computation error for {symbol}: {e}")
            
            market_data_queue.task_done()


class SignalService:
    """
    Enhanced signal consumer with Signal Generator + Filter + Risk Engine + Order Execution.
    """
    def __init__(self):
        self.signal_gen = SignalGenerator()
        self.signal_filter = SignalFilter(FilterConfig(
            min_confidence=SIGNAL_MIN_CONFIDENCE,
            min_model_confidence=SIGNAL_MIN_MODEL_CONFIDENCE,
            cooldown_seconds=SIGNAL_COOLDOWN_SECONDS,
            max_volatility_ratio=SIGNAL_MAX_VOLATILITY_RATIO,
        ))
        self.risk_manager = RiskManager(RiskConfig(
            max_capital=MAX_CAPITAL,
            max_risk_per_trade=MAX_RISK_PER_TRADE,
            max_open_positions=MAX_OPEN_POSITIONS,
            max_exposure_per_symbol=MAX_EXPOSURE_PER_SYMBOL,
            max_daily_loss=MAX_DAILY_LOSS,
        ))
        self.sl_engine = StopLossEngine()
        self.sizer = PositionSizer()
        self.broker = PaperTradingBroker(initial_capital=MAX_CAPITAL)
        self.order_manager = OrderManager(self.broker, self.risk_manager, MAX_CAPITAL)
        self.cache = RedisCache()
        os.makedirs("data", exist_ok=True)
        self.signals_file = os.path.join("data", "latest_signals.json")
        self.portfolio_file = os.path.join("data", "portfolio.json")
    
    async def run(self, signal_queue: asyncio.Queue):
        logger.info("SignalService started listening for model outputs...")
        while True:
            model_output = await signal_queue.get()
            symbol = model_output['symbol']
            timestamp = model_output['timestamp']
            
            try:
                # 1. Generate enriched signal
                features = model_output.get('features', pd.DataFrame())
                signal = self.signal_gen.generate_signal(model_output, features)
                
                # 2. Filter signal
                features_row = None
                if not features.empty:
                    features_row = features.iloc[-1].to_dict()
                
                passed, reason = self.signal_filter.should_pass(signal, features_row)
                close_price = model_output.get('close_price', 0)
                
                if close_price > 0:
                    await self.order_manager.check_exits({symbol: close_price})
                
                signal_data = {
                    "timestamp": timestamp,
                    "strength": signal.strength.value,
                    "confidence": f"{signal.confidence*100:.2f}%",
                    "model_confidence": f"{signal.model_confidence*100:.2f}%",
                    "rsi": f"{signal.rsi_value:.1f}",
                    "trend": signal.trend_alignment,
                    "reasons": signal.reasons,
                    "filter_passed": passed,
                    "filter_reason": reason,
                }
                
                # 3. Risk check + execution (if signal passed)
                if passed and signal.strength in [SignalStrength.STRONG_BUY, SignalStrength.BUY,
                                                   SignalStrength.SELL, SignalStrength.STRONG_SELL]:
                    atr = 0
                    if features_row:
                        atr = features_row.get('atr_14', close_price * 0.01)
                    
                    placed_order = await self.order_manager.process_signal(signal, close_price, atr)
                    if placed_order:
                        signal_data["risk_approved"] = True
                        signal_data["risk_reason"] = "Order executed"
                        signal_data["position_size"] = placed_order.quantity
                    else:
                        signal_data["risk_approved"] = False
                        signal_data["risk_reason"] = "Rejected by Risk Manager or 0 size"
                
                # 4. Update cache and save
                self.latest_signals[symbol] = signal_data
                self.cache.set_latest_signal(symbol, signal_data)
                await asyncio.to_thread(safe_write_json, self.signals_file, self.latest_signals)
                
                # Save portfolio state
                portfolio = self.risk_manager.get_portfolio_summary()
                await asyncio.to_thread(safe_write_json, self.portfolio_file, portfolio)
                
                # Log
                emoji = "🟢" if "BUY" in signal.strength.value else "🔴" if "SELL" in signal.strength.value else "⚪"
                logger.info(f"{emoji} {symbol} → {signal.strength.value} | "
                           f"Conf={signal.confidence:.0%} | Filter={'✅' if passed else '❌'} | {reason}")
                
            except Exception as e:
                logger.error(f"Signal processing error for {symbol}: {e}")
            
            signal_queue.task_done()


async def main():
    logger.info("=" * 60)
    logger.info("Starting Live Quant Architecture Services")
    logger.info("=" * 60)
    
    # 1) Setup Queues  
    market_data_queue = asyncio.Queue()
    feature_queue = asyncio.Queue()
    signal_queue = asyncio.Queue()

    # 2) Initialize Services
    feature_service = FeatureService()
    predictor_service = RealtimePredictor()
    signal_service = SignalService()
    
    # 3) Launch async tasks
    mock_feed_tasks = []
    for symbol in TARGET_SYMBOLS:
        task = asyncio.create_task(
            mock_websocket_feed(market_data_queue, symbol, emit_delay=MOCK_EMIT_DELAY)
        )
        mock_feed_tasks.append(task)
    
    feature_task = asyncio.create_task(
        feature_service.run(market_data_queue, feature_queue)
    )
    
    predictor_task = asyncio.create_task(
        predictor_service.prediction_loop(feature_queue, signal_queue)
    )
    
    signal_task = asyncio.create_task(
        signal_service.run(signal_queue)
    )
    
    # 4) Wait for all mock feeds to finish
    await asyncio.gather(*mock_feed_tasks)
    
    # 5) Drain queues
    await market_data_queue.join()
    await feature_queue.join()
    await signal_queue.join()
    
    # Cleanup
    feature_task.cancel()
    predictor_task.cancel()
    signal_task.cancel()
    
    logger.info("All historical candles processed. Shutting down gracefully.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
