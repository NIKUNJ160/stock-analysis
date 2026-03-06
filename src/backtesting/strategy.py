from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional

from src.signal_engine.signal_generator import SignalGenerator, TradingSignal, SignalStrength
from src.utils.logger import get_logger

logger = get_logger("Backtesting.Strategy")


class Strategy(ABC):
    """Abstract base class for backtesting strategies."""
    
    @abstractmethod
    def on_candle(self, features_row: pd.Series, model_output: dict) -> Optional[TradingSignal]:
        """Process a new candle and optionally return a signal."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        pass


class MLStrategy(Strategy):
    """
    ML-based strategy that uses the trained model predictions
    combined with the Signal Generator for trade decisions.
    """
    
    def __init__(self):
        self.signal_gen = SignalGenerator()
    
    def name(self) -> str:
        return "ML_RandomForest_SignalCombo"
    
    def on_candle(self, features_df: pd.DataFrame, model_output: dict) -> Optional[TradingSignal]:
        """
        Generate signal from model output + features.
        """
        signal = self.signal_gen.generate_signal(model_output, features_df)
        
        # Only return actionable signals
        if signal.strength in [SignalStrength.STRONG_BUY, SignalStrength.BUY, 
                               SignalStrength.SELL, SignalStrength.STRONG_SELL]:
            return signal
        
        return None


class MomentumStrategy(Strategy):
    """
    Pure technical momentum strategy (no ML).
    Buys when RSI < 30 + bullish MACD, sells when RSI > 70 + bearish MACD.
    """
    
    def name(self) -> str:
        return "Momentum_RSI_MACD"
    
    def on_candle(self, features_df: pd.DataFrame, model_output: dict) -> Optional[TradingSignal]:
        latest = features_df.iloc[-1] if len(features_df) > 0 else pd.Series()
        
        rsi = latest.get('rsi_14', 50)
        macd_diff = latest.get('macd_diff', 0)
        symbol = model_output.get('symbol', 'UNKNOWN')
        timestamp = model_output.get('timestamp', '')
        
        if rsi < 30 and macd_diff > 0:
            return TradingSignal(
                symbol=symbol, timestamp=timestamp,
                strength=SignalStrength.STRONG_BUY,
                confidence=0.7, model_prediction=1, model_confidence=0.7,
                rsi_value=rsi, macd_signal="bullish",
                trend_alignment="oversold_bounce",
                reasons=["RSI oversold", "MACD bullish crossover"],
            )
        elif rsi > 70 and macd_diff < 0:
            return TradingSignal(
                symbol=symbol, timestamp=timestamp,
                strength=SignalStrength.STRONG_SELL,
                confidence=0.7, model_prediction=0, model_confidence=0.7,
                rsi_value=rsi, macd_signal="bearish",
                trend_alignment="overbought_reversal",
                reasons=["RSI overbought", "MACD bearish crossover"],
            )
        
        return None
