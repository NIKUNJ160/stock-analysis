import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger("SignalEngine.Generator")


class SignalStrength(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class TradingSignal:
    symbol: str
    timestamp: str
    strength: SignalStrength
    confidence: float
    model_prediction: int
    model_confidence: float
    rsi_value: float
    macd_signal: str
    trend_alignment: str
    reasons: list[str]
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "model_prediction": self.model_prediction,
            "model_confidence": self.model_confidence,
            "rsi_value": self.rsi_value,
            "macd_signal": self.macd_signal,
            "trend_alignment": self.trend_alignment,
            "reasons": self.reasons,
        }


class SignalGenerator:
    """
    Combines model prediction with technical indicators to produce
    multi-level trading signals: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL.
    
    Logic:
    - Model says BUY + RSI oversold + bullish trend = STRONG BUY
    - Model says BUY + neutral indicators = BUY
    - Model says SELL + RSI overbought + bearish trend = STRONG SELL
    - Conflicting signals = HOLD
    """
    
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    
    def generate_signal(self, model_output: dict, features: pd.DataFrame) -> TradingSignal:
        """
        Generate a trading signal from model prediction + latest features.
        
        Args:
            model_output: Dict with 'model_prediction', 'confidence', 'symbol', 'timestamp'
            features: Latest feature row from the feature engine
        """
        symbol = model_output['symbol']
        timestamp = model_output['timestamp']
        model_pred = model_output['model_prediction']
        model_conf = model_output['confidence']
        
        # Extract indicator values from latest features
        latest = features.iloc[-1] if len(features) > 0 else pd.Series()
        
        rsi = latest.get('rsi_14', 50)
        macd_diff = latest.get('macd_diff', 0)
        ema_9 = latest.get('ema_9', 0)
        ema_21 = latest.get('ema_21', 0)
        ema_50 = latest.get('ema_50', 0)
        close = latest.get('close', latest.get('ema_9', 0))
        
        reasons = []
        score = 0  # -100 (max sell) to +100 (max buy)
        
        # --- Factor 1: Model Prediction (weight: 40%) ---
        if model_pred == 1:  # BUY
            score += 40
            reasons.append(f"Model predicts BUY ({model_conf:.1%})")
        else:
            score -= 40
            reasons.append(f"Model predicts HOLD/SELL ({model_conf:.1%})")
        
        # --- Factor 2: RSI (weight: 20%) ---
        if rsi < self.RSI_OVERSOLD:
            score += 20
            reasons.append(f"RSI oversold ({rsi:.1f})")
            macd_str = "oversold"
        elif rsi > self.RSI_OVERBOUGHT:
            score -= 20
            reasons.append(f"RSI overbought ({rsi:.1f})")
            macd_str = "overbought"
        else:
            macd_str = "neutral"
            reasons.append(f"RSI neutral ({rsi:.1f})")
        
        # --- Factor 3: MACD Crossover (weight: 15%) ---
        if macd_diff > 0:
            score += 15
            macd_signal = "bullish_crossover"
            reasons.append("MACD bullish crossover")
        elif macd_diff < 0:
            score -= 15
            macd_signal = "bearish_crossover"
            reasons.append("MACD bearish crossover")
        else:
            macd_signal = "neutral"
        
        # --- Factor 4: Trend Alignment (weight: 25%) ---
        if ema_9 > ema_21 > ema_50:
            score += 25
            trend = "bullish"
            reasons.append("Strong bullish trend (EMA 9 > 21 > 50)")
        elif ema_9 < ema_21 < ema_50:
            score -= 25
            trend = "bearish"
            reasons.append("Strong bearish trend (EMA 9 < 21 < 50)")
        elif ema_9 > ema_21:
            score += 10
            trend = "mild_bullish"
            reasons.append("Mild bullish trend (EMA 9 > 21)")
        elif ema_9 < ema_21:
            score -= 10
            trend = "mild_bearish"
            reasons.append("Mild bearish trend (EMA 9 < 21)")
        else:
            trend = "sideways"
            reasons.append("Sideways market")
        
        # --- Determine Signal Strength ---
        if score >= 60:
            strength = SignalStrength.STRONG_BUY
        elif score >= 20:
            strength = SignalStrength.BUY
        elif score <= -60:
            strength = SignalStrength.STRONG_SELL
        elif score <= -20:
            strength = SignalStrength.SELL
        else:
            strength = SignalStrength.HOLD
        
        # Normalized confidence [0, 1]
        confidence = min(abs(score) / 100, 1.0)
        
        signal = TradingSignal(
            symbol=symbol,
            timestamp=timestamp,
            strength=strength,
            confidence=confidence,
            model_prediction=model_pred,
            model_confidence=model_conf,
            rsi_value=rsi,
            macd_signal=macd_signal,
            trend_alignment=trend,
            reasons=reasons,
        )
        
        logger.info(f"{symbol} → {strength.value} (conf={confidence:.1%}, score={score})")
        return signal
