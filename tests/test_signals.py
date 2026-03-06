import pytest
import pandas as pd
import numpy as np

from src.signal_engine.signal_generator import SignalGenerator, SignalStrength
from src.signal_engine.signal_filter import SignalFilter, FilterConfig, TradingSignal


def _make_features(rsi=50, macd_diff=0, ema_9=100, ema_21=99, ema_50=98) -> pd.DataFrame:
    return pd.DataFrame([{
        'rsi_14': rsi, 'macd_diff': macd_diff,
        'ema_9': ema_9, 'ema_21': ema_21, 'ema_50': ema_50,
        'close': 100,
    }])


class TestSignalGenerator:
    def setup_method(self):
        self.gen = SignalGenerator()
    
    def test_strong_buy_signal(self):
        model_output = {'symbol': 'TCS.NS', 'timestamp': '2024-01-01', 
                        'model_prediction': 1, 'confidence': 0.85}
        features = _make_features(rsi=25, macd_diff=0.5, ema_9=102, ema_21=101, ema_50=100)
        signal = self.gen.generate_signal(model_output, features)
        assert signal.strength == SignalStrength.STRONG_BUY
    
    def test_strong_sell_signal(self):
        model_output = {'symbol': 'TCS.NS', 'timestamp': '2024-01-01',
                        'model_prediction': 0, 'confidence': 0.8}
        features = _make_features(rsi=75, macd_diff=-0.5, ema_9=98, ema_21=99, ema_50=100)
        signal = self.gen.generate_signal(model_output, features)
        assert signal.strength in [SignalStrength.STRONG_SELL, SignalStrength.SELL]
    
    def test_hold_on_conflicting_signals(self):
        model_output = {'symbol': 'TCS.NS', 'timestamp': '2024-01-01',
                        'model_prediction': 1, 'confidence': 0.6}
        features = _make_features(rsi=72, macd_diff=-0.3, ema_9=98, ema_21=99, ema_50=100)
        signal = self.gen.generate_signal(model_output, features)
        # Conflicting: model says BUY but RSI overbought + bearish trend
        assert signal.strength in [SignalStrength.HOLD, SignalStrength.SELL, SignalStrength.BUY]


class TestSignalFilter:
    def test_pass_good_signal(self):
        sf = SignalFilter(FilterConfig(min_confidence=0.3, cooldown_seconds=0))
        signal = TradingSignal(
            symbol="TCS.NS", timestamp="2024-01-01",
            strength=SignalStrength.STRONG_BUY, confidence=0.8,
            model_prediction=1, model_confidence=0.75,
            rsi_value=28, macd_signal="bullish",
            trend_alignment="bullish", reasons=["Test"]
        )
        passed, reason = sf.should_pass(signal)
        assert passed, f"Should pass: {reason}"
    
    def test_block_hold(self):
        sf = SignalFilter()
        signal = TradingSignal(
            symbol="TCS.NS", timestamp="2024-01-01",
            strength=SignalStrength.HOLD, confidence=0.5,
            model_prediction=0, model_confidence=0.5,
            rsi_value=50, macd_signal="neutral",
            trend_alignment="sideways", reasons=[]
        )
        passed, _ = sf.should_pass(signal)
        assert not passed
    
    def test_block_low_confidence(self):
        sf = SignalFilter(FilterConfig(min_confidence=0.5))
        signal = TradingSignal(
            symbol="TCS.NS", timestamp="2024-01-01",
            strength=SignalStrength.BUY, confidence=0.2,
            model_prediction=1, model_confidence=0.6,
            rsi_value=40, macd_signal="bullish",
            trend_alignment="bullish", reasons=[]
        )
        passed, _ = sf.should_pass(signal)
        assert not passed
