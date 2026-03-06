import pytest
import pandas as pd
import numpy as np

from src.feature_engineering.feature_builder import build_features
from src.feature_engineering.indicators import add_all_indicators
from src.feature_engineering.candlestick_features import add_candlestick_features, add_rolling_features


def _make_sample_ohlcv(n: int = 200) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        'open': close + np.random.randn(n) * 0.2,
        'high': close + abs(np.random.randn(n) * 0.5),
        'low': close - abs(np.random.randn(n) * 0.5),
        'close': close,
        'volume': np.random.randint(1000, 100000, size=n).astype(float),
    }, index=dates)
    return df


class TestBuildFeatures:
    def test_output_not_empty(self):
        df = _make_sample_ohlcv()
        result = build_features(df)
        assert not result.empty, "Features should not be empty"
    
    def test_no_nans(self):
        df = _make_sample_ohlcv()
        result = build_features(df)
        assert not result.isna().any().any(), "Features should have no NaNs after dropna"
    
    def test_has_expected_columns(self):
        df = _make_sample_ohlcv()
        result = build_features(df)
        expected = ['rsi_14', 'macd_diff', 'ema_9', 'ema_21', 'ema_50', 
                    'atr_14', 'body_size', 'candle_direction', 'rolling_vol_5']
        for col in expected:
            assert col in result.columns, f"Missing feature column: {col}"
    
    def test_empty_input(self):
        df = pd.DataFrame()
        result = build_features(df)
        assert result.empty
    
    def test_feature_count(self):
        df = _make_sample_ohlcv()
        result = build_features(df)
        # Should have more columns than raw OHLCV (5)
        assert len(result.columns) > 15, f"Expected >15 features, got {len(result.columns)}"


class TestIndicators:
    def test_add_all_indicators(self):
        df = _make_sample_ohlcv()
        result = add_all_indicators(df)
        assert 'rsi_14' in result.columns
        assert 'macd_diff' in result.columns
        assert 'ema_50' in result.columns
        assert 'atr_14' in result.columns
    
    def test_rsi_range(self):
        df = _make_sample_ohlcv()
        result = add_all_indicators(df).dropna()
        rsi = result['rsi_14']
        assert rsi.min() >= 0, "RSI should be >= 0"
        assert rsi.max() <= 100, "RSI should be <= 100"


class TestCandlestickFeatures:
    def test_candlestick_shapes(self):
        df = _make_sample_ohlcv()
        result = add_candlestick_features(df)
        assert 'body_size' in result.columns
        assert 'upper_wick' in result.columns
        assert 'candle_direction' in result.columns
        assert (result['body_size'] >= 0).all(), "Body size should be non-negative"
    
    def test_rolling_features(self):
        df = _make_sample_ohlcv()
        result = add_rolling_features(df)
        assert 'rolling_vol_5' in result.columns
        assert 'volume_ratio_5' in result.columns
