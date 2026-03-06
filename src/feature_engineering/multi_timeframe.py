import pandas as pd
import numpy as np
from typing import Optional

from src.feature_engineering.feature_builder import build_features
from src.utils.logger import get_logger

logger = get_logger("FeatureEngine.MultiTimeframe")


class MultiTimeframeAnalyzer:
    """
    Multi-timeframe analysis engine.
    
    Resamples 5-minute data into higher timeframes (15m, 30m, 1h)
    and computes features at each level. The trend agreement across
    timeframes is a powerful signal.
    
    Usage:
        analyzer = MultiTimeframeAnalyzer()
        mtf_features = analyzer.compute(df_5m)
    """
    
    RESAMPLE_MAP = {
        '15m': '15min',
        '30m': '30min',
        '1h': '1h',
    }
    
    OHLCV_RESAMPLE = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    
    def resample_ohlcv(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        Resample OHLCV data from a lower timeframe to a higher one.
        
        Example: 5m → 15m, 5m → 1h
        """
        rule = self.RESAMPLE_MAP.get(target_timeframe)
        if rule is None:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")
        
        resampled = df.resample(rule).agg(self.OHLCV_RESAMPLE).dropna()
        logger.debug(f"Resampled {len(df)} bars → {len(resampled)} bars ({target_timeframe})")
        return resampled
    
    def compute(self, df_base: pd.DataFrame, timeframes: list[str] = None) -> pd.DataFrame:
        """
        Compute multi-timeframe features and merge them back to base timeframe.
        
        Args:
            df_base: Base OHLCV DataFrame (e.g., 5m candles)
            timeframes: List of higher timeframes to analyze
        
        Returns:
            DataFrame with original + multi-timeframe features
        """
        timeframes = timeframes or ['15m', '1h']
        result = df_base.copy()
        
        for tf in timeframes:
            try:
                # Resample to higher timeframe
                df_higher = self.resample_ohlcv(df_base, tf)
                
                if len(df_higher) < 60:
                    logger.warning(f"Not enough data for {tf} analysis ({len(df_higher)} bars)")
                    continue
                
                # Build features at higher timeframe
                features_higher = build_features(df_higher)
                
                # Select key features to transfer down
                key_features = []
                for col in features_higher.columns:
                    if col in ['rsi_14', 'macd_diff', 'ema_9', 'ema_21', 'ema_50',
                               'atr_14', 'bb_width_20', 'candle_direction']:
                        key_features.append(col)
                
                # Rename with timeframe prefix
                renamed = features_higher[key_features].copy()
                renamed.columns = [f"{tf}_{col}" for col in renamed.columns]
                
                # Forward-fill merge to base timeframe
                # Each higher-timeframe value applies to all base bars within that period
                result = result.join(renamed, how='left')
                result = result.ffill()
                
                logger.info(f"Added {len(key_features)} features from {tf} timeframe")
                
            except Exception as e:
                logger.error(f"Error computing {tf} features: {e}")
        
        # Trend agreement score: how many timeframes agree on direction
        trend_cols = [c for c in result.columns if 'candle_direction' in c]
        if trend_cols:
            result['trend_agreement'] = result[trend_cols].mean(axis=1)
            result['trend_strength'] = result[trend_cols].apply(
                lambda row: 1 if all(v > 0 for v in row) else (-1 if all(v < 0 for v in row) else 0),
                axis=1
            )
        
        result = result.dropna()
        logger.info(f"Multi-timeframe features: {len(result)} rows, {len(result.columns)} columns")
        
        return result
    
    def get_trend_summary(self, df_base: pd.DataFrame) -> dict:
        """
        Get a quick trend summary across timeframes.
        
        Returns dict like:
        {
            '5m': 'bullish',
            '15m': 'bearish', 
            '1h': 'bullish',
            'agreement': 'mixed'
        }
        """
        summary = {}
        
        # Base timeframe trend
        if len(df_base) >= 50:
            features = build_features(df_base)
            if len(features) > 0:
                latest = features.iloc[-1]
                ema_9 = latest.get('ema_9', 0)
                ema_21 = latest.get('ema_21', 0)
                summary['5m'] = 'bullish' if ema_9 > ema_21 else 'bearish'
        
        # Higher timeframes
        for tf in ['15m', '1h']:
            try:
                df_higher = self.resample_ohlcv(df_base, tf)
                if len(df_higher) >= 50:
                    features = build_features(df_higher)
                    if len(features) > 0:
                        latest = features.iloc[-1]
                        ema_9 = latest.get('ema_9', 0)
                        ema_21 = latest.get('ema_21', 0)
                        summary[tf] = 'bullish' if ema_9 > ema_21 else 'bearish'
            except Exception:
                pass
        
        # Agreement
        trends = list(summary.values())
        if trends and all(t == trends[0] for t in trends):
            summary['agreement'] = trends[0]
        else:
            summary['agreement'] = 'mixed'
        
        return summary
