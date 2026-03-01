import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Adds Relative Strength Index (RSI)."""
    indicator = RSIIndicator(close=df["close"], window=window)
    df[f"rsi_{window}"] = indicator.rsi()
    return df

def add_macd(df: pd.DataFrame, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9) -> pd.DataFrame:
    """Adds MACD and MACD Signal."""
    indicator = MACD(close=df["close"], window_slow=window_slow, window_fast=window_fast, window_sign=window_sign)
    df[f"macd_{window_fast}_{window_slow}"] = indicator.macd()
    df[f"macd_signal_{window_sign}"] = indicator.macd_signal()
    df[f"macd_diff"] = indicator.macd_diff() # Histogram
    return df

def add_bollinger_bands(df: pd.DataFrame, window: int = 20, window_dev: int = 2) -> pd.DataFrame:
    """Adds Bollinger Bands."""
    indicator = BollingerBands(close=df["close"], window=window, window_dev=window_dev)
    df[f"bb_high_{window}"] = indicator.bollinger_hband()
    df[f"bb_low_{window}"] = indicator.bollinger_lband()
    df[f"bb_mid_{window}"] = indicator.bollinger_mavg()
    df[f"bb_width_{window}"] = indicator.bollinger_hband() - indicator.bollinger_lband()
    return df

def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Adds Average True Range (ATR) for volatility measurement."""
    indicator = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=window)
    df[f"atr_{window}"] = indicator.average_true_range()
    return df
    
def add_ema(df: pd.DataFrame, window: int) -> pd.DataFrame:
    indicator = EMAIndicator(close=df["close"], window=window)
    df[f"ema_{window}"] = indicator.ema_indicator()
    return df
    
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates all defined technical indicators on the dataframe."""
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_ema(df, window=9)
    df = add_ema(df, window=21)
    df = add_ema(df, window=50)
    return df
