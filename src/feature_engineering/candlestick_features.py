import pandas as pd
import numpy as np

def add_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes custom features based on candlestick geometry.
    This helps the model 'see' the shape of the candle, not just the raw numbers.
    """
    # Price difference metrics
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['total_range'] = df['high'] - df['low']
    
    # Ratios (normalized shapes)
    df['body_ratio'] = df['body_size'] / (df['total_range'] + 1e-8)
    df['upper_wick_ratio'] = df['upper_wick'] / (df['total_range'] + 1e-8)
    df['lower_wick_ratio'] = df['lower_wick'] / (df['total_range'] + 1e-8)

    # Direction (1 for Green/Up, -1 for Red/Down)
    df['candle_direction'] = np.where(df['close'] > df['open'], 1, -1)
    df.loc[df['close'] == df['open'], 'candle_direction'] = 0
    
    # Gap from previous candle's close
    df['gap'] = df['open'] - df['close'].shift(1)
    
    return df
    
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds statistical features over recent rolling windows.
    """
    df['rolling_vol_5'] = df['close'].rolling(5).std()
    df['rolling_vol_10'] = df['close'].rolling(10).std()
    
    # Normalize volume against recent average
    df['volume_ratio_5'] = df['volume'] / (df['volume'].rolling(5).mean() + 1e-8)
    
    return df
