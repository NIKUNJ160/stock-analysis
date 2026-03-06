import numpy as np
import pandas as pd
import os
import joblib

from config.settings import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger("Models.LSTM")

# Lazy imports to avoid TensorFlow startup time when not needed
_tf = None
_keras = None


def _import_tf():
    global _tf, _keras
    if _tf is None:
        import tensorflow as tf
        _tf = tf
        _keras = tf.keras
        # Suppress TF warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')
    return _tf, _keras


def prepare_lstm_data(features_df: pd.DataFrame, lookback: int = 50, 
                       target_col: str = 'target') -> tuple:
    """
    Prepare data for LSTM: create sequences of `lookback` timesteps.
    
    Args:
        features_df: DataFrame with features and target
        lookback: Number of past candles to use as input sequence
        target_col: Column name for the target
    
    Returns:
        (X, y) where X has shape (samples, lookback, features)
    """
    if target_col not in features_df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    feature_cols = [c for c in features_df.columns 
                    if c not in [target_col, 'future_close', 'open', 'high', 'low', 'close', 'volume']]
    
    data = features_df[feature_cols].values
    targets = features_df[target_col].values
    
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(targets[i])
    
    return np.array(X), np.array(y)


def build_lstm_model(input_shape: tuple, units: int = 64) -> 'keras.Model':
    """Build an LSTM model for binary classification."""
    tf, keras = _import_tf()
    
    model = keras.Sequential([
        keras.layers.LSTM(units, return_sequences=True, input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(units // 2, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    
    return model


def train_lstm(X_train: np.ndarray, y_train: np.ndarray, 
               X_val: np.ndarray, y_val: np.ndarray,
               symbol: str, epochs: int = 50, batch_size: int = 32):
    """
    Train LSTM model with early stopping.
    
    Input shapes:
        X_train: (samples, lookback, features)
        y_train: (samples,)
    """
    tf, keras = _import_tf()
    
    logger.info(f"Training LSTM for {symbol}...")
    logger.info(f"Input shape: {X_train.shape}")
    
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"LSTM Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    return model, history


def save_lstm(model, symbol: str, timeframe: str):
    """Save LSTM model to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    filepath = MODELS_DIR / f"{symbol}_{timeframe}_lstm.keras"
    model.save(filepath)
    logger.info(f"LSTM model saved to {filepath}")
    return filepath


def load_lstm(symbol: str, timeframe: str):
    """Load LSTM model from disk."""
    tf, keras = _import_tf()
    filepath = MODELS_DIR / f"{symbol}_{timeframe}_lstm.keras"
    if filepath.exists():
        model = keras.models.load_model(filepath)
        logger.info(f"LSTM model loaded from {filepath}")
        return model
    logger.warning(f"LSTM model not found at {filepath}")
    return None


class LSTMPredictor:
    """Wrapper for LSTM inference compatible with the prediction pipeline."""
    
    def __init__(self, symbol: str, timeframe: str, lookback: int = 50):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback
        self.model = load_lstm(symbol, timeframe)
    
    def predict(self, features_sequence: np.ndarray) -> tuple[int, float]:
        """
        Predict on a sequence of features.
        
        Args:
            features_sequence: shape (lookback, n_features)
        
        Returns:
            (prediction, confidence)
        """
        if self.model is None:
            return 0, 0.5
        
        X = features_sequence.reshape(1, *features_sequence.shape)
        prob = float(self.model.predict(X, verbose=0)[0][0])
        prediction = 1 if prob > 0.5 else 0
        confidence = prob if prediction == 1 else (1 - prob)
        
        return prediction, confidence
