import asyncio
import joblib
import pandas as pd
import numpy as np

from config.settings import MODELS_DIR, TARGET_SYMBOLS, TIMEFRAME
from src.utils.logger import get_logger

logger = get_logger("ModelService")


class RealtimePredictor:
    def __init__(self):
        self.models = {}
        self.load_models()
        
    def load_models(self):
        from src.utils.helpers import safe_load_model
        for symbol in TARGET_SYMBOLS:
            filepath = MODELS_DIR / f"{symbol}_{TIMEFRAME}_rf.pkl"
            if filepath.exists():
                self.models[symbol] = safe_load_model(filepath)
                logger.info(f"Loaded model for {symbol}")
            else:
                logger.warning(f"No trained model found for {symbol} at {filepath}")

    async def prediction_loop(self, feature_queue: asyncio.Queue, signal_queue: asyncio.Queue):
        """
        Continuously listens for new feature vectors, runs inference, and passes signals.
        """
        logger.info("Listening for incoming features...")
        while True:
            packet = await feature_queue.get()
            symbol = packet['symbol']
            timestamp = packet['timestamp']
            features_df = packet['features']
            
            if symbol not in self.models:
                logger.warning(f"Skipping prediction for {symbol}: No model loaded.")
                feature_queue.task_done()
                continue
            
            try:
                model = self.models[symbol]
                
                # Clean features: drop raw OHLCV to avoid leakage during inference
                drop_cols = ['open', 'high', 'low', 'close', 'volume']
                feature_cols = [c for c in features_df.columns if c not in drop_cols]
                clean_features = features_df[feature_cols]
                
                # Check for NaN/Inf values
                if clean_features.isna().any().any() or np.isinf(clean_features.values).any():
                    logger.warning(f"NaN/Inf detected in features for {symbol}, skipping")
                    feature_queue.task_done()
                    continue
                
                # Check for schema drift using model.feature_names_in_ if available
                if hasattr(model, 'feature_names_in_'):
                    expected_features = list(model.feature_names_in_)
                    actual_features = clean_features.columns.tolist()
                    if expected_features != actual_features:
                        class SchemaMismatchError(Exception): pass
                        raise SchemaMismatchError(f"SCHEMA MISMATCH for {symbol}! Expected {len(expected_features)} features, got {len(actual_features)}. Stopping service to prevent bad predictions.")
                
                # Run Inference
                probabilities = model.predict_proba(clean_features)[0]
                prediction = model.predict(clean_features)[0]
                pred_index = list(model.classes_).index(prediction)
                confidence = float(probabilities[pred_index])
                
                signal_msg = {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "model_prediction": int(prediction),
                    "prediction_label": "BUY" if prediction == 1 else "HOLD/SELL",
                    "confidence": confidence,
                    "probabilities": probabilities.tolist(),
                    "features_used": clean_features.columns.tolist(),
                    "features": features_df,
                    "close_price": packet.get('close_price', features_df['close'].iloc[-1] if 'close' in features_df.columns else 0)
                }
                
                await signal_queue.put(signal_msg)
                logger.debug(f"Prediction for {symbol}: {'BUY' if prediction == 1 else 'HOLD/SELL'} ({confidence:.2%})")
                
            except Exception as e:
                logger.error(f"Prediction error for {symbol}: {e}")
            
            feature_queue.task_done()
