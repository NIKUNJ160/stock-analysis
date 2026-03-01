import asyncio
import joblib
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from config.settings import MODELS_DIR, TARGET_SYMBOLS, TIMEFRAME

class RealtimePredictor:
    def __init__(self):
        self.models = {}
        self.load_models()
        
    def load_models(self):
        """Loads models from disk into memory."""
        for symbol in TARGET_SYMBOLS:
            filepath = MODELS_DIR / f"{symbol}_{TIMEFRAME}_rf.pkl"
            if filepath.exists():
                self.models[symbol] = joblib.load(filepath)
                print(f"[ModelService] Loaded model for {symbol}")
            else:
                print(f"[ModelService] Warning: No trained model found for {symbol} at {filepath}")

    async def prediction_loop(self, feature_queue: asyncio.Queue, signal_queue: asyncio.Queue):
        """
        Continuously listens for new feature vectors, runs inference, and passes signals.
        """
        print("[ModelService] Listening for incoming features...")
        while True:
            packet = await feature_queue.get()
            symbol = packet['symbol']
            timestamp = packet['timestamp']
            
            # The feature vector from the feature engine
            features_df = packet['features'] 
            
            # Make sure we have a model
            if symbol not in self.models:
                print(f"[ModelService] Skipping prediction for {symbol}: No model loaded.")
                feature_queue.task_done()
                continue
                
            model = self.models[symbol]
            
            # Run Inference
            # predict_proba returns array like [[prob_class0, prob_class1]]
            probabilities = model.predict_proba(features_df)[0]
            prediction = model.predict(features_df)[0]
            
            confidence = float(probabilities[prediction])
            
            # Construct Signal
            signal_msg = {
                "symbol": symbol,
                "timestamp": timestamp,
                "model_prediction": "BUY" if prediction == 1 else "HOLD/SELL",
                "confidence": confidence,
                "raw_features_used": features_df.to_dict('records')[0] # useful for debugging
            }
            
            await signal_queue.put(signal_msg)
            feature_queue.task_done()
