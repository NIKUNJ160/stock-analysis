import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional

from config.settings import MODELS_DIR, TARGET_SYMBOLS, TIMEFRAME
from src.utils.logger import get_logger

logger = get_logger("Models.Ensemble")


class EnsemblePredictor:
    """
    Voting ensemble combining RandomForest + XGBoost (+ optional LSTM).
    
    Uses weighted soft voting:
    - RF confidence * weight_rf
    - XGB confidence * weight_xgb
    - LSTM confidence * weight_lstm
    
    Final prediction = argmax of weighted average probabilities.
    """
    
    def __init__(self, weights: dict = None):
        self.weights = weights or {
            'rf': 0.4,
            'xgb': 0.4,
            'lstm': 0.2,
        }
        self.models: dict[str, dict] = {}  # symbol -> {rf: model, xgb: model, lstm: model}
    
    def load_models(self, symbols: list[str] = None):
        """Load all available models for each symbol."""
        symbols = symbols or TARGET_SYMBOLS
        
        for symbol in symbols:
            self.models[symbol] = {}
            
            # RandomForest
            rf_path = MODELS_DIR / f"{symbol}_{TIMEFRAME}_rf.pkl"
            if rf_path.exists():
                self.models[symbol]['rf'] = safe_load_model(rf_path)
                logger.info(f"Loaded RF model for {symbol}")
            
            # XGBoost
            xgb_path = MODELS_DIR / f"{symbol}_{TIMEFRAME}_xgb.pkl"
            if xgb_path.exists():
                self.models[symbol]['xgb'] = safe_load_model(xgb_path)
                logger.info(f"Loaded XGB model for {symbol}")
            
            # LSTM (optional — loaded separately due to TF dependency)
            try:
                from src.models.lstm_model import load_lstm
                lstm_model = load_lstm(symbol, TIMEFRAME)
                if lstm_model is not None:
                    self.models[symbol]['lstm'] = lstm_model
            except ImportError:
                pass
            
            if not self.models[symbol]:
                logger.warning(f"No models found for {symbol}")
    
    def predict(self, symbol: str, features_df: pd.DataFrame, 
                lstm_sequence: np.ndarray = None) -> dict:
        """
        Ensemble prediction with weighted voting.
        
        Args:
            symbol: Stock symbol
            features_df: Latest feature row(s) — shape (1, n_features)
            lstm_sequence: Optional LSTM input — shape (lookback, n_features)
        
        Returns:
            Dict with prediction, confidence, and per-model details
        """
        if symbol not in self.models:
            return {"prediction": 0, "confidence": 0.5, "error": "No models loaded"}
        
        model_dict = self.models[symbol]
        predictions = {}
        weighted_probs = np.array([0.0, 0.0])  # [prob_class_0, prob_class_1]
        total_weight = 0
        
        # RandomForest
        if 'rf' in model_dict:
            try:
                rf_probs = model_dict['rf'].predict_proba(features_df)[0]
                rf_pred = int(np.argmax(rf_probs))
                predictions['rf'] = {
                    'prediction': rf_pred,
                    'confidence': float(rf_probs[rf_pred]),
                    'probabilities': rf_probs.tolist(),
                }
                weighted_probs += rf_probs * self.weights['rf']
                total_weight += self.weights['rf']
            except Exception as e:
                logger.warning(f"RF prediction failed for {symbol}: {e}")
        
        # XGBoost
        if 'xgb' in model_dict:
            try:
                xgb_probs = model_dict['xgb'].predict_proba(features_df)[0]
                xgb_pred = int(np.argmax(xgb_probs))
                predictions['xgb'] = {
                    'prediction': xgb_pred,
                    'confidence': float(xgb_probs[xgb_pred]),
                    'probabilities': xgb_probs.tolist(),
                }
                weighted_probs += xgb_probs * self.weights['xgb']
                total_weight += self.weights['xgb']
            except Exception as e:
                logger.warning(f"XGB prediction failed for {symbol}: {e}")
        
        # LSTM
        if 'lstm' in model_dict and lstm_sequence is not None:
            try:
                lstm_prob = float(model_dict['lstm'].predict(
                    lstm_sequence.reshape(1, *lstm_sequence.shape), verbose=0
                )[0][0])
                lstm_probs = np.array([1 - lstm_prob, lstm_prob])
                lstm_pred = 1 if lstm_prob > 0.5 else 0
                predictions['lstm'] = {
                    'prediction': lstm_pred,
                    'confidence': float(lstm_probs[lstm_pred]),
                    'probabilities': lstm_probs.tolist(),
                }
                weighted_probs += lstm_probs * self.weights['lstm']
                total_weight += self.weights['lstm']
            except Exception as e:
                logger.warning(f"LSTM prediction failed for {symbol}: {e}")
        
        # Normalize weights
        if total_weight > 0:
            weighted_probs /= total_weight
        
        # Final ensemble prediction
        ensemble_pred = int(np.argmax(weighted_probs))
        ensemble_conf = float(weighted_probs[ensemble_pred])
        
        # Voting agreement
        votes = [v['prediction'] for v in predictions.values()]
        agreement = votes.count(ensemble_pred) / len(votes) if votes else 0
        
        result = {
            'prediction': ensemble_pred,
            'prediction_label': 'BUY' if ensemble_pred == 1 else 'HOLD/SELL',
            'confidence': ensemble_conf,
            'agreement': agreement,
            'weighted_probabilities': weighted_probs.tolist(),
            'individual_models': predictions,
            'models_used': list(predictions.keys()),
        }
        
        logger.debug(f"Ensemble {symbol}: {result['prediction_label']} "
                      f"(conf={ensemble_conf:.1%}, agree={agreement:.0%})")
        
        return result
