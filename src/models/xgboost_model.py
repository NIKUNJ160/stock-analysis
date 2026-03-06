import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

from config.settings import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger("Models.XGBoost")


def train_xgb(X_train: pd.DataFrame, y_train: pd.Series, symbol: str) -> XGBClassifier:
    """Trains an XGBoost classifier."""
    logger.info(f"Training XGBoost for {symbol}...")
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        use_label_encoder=False,
    )
    
    model.fit(X_train, y_train, verbose=False)
    return model


def evaluate_xgb(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluates the model and returns metrics."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Classification Report:\n{classification_report(y_test, preds)}")
    
    # Feature importance
    importance = dict(zip(X_test.columns, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info(f"Top 10 features: {top_features}")
    
    return {"accuracy": acc, "report": report, "feature_importance": importance}


def save_xgb(model: XGBClassifier, symbol: str, timeframe: str):
    """Saves XGBoost model to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    filepath = MODELS_DIR / f"{symbol}_{timeframe}_xgb.pkl"
    joblib.dump(model, filepath)
    logger.info(f"XGBoost model saved to {filepath}")
    return filepath
