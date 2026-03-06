import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

from config.settings import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger("Models.RandomForest")


def train_rf(X_train: pd.DataFrame, y_train: pd.Series, symbol: str) -> RandomForestClassifier:
    """Trains a Random Forest classifier."""
    logger.info(f"Training Random Forest for {symbol}...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluates the model and returns classification metrics."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Classification Report:\n{classification_report(y_test, preds)}")
    
    return {"accuracy": acc, "report": report}


def save_model(model: RandomForestClassifier, symbol: str, timeframe: str):
    """Serializes the trained model to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    filepath = MODELS_DIR / f"{symbol}_{timeframe}_rf.pkl"
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")
    return filepath
