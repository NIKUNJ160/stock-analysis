import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.settings import MODELS_DIR

def train_rf(X_train: pd.DataFrame, y_train: pd.Series, symbol: str) -> RandomForestClassifier:
    """Trains a Random Forest classifier."""
    print(f"Training Random Forest for {symbol}...")
    
    # Very basic hyperparams for stable MVP
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10, 
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1 # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluates the model and prints classification metrics."""
    preds = model.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))

def save_model(model: RandomForestClassifier, symbol: str, timeframe: str):
    """Serializes the trained model to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    filepath = MODELS_DIR / f"{symbol}_{timeframe}_rf.pkl"
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
    return filepath
