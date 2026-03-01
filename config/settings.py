import os
from pathlib import Path

# Project Root Setup
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Training Targets (MVP)
TARGET_SYMBOLS = ["^NSEI", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"] # Using NIFTY 50 and top Indian stocks
TIMEFRAME = "5m" # 5-minute candles
HISTORY_DAYS = 60 # Number of days of history to fetch for training (yfinance limit for intraday)

# Model configuration
TRAIN_TEST_SPLIT = 0.8
PREDICT_HORIZON = 1 # Predict 1 candle ahead
TARGET_COLUMN = "close"

# Risk parameters (for Risk Engine later)
MAX_RISK_PER_TRADE = 0.02 # 2% of capital
