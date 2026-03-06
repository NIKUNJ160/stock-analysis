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
os.makedirs(DATA_DIR / "processed", exist_ok=True)
os.makedirs(DATA_DIR / "features", exist_ok=True)
os.makedirs(DATA_DIR / "backtest_results", exist_ok=True)

# ═══════════════════════════════════════════
#  Market Data Configuration
# ═══════════════════════════════════════════
TARGET_SYMBOLS = ["^NSEI", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
TIMEFRAME = "5m"
HISTORY_DAYS = 60
MULTI_TIMEFRAMES = ["15m", "1h"]  # Higher timeframes for MTF analysis

# WebSocket Configuration
WS_URL = "ws://localhost:8765"  # For live data feed
USE_MOCK_FEED = True  # Use historical data replay instead of live WS
MOCK_EMIT_DELAY = 0.05  # Seconds between mock candle emissions

# ═══════════════════════════════════════════
#  Model Configuration
# ═══════════════════════════════════════════
TRAIN_TEST_SPLIT = 0.8
PREDICT_HORIZON = 1
TARGET_COLUMN = "close"

# Ensemble weights (RF + XGBoost + LSTM)
ENSEMBLE_WEIGHTS = {
    "rf": 0.4,
    "xgb": 0.4,
    "lstm": 0.2,
}

# LSTM Configuration
LSTM_LOOKBACK = 50
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32

# ═══════════════════════════════════════════
#  Signal Engine Configuration
# ═══════════════════════════════════════════
SIGNAL_MIN_CONFIDENCE = 0.3
SIGNAL_MIN_MODEL_CONFIDENCE = 0.55
SIGNAL_COOLDOWN_SECONDS = 300  # 5 min cooldown between signals per symbol
SIGNAL_MAX_VOLATILITY_RATIO = 3.0

# ═══════════════════════════════════════════
#  Risk Management Configuration
# ═══════════════════════════════════════════
MAX_CAPITAL = 1_000_000
MAX_RISK_PER_TRADE = 0.02       # 2% of capital
MAX_OPEN_POSITIONS = 5
MAX_EXPOSURE_PER_SYMBOL = 0.20  # 20% of capital per symbol
MAX_DAILY_LOSS = 0.05           # 5% max daily loss

# Stop-Loss Configuration
SL_ATR_MULTIPLIER = 1.5
SL_TRAILING_ENABLED = True
SL_BREAKEVEN_AFTER_CANDLES = 20

# ═══════════════════════════════════════════
#  Infrastructure
# ═══════════════════════════════════════════
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# API Configuration
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = 8000

# Feature Buffer
MAX_CANDLE_BUFFER = 200  # Keep latest N candles in memory
