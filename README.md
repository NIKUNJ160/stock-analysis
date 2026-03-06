# 📈 Live Quant Trading System

AI-powered quantitative trading system for Indian and US equities using real-time feature engineering, ensemble ML models, and risk-controlled signal generation.

## Architecture

```
Live Market Data (WebSocket / Mock Feed)
        ↓
┌─ Market Data Service ──────────────┐
│  WebSocket Listener → Candle       │
│  Aggregator → Message Queue        │
└────────────────────────────────────┘
        ↓
┌─ Feature Engine ───────────────────┐
│  Technical Indicators (RSI, MACD,  │
│  Bollinger, ATR, EMA) +            │
│  Candlestick Shapes +              │
│  Multi-Timeframe Analysis          │
└────────────────────────────────────┘
        ↓
┌─ Model Service ────────────────────┐
│  Ensemble: RandomForest + XGBoost  │
│  + LSTM (weighted soft voting)     │
└────────────────────────────────────┘
        ↓
┌─ Signal Engine ────────────────────┐
│  Signal Generator (score-based)    │
│  → Signal Filter (confidence,     │
│  cooldown, volatility gates)       │
└────────────────────────────────────┘
        ↓
┌─ Risk Engine ──────────────────────┐
│  Position Sizing (Kelly/Fixed)     │
│  → Stop-Loss (ATR/Trailing)        │
│  → Risk Validation (exposure,     │
│  daily loss, max positions)        │
└────────────────────────────────────┘
        ↓
┌─ Execution ────────────────────────┐
│  Paper Trading Broker              │
│  (Zerodha Kite / IB ready)        │
└────────────────────────────────────┘
        ↓
  Dashboard (Streamlit) + API (FastAPI)
```

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
# Windows CMD:
venv\Scripts\activate
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fetch & process market data
python -m src.data_pipeline.fetch_data
python -m src.data_pipeline.clean_data

# 4. Build features & train models
python -m src.feature_engineering.feature_builder
python -m src.models.train_model

# 5. Run the live engine
python main.py

# 6. Or launch the dashboard
streamlit run app.py
```

## Project Structure

```
live/
├── config/settings.py          # All configuration
├── main.py                     # Async pipeline orchestrator
├── app.py                      # Streamlit dashboard
│
├── services/
│   ├── market_data_service/
│   │   ├── mock_feed.py        # Historical data replay
│   │   └── websocket_listener.py # Live WebSocket + Candle Aggregator
│   ├── model_service/
│   │   └── realtime_predictor.py # Real-time ML inference
│   └── execution_service/
│       ├── broker_api.py       # Abstract Broker + Paper Trading
│       └── order_manager.py    # Order lifecycle management
│
├── src/
│   ├── data_pipeline/          # Fetch + Clean historical data
│   ├── feature_engineering/
│   │   ├── indicators.py       # RSI, MACD, Bollinger, ATR, EMA
│   │   ├── candlestick_features.py
│   │   ├── feature_builder.py  # Main feature pipeline
│   │   └── multi_timeframe.py  # 5m → 15m → 1h analysis
│   ├── models/
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── lstm_model.py
│   │   ├── ensemble.py         # Weighted RF+XGB+LSTM voting
│   │   └── train_model.py
│   ├── signal_engine/
│   │   ├── signal_generator.py # Multi-factor signal scoring
│   │   └── signal_filter.py    # Confidence/cooldown/volatility gates
│   ├── risk_management/
│   │   ├── risk_manager.py     # Portfolio-level risk validation
│   │   ├── stoploss.py         # ATR/trailing/time-based stops
│   │   └── position_sizing.py  # Kelly/fixed-fraction/volatility sizing
│   ├── backtesting/
│   │   ├── backtester.py       # Event-driven backtest engine
│   │   ├── metrics.py          # Sharpe, Sortino, drawdown, etc.
│   │   └── strategy.py         # Strategy interface + implementations
│   ├── api/main_api.py         # FastAPI REST endpoints
│   └── utils/
│       ├── logger.py           # Structured logging
│       └── helpers.py          # JSON I/O, validation, formatting
│
├── infrastructure/
│   ├── redis_cache.py          # Redis cache (with in-memory fallback)
│   └── message_queue.py        # Async pub/sub message queue
│
├── tests/                      # pytest test suite
├── data/                       # Market data (raw/processed/features)
├── models/                     # Trained model files (.pkl/.keras)
└── requirements.txt
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## API Endpoints

Start the API server:

```bash
python -m src.api.main_api
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | System health check |
| `/signals` | GET | All latest trading signals |
| `/signals/{symbol}` | GET | Signal for specific symbol |
| `/portfolio` | GET | Portfolio status & positions |
| `/backtest` | GET | Backtest results summary |
| `/backtest/{symbol}` | GET | Symbol-specific backtest |
| `/symbols` | GET | List target symbols |
| `/models` | GET | List trained models |

## Running Backtests

```bash
python -m src.backtesting.backtester
```

## Target Symbols

| Symbol | Market |
|---|---|
| ^NSEI | NIFTY 50 Index |
| RELIANCE.NS | Reliance Industries |
| TCS.NS | Tata Consultancy Services |
| HDFCBANK.NS | HDFC Bank |

## Tech Stack

- **ML**: scikit-learn, XGBoost, TensorFlow/Keras (LSTM)
- **Data**: pandas, yfinance, ta (technical indicators)
- **Async**: asyncio, websockets
- **Dashboard**: Streamlit
- **API**: FastAPI + Uvicorn
- **Cache**: Redis (with in-memory fallback)
- **Testing**: pytest
