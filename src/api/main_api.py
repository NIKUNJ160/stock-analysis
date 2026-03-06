from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from pathlib import Path
from datetime import datetime

from config.settings import TARGET_SYMBOLS, BASE_DIR, API_HOST, API_PORT
from src.utils.logger import get_logger
from src.utils.helpers import safe_read_json

logger = get_logger("API")

app = FastAPI(
    title="Live Quant Trading API",
    description="Real-time quantitative trading system API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """System health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "symbols": TARGET_SYMBOLS,
    }


@app.get("/signals")
async def get_all_signals():
    """Get latest trading signals for all symbols."""
    signals_path = BASE_DIR / "data" / "latest_signals.json"
    signals = safe_read_json(signals_path)
    
    if not signals:
        return {"signals": {}, "message": "No signals generated yet. Start the engine first."}
    
    return {"signals": signals, "count": len(signals)}


@app.get("/signals/{symbol}")
async def get_signal(symbol: str):
    """Get latest signal for a specific symbol."""
    signals_path = BASE_DIR / "data" / "latest_signals.json"
    signals = safe_read_json(signals_path)
    
    if symbol not in signals:
        raise HTTPException(status_code=404, detail=f"No signal found for {symbol}")
    
    return {"symbol": symbol, "signal": signals[symbol]}


@app.get("/portfolio")
async def get_portfolio():
    """Get current portfolio status."""
    portfolio_path = BASE_DIR / "data" / "portfolio.json"
    portfolio = safe_read_json(portfolio_path)
    
    return portfolio or {"message": "No portfolio data. Engine not running."}


@app.get("/backtest")
async def get_backtest_results():
    """Get latest backtest results."""
    results_path = BASE_DIR / "data" / "backtest_results" / "backtest_summary.json"
    results = safe_read_json(results_path)
    
    if not results:
        return {"message": "No backtest results. Run the backtester first."}
    
    return {"results": results}


@app.get("/backtest/{symbol}")
async def get_backtest_symbol(symbol: str):
    """Get backtest results for a specific symbol."""
    results_path = BASE_DIR / "data" / "backtest_results" / "backtest_summary.json"
    results = safe_read_json(results_path)
    
    if symbol not in results:
        raise HTTPException(status_code=404, detail=f"No backtest results for {symbol}")
    
    return {"symbol": symbol, "metrics": results[symbol]}


@app.get("/symbols")
async def list_symbols():
    """List all target symbols."""
    return {"symbols": TARGET_SYMBOLS}


@app.get("/models")
async def list_models():
    """List all trained models."""
    models_dir = BASE_DIR / "models"
    if not models_dir.exists():
        return {"models": []}
    
    models = []
    for f in models_dir.iterdir():
        if f.suffix in ['.pkl', '.keras', '.h5']:
            models.append({
                "filename": f.name,
                "size_mb": f.stat().st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
    
    return {"models": models, "count": len(models)}


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")
