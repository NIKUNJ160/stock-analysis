import pandas as pd
import numpy as np
import joblib
import os
import json
from pathlib import Path

from config.settings import (
    TARGET_SYMBOLS, TIMEFRAME, RAW_DATA_DIR, MODELS_DIR
)
from src.feature_engineering.feature_builder import build_features
from src.backtesting.strategy import MLStrategy, Strategy
from src.backtesting.metrics import calculate_metrics, print_metrics_report
from src.risk_management.stoploss import StopLossEngine
from src.risk_management.position_sizing import PositionSizer
from src.signal_engine.signal_generator import SignalStrength
from src.signal_engine.signal_filter import SignalFilter, FilterConfig
from src.utils.logger import get_logger

logger = get_logger("Backtesting.Engine")

# Raw OHLCV columns to drop for model inference
RAW_OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']


class Backtester:
    """
    Event-driven backtester that replays historical data through the full pipeline:
    
    Historical Data → Feature Engine → Model Prediction → Signal Generator → 
    Signal Filter → Risk Check → Position Sizing → SL/TP → Trade Execution → Metrics
    """
    
    def __init__(self, strategy: Strategy = None, initial_capital: float = 1_000_000,
                 risk_per_trade: float = 0.02):
        self.strategy = strategy or MLStrategy()
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.sl_engine = StopLossEngine()
        self.sizer = PositionSizer()
        self.signal_filter = SignalFilter(FilterConfig(
            min_confidence=0.3,
            cooldown_seconds=0,  # No cooldown in backtest
        ))
    
    def run(self, symbol: str) -> dict:
        """
        Run backtest for a single symbol.
        
        Returns:
            Dict with 'metrics', 'trades', 'equity_curve'
        """
        logger.info(f"{'='*60}")
        logger.info(f"Running backtest for {symbol} using {self.strategy.name()}")
        logger.info(f"{'='*60}")
        
        # 1. Load processed data
        processed_dir = RAW_DATA_DIR.parent / 'processed'
        data_file = processed_dir / f"{symbol}_{TIMEFRAME}_cleaned.csv"
        
        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            return {"error": f"Data not found for {symbol}"}
        
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(df)} candles")
        
        # 2. Build features on full dataset
        features_df = build_features(df)
        logger.info(f"Generated {len(features_df)} feature rows")
        
        # 3. Load model
        model_path = MODELS_DIR / f"{symbol}_{TIMEFRAME}_rf.pkl"
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return {"error": f"Model not found for {symbol}"}
        from src.utils.helpers import safe_load_model
        model = safe_load_model(model_path)
        
        # 4. Simulate trading
        capital = self.initial_capital
        position = None  # {side, entry, qty, sl, tp, entry_idx}
        trades = []
        
        # Get feature columns (exclude raw OHLCV)
        feature_cols = [c for c in features_df.columns if c not in RAW_OHLCV_COLS]
        
        for i in range(60, len(features_df)):
            row = features_df.iloc[i]
            current_price = row.get('close', 0)
            atr = row.get('atr_14', current_price * 0.01)  # Fallback ATR
            timestamp = str(features_df.index[i])
            
            # Check exits if in a position
            if position is not None:
                should_exit = False
                exit_reason = ""
                
                if position['side'] == 'LONG':
                    if current_price <= position['sl']:
                        should_exit = True
                        exit_reason = "stop_loss"
                    elif current_price >= position['tp']:
                        should_exit = True
                        exit_reason = "take_profit"
                else:  # SHORT
                    if current_price >= position['sl']:
                        should_exit = True
                        exit_reason = "stop_loss"
                    elif current_price <= position['tp']:
                        should_exit = True
                        exit_reason = "take_profit"
                
                if should_exit:
                    if position['side'] == 'LONG':
                        pnl = (current_price - position['entry']) * position['qty']
                    else:
                        pnl = (position['entry'] - current_price) * position['qty']
                    
                    trades.append({
                        'symbol': symbol,
                        'side': position['side'],
                        'entry_price': position['entry'],
                        'exit_price': current_price,
                        'quantity': position['qty'],
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'timestamp': timestamp,
                    })
                    capital += pnl
                    position = None
            
            # Skip if already in a position
            if position is not None:
                continue
            
            # Model prediction
            try:
                X = features_df.iloc[[i]][feature_cols]
                if X.isna().any().any():
                    continue
                    
                prediction = model.predict(X)[0]
                probabilities = model.predict_proba(X)[0]
                confidence = float(probabilities[prediction])
                
                model_output = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'model_prediction': int(prediction),
                    'confidence': confidence,
                }
            except Exception as e:
                logger.debug(f"Skipping prediction at idx {i}: {e}")
                continue
            
            # Strategy decision
            signal = self.strategy.on_candle(features_df.iloc[max(0,i-5):i+1], model_output)
            
            if signal is None:
                continue
            
            # Signal filter
            passed, reason = self.signal_filter.should_pass(signal)
            if not passed:
                continue
            
            # Calculate SL/TP
            side = "LONG" if signal.strength in [SignalStrength.STRONG_BUY, SignalStrength.BUY] else "SHORT"
            sl, tp = self.sl_engine.atr_stop(current_price, atr, multiplier=1.5, side=side)
            
            # Position sizing
            qty = self.sizer.fixed_fraction(capital, self.risk_per_trade, current_price, sl)
            if qty <= 0:
                continue
            
            # Open position
            position = {
                'side': side,
                'entry': current_price,
                'qty': qty,
                'sl': sl,
                'tp': tp,
                'entry_idx': i,
            }
        
        # Close any open position at end
        if position is not None:
            final_price = features_df.iloc[-1].get('close', 0)
            if position['side'] == 'LONG':
                pnl = (final_price - position['entry']) * position['qty']
            else:
                pnl = (position['entry'] - final_price) * position['qty']
            
            trades.append({
                'symbol': symbol,
                'side': position['side'],
                'entry_price': position['entry'],
                'exit_price': final_price,
                'quantity': position['qty'],
                'pnl': pnl,
                'exit_reason': 'end_of_data',
                'timestamp': str(features_df.index[-1]),
            })
        
        # 5. Calculate metrics
        metrics = calculate_metrics(trades, self.initial_capital)
        print_metrics_report(metrics)
        
        return {
            'symbol': symbol,
            'strategy': self.strategy.name(),
            'metrics': {k: v for k, v in metrics.items() if k != 'equity_curve'},
            'trades': trades,
            'equity_curve': metrics.get('equity_curve', []),
        }
    
    def run_all(self) -> dict:
        """Run backtest across all target symbols."""
        results = {}
        for symbol in TARGET_SYMBOLS:
            results[symbol] = self.run(symbol)
        
        # Save results
        results_dir = RAW_DATA_DIR.parent / 'backtest_results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary (without equity curve for JSON serialization)
        summary = {}
        for sym, res in results.items():
            if 'error' not in res:
                summary[sym] = res['metrics']
        
        with open(results_dir / "backtest_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_dir}")
        return results


if __name__ == "__main__":
    backtester = Backtester()
    backtester.run_all()
