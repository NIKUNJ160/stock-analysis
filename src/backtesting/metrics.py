import numpy as np
import pandas as pd
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger("Backtesting.Metrics")


def calculate_metrics(trade_log: list[dict], initial_capital: float = 1_000_000) -> dict:
    """
    Calculate comprehensive backtesting performance metrics.
    
    Args:
        trade_log: List of trade dicts with 'pnl', 'timestamp' keys
        initial_capital: Starting capital
    
    Returns:
        Dict of performance metrics
    """
    if not trade_log:
        return {"error": "No trades to analyze"}
    
    pnls = [t['pnl'] for t in trade_log]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    total_trades = len(pnls)
    total_pnl = sum(pnls)
    
    # Win rate
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    
    # Average win / loss
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Equity curve and drawdown
    equity_curve = [initial_capital]
    for pnl in pnls:
        equity_curve.append(equity_curve[-1] + pnl)
    
    equity_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - peak) / peak
    max_drawdown = float(np.min(drawdown))
    
    # Return metrics
    total_return = (equity_curve[-1] - initial_capital) / initial_capital
    
    # Sharpe Ratio (annualized, assuming 5-min candles ≈ 252*78 periods/year)
    if len(pnls) > 1:
        returns = np.array(pnls) / initial_capital
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 78)
    else:
        sharpe = 0
    
    # Sortino Ratio (downside only)
    if len(pnls) > 1:
        downside_returns = np.array([r for r in pnls if r < 0]) / initial_capital
        if len(downside_returns) > 0:
            sortino = np.mean(np.array(pnls) / initial_capital) / (np.std(downside_returns) + 1e-8) * np.sqrt(252 * 78)
        else:
            sortino = float('inf')
    else:
        sortino = 0
    
    # Expectancy (avg profit per trade)
    expectancy = total_pnl / total_trades if total_trades > 0 else 0
    
    # Payoff ratio (avg win / avg loss)
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    # Consecutive wins/losses
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0
    for pnl in pnls:
        if pnl > 0:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
    
    metrics = {
        "total_trades": total_trades,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "largest_win": max(wins) if wins else 0,
        "largest_loss": min(losses) if losses else 0,
        "profit_factor": profit_factor,
        "payoff_ratio": payoff_ratio,
        "expectancy": expectancy,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "equity_curve": equity_curve,
        "initial_capital": initial_capital,
        "final_capital": equity_curve[-1],
    }
    
    return metrics


def print_metrics_report(metrics: dict):
    """Pretty-print a metrics report."""
    if "error" in metrics:
        logger.warning(f"BACKTEST ERROR: {metrics['error']}")
        return
        
    logger.info("=" * 60)
    logger.info("BACKTEST PERFORMANCE REPORT")
    logger.info("=" * 60)
    logger.info(f"Total Trades:          {metrics['total_trades']}")
    logger.info(f"Win Rate:              {metrics['win_rate']:.1%}")
    logger.info(f"Total P&L:             ₹{metrics['total_pnl']:,.2f}")
    logger.info(f"Total Return:          {metrics['total_return']:.2%}")
    logger.info(f"Profit Factor:         {metrics['profit_factor']:.2f}")
    logger.info(f"Sharpe Ratio:          {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Sortino Ratio:         {metrics['sortino_ratio']:.2f}")
    logger.info(f"Max Drawdown:          {metrics['max_drawdown']:.2%}")
    logger.info(f"Avg Win:               ₹{metrics['avg_win']:,.2f}")
    logger.info(f"Avg Loss:              ₹{metrics['avg_loss']:,.2f}")
    logger.info(f"Payoff Ratio:          {metrics['payoff_ratio']:.2f}")
    logger.info(f"Expectancy/Trade:      ₹{metrics['expectancy']:,.2f}")
    logger.info(f"Max Consec. Wins:      {metrics['max_consecutive_wins']}")
    logger.info(f"Max Consec. Losses:    {metrics['max_consecutive_losses']}")
    logger.info(f"Final Capital:         ₹{metrics['final_capital']:,.2f}")
    logger.info("=" * 60)
