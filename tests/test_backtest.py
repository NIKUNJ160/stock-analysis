import pytest
from src.backtesting.metrics import calculate_metrics


class TestMetrics:
    def test_basic_metrics(self):
        trades = [
            {"pnl": 1000, "timestamp": "2024-01-01"},
            {"pnl": -500, "timestamp": "2024-01-02"},
            {"pnl": 800, "timestamp": "2024-01-03"},
            {"pnl": -200, "timestamp": "2024-01-04"},
            {"pnl": 1500, "timestamp": "2024-01-05"},
        ]
        m = calculate_metrics(trades, initial_capital=100_000)
        
        assert m['total_trades'] == 5
        assert m['winning_trades'] == 3
        assert m['losing_trades'] == 2
        assert m['win_rate'] == 0.6
        assert m['total_pnl'] == 2600
    
    def test_empty_trades(self):
        m = calculate_metrics([])
        assert "error" in m
    
    def test_all_wins(self):
        trades = [{"pnl": 100, "timestamp": "2024-01-01"}] * 5
        m = calculate_metrics(trades)
        assert m['win_rate'] == 1.0
        assert m['max_consecutive_wins'] == 5
    
    def test_drawdown(self):
        trades = [
            {"pnl": 10000, "timestamp": "2024-01-01"},
            {"pnl": -5000, "timestamp": "2024-01-02"},
            {"pnl": -5000, "timestamp": "2024-01-03"},
        ]
        m = calculate_metrics(trades, initial_capital=100_000)
        assert m['max_drawdown'] < 0, "Max drawdown should be negative"
