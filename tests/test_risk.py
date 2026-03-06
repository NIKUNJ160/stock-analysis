import pytest
from src.risk_management.risk_manager import RiskManager, RiskConfig
from src.risk_management.stoploss import StopLossEngine
from src.risk_management.position_sizing import PositionSizer
from src.signal_engine.signal_generator import TradingSignal, SignalStrength


def _make_buy_signal(symbol: str = "TCS.NS") -> TradingSignal:
    return TradingSignal(
        symbol=symbol, timestamp="2024-01-01T10:00:00",
        strength=SignalStrength.STRONG_BUY, confidence=0.8,
        model_prediction=1, model_confidence=0.75,
        rsi_value=28, macd_signal="bullish",
        trend_alignment="bullish", reasons=["Test"]
    )


class TestRiskManager:
    def test_approve_valid_trade(self):
        rm = RiskManager(RiskConfig(max_capital=1_000_000))
        signal = _make_buy_signal()
        approved, reason, size = rm.validate_trade(signal, 100, 1000, 980)
        assert approved, f"Should approve: {reason}"
        assert size > 0
    
    def test_reject_when_max_positions_reached(self):
        rm = RiskManager(RiskConfig(max_capital=1_000_000, max_open_positions=1))
        rm.open_position("RELIANCE.NS", "LONG", 2500, 10, 2450, 2600)
        
        signal = _make_buy_signal("TCS.NS")
        approved, reason, _ = rm.validate_trade(signal, 100, 3500, 3400)
        assert not approved
        assert "Max positions" in reason
    
    def test_close_position_pnl(self):
        rm = RiskManager()
        rm.open_position("TCS.NS", "LONG", 3500, 10, 3400, 3700)
        pnl = rm.close_position("TCS.NS", 3600)
        assert pnl == (3600 - 3500) * 10  # 1000
    
    def test_stop_loss_trigger(self):
        rm = RiskManager()
        rm.open_position("TCS.NS", "LONG", 3500, 10, 3400, 3700)
        to_close = rm.check_stop_losses({"TCS.NS": 3350})
        assert "TCS.NS" in to_close


class TestStopLoss:
    def test_atr_stop_long(self):
        sl, tp = StopLossEngine.atr_stop(100, 2.0, multiplier=1.5, side="LONG")
        assert sl == 97.0  # 100 - 2*1.5
        assert tp == 106.0  # 100 + 2*1.5*2
    
    def test_atr_stop_short(self):
        sl, tp = StopLossEngine.atr_stop(100, 2.0, multiplier=1.5, side="SHORT")
        assert sl == 103.0
        assert tp == 94.0
    
    def test_trailing_stop_only_tightens(self):
        new_sl = StopLossEngine.trailing_stop_update(105, 97, 3, "LONG")
        assert new_sl == 102  # 105 - 3 = 102, which is > 97
        
        # Should NOT loosen
        new_sl2 = StopLossEngine.trailing_stop_update(99, 102, 3, "LONG")
        assert new_sl2 == 102  # max(96, 102) = 102


class TestPositionSizing:
    def test_fixed_fraction(self):
        size = PositionSizer.fixed_fraction(1_000_000, 0.02, 100, 95)
        # Risk = 1M * 0.02 = 20,000. Risk/share = 5. Size = 4000
        assert size == 4000
    
    def test_fixed_fraction_zero_risk(self):
        size = PositionSizer.fixed_fraction(1_000_000, 0.02, 100, 100)
        assert size == 0
    
    def test_volatility_adjusted(self):
        size = PositionSizer.volatility_adjusted(1_000_000, 0.02, 100, 2.0, 2.0)
        # Risk = 20,000. Risk/share = 4. Size = 5000
        assert size == 5000
