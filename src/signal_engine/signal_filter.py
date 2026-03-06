import time
from typing import Optional
from dataclasses import dataclass, field

from src.signal_engine.signal_generator import TradingSignal, SignalStrength
from src.utils.logger import get_logger

logger = get_logger("SignalEngine.Filter")


@dataclass
class FilterConfig:
    min_confidence: float = 0.3           # Minimum signal confidence to pass
    min_model_confidence: float = 0.55    # Minimum model probability to pass
    cooldown_seconds: float = 300         # Min seconds between signals for same symbol
    max_volatility_ratio: float = 3.0     # Max volume_ratio_5 to consider (avoids panic)
    blocked_strengths: list = field(default_factory=lambda: [SignalStrength.HOLD])


class SignalFilter:
    """
    Filters trading signals based on confidence, volatility, cooldown,
    and other safety gates before they reach the Risk Engine.
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
        self.last_signal_time: dict[str, float] = {}  # symbol -> unix timestamp
        self.signal_counts: dict[str, int] = {}  # symbol -> count today
    
    def should_pass(self, signal: TradingSignal, features_row: Optional[dict] = None) -> tuple[bool, str]:
        """
        Returns (should_pass, reason) tuple.
        """
        symbol = signal.symbol
        
        # 1. Block HOLD signals
        if signal.strength in self.config.blocked_strengths:
            return False, f"Signal strength {signal.strength.value} is blocked"
        
        # 2. Minimum confidence gate
        if signal.confidence < self.config.min_confidence:
            return False, f"Confidence {signal.confidence:.1%} below minimum {self.config.min_confidence:.1%}"
        
        # 3. Minimum model confidence
        if signal.model_confidence < self.config.min_model_confidence:
            return False, f"Model confidence {signal.model_confidence:.1%} below minimum"
        
        # 4. Cooldown period
        now = time.time()
        if symbol in self.last_signal_time:
            elapsed = now - self.last_signal_time[symbol]
            if elapsed < self.config.cooldown_seconds:
                remaining = self.config.cooldown_seconds - elapsed
                return False, f"Cooldown active for {symbol}: {remaining:.0f}s remaining"
        
        # 5. Volatility gate
        if features_row:
            vol_ratio = features_row.get('volume_ratio_5', 1.0)
            if vol_ratio > self.config.max_volatility_ratio:
                return False, f"Volume ratio {vol_ratio:.1f} exceeds max {self.config.max_volatility_ratio}"
        
        # Signal passes — update cooldown tracker
        self.last_signal_time[symbol] = now
        self.signal_counts[symbol] = self.signal_counts.get(symbol, 0) + 1
        
        logger.info(f"Signal PASSED for {symbol}: {signal.strength.value} (conf={signal.confidence:.1%})")
        return True, "passed"
    
    def reset_daily(self):
        """Reset daily counters (call at market open)."""
        self.signal_counts.clear()
        logger.info("Daily signal counters reset")
