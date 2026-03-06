import math
from src.utils.logger import get_logger

logger = get_logger("RiskEngine.PositionSizing")


class PositionSizer:
    """
    Determines the optimal position size based on risk parameters.
    Supports fixed-fraction, Kelly criterion, and volatility-adjusted sizing.
    """
    
    @staticmethod
    def fixed_fraction(capital: float, risk_fraction: float, entry_price: float, 
                       stop_loss: float) -> float:
        """
        Fixed fractional position sizing.
        Risk a fixed percentage of capital on each trade.
        
        Example: $100k capital, 2% risk, entry=100, SL=95
        Risk = $100k * 0.02 = $2,000
        Risk per share = $100 - $95 = $5
        Position size = $2,000 / $5 = 400 shares
        """
        risk_amount = capital * risk_fraction
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            logger.warning("Risk per share is zero or negative")
            return 0
        
        size = risk_amount / risk_per_share
        logger.debug(f"Fixed-fraction: capital={capital:.0f}, risk={risk_amount:.0f}, size={size:.2f}")
        return math.floor(size)
    
    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float, 
                        capital: float, entry_price: float, kelly_fraction: float = 0.5) -> float:
        """
        Kelly Criterion position sizing (half-Kelly for safety).
        
        Kelly% = W - [(1 - W) / R]
        where W = win rate, R = win/loss ratio
        """
        if avg_loss == 0:
            return 0
        
        win_loss_ratio = avg_win / abs(avg_loss)
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Use fractional Kelly (default half-Kelly) for safety
        kelly_pct = max(0, kelly_pct * kelly_fraction)
        
        # Cap at 25% of capital
        kelly_pct = min(kelly_pct, 0.25)
        
        position_value = capital * kelly_pct
        size = position_value / entry_price if entry_price > 0 else 0
        
        logger.debug(f"Kelly: win_rate={win_rate:.1%}, ratio={win_loss_ratio:.2f}, "
                      f"kelly={kelly_pct:.1%}, size={size:.2f}")
        return math.floor(size)
    
    @staticmethod
    def volatility_adjusted(capital: float, risk_fraction: float, 
                            entry_price: float, atr: float, 
                            atr_multiplier: float = 2.0) -> float:
        """
        Volatility-adjusted position sizing using ATR.
        Higher volatility → smaller position, lower volatility → larger position.
        """
        if atr <= 0:
            logger.warning("ATR is zero or negative")
            return 0
        
        risk_amount = capital * risk_fraction
        risk_per_share = atr * atr_multiplier
        
        size = risk_amount / risk_per_share
        logger.debug(f"Volatility-adjusted: ATR={atr:.2f}, risk_per_share={risk_per_share:.2f}, size={size:.2f}")
        return math.floor(size)
    
    @staticmethod
    def equal_weight(capital: float, num_positions: int, entry_price: float) -> float:
        """Equal-weight allocation across all positions."""
        if num_positions <= 0 or entry_price <= 0:
            return 0
        
        allocation = capital / num_positions
        size = allocation / entry_price
        return math.floor(size)
