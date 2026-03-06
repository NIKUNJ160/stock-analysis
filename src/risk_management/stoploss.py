from src.utils.logger import get_logger

logger = get_logger("RiskEngine.StopLoss")


class StopLossEngine:
    """
    Calculates stop-loss and take-profit levels using ATR, 
    fixed percentage, and trailing stop strategies.
    """
    
    @staticmethod
    def atr_stop(entry_price: float, atr_value: float, multiplier: float = 1.5, 
                 side: str = "LONG") -> tuple[float, float]:
        """
        ATR-based stop-loss and take-profit.
        
        Args:
            entry_price: Trade entry price
            atr_value: Current ATR value
            multiplier: ATR multiplier for stop distance
            side: "LONG" or "SHORT"
            
        Returns:
            (stop_loss, take_profit)
        """
        stop_distance = atr_value * multiplier
        tp_distance = atr_value * multiplier * 2  # 2:1 reward-to-risk
        
        if side == "LONG":
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - tp_distance
        
        logger.debug(f"ATR Stop: entry={entry_price:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}")
        return stop_loss, take_profit
    
    @staticmethod
    def percentage_stop(entry_price: float, stop_pct: float = 0.02, 
                        tp_pct: float = 0.04, side: str = "LONG") -> tuple[float, float]:
        """
        Fixed percentage stop-loss and take-profit.
        """
        if side == "LONG":
            stop_loss = entry_price * (1 - stop_pct)
            take_profit = entry_price * (1 + tp_pct)
        else:
            stop_loss = entry_price * (1 + stop_pct)
            take_profit = entry_price * (1 - tp_pct)
        
        return stop_loss, take_profit
    
    @staticmethod
    def trailing_stop_update(current_price: float, current_stop: float, 
                             trail_distance: float, side: str = "LONG") -> float:
        """
        Update trailing stop-loss based on current price movement.
        Only tightens the stop, never loosens it.
        """
        if side == "LONG":
            new_stop = current_price - trail_distance
            return max(new_stop, current_stop)  # Only move up
        else:
            new_stop = current_price + trail_distance
            return min(new_stop, current_stop)  # Only move down
    
    @staticmethod
    def time_based_tighten(entry_price: float, current_price: float, 
                           current_stop: float, candles_held: int,
                           tighten_after: int = 20, side: str = "LONG") -> float:
        """
        Tighten stop-loss after holding for a certain number of candles.
        Moves stop to breakeven after `tighten_after` candles if in profit.
        """
        if candles_held >= tighten_after:
            if side == "LONG" and current_price > entry_price:
                # Move stop to breakeven
                new_stop = max(entry_price, current_stop)
                if new_stop > current_stop:
                    logger.info(f"Time-based stop tightened to breakeven: {new_stop:.2f}")
                return new_stop
            elif side == "SHORT" and current_price < entry_price:
                new_stop = min(entry_price, current_stop)
                if new_stop < current_stop:
                    logger.info(f"Time-based stop tightened to breakeven: {new_stop:.2f}")
                return new_stop
        
        return current_stop
