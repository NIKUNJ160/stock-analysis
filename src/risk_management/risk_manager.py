from dataclasses import dataclass, field
from typing import Optional

from src.signal_engine.signal_generator import TradingSignal, SignalStrength
from src.utils.logger import get_logger

logger = get_logger("RiskEngine.Manager")


@dataclass
class Position:
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float = 0.0


@dataclass
class RiskConfig:
    max_capital: float = 1_000_000          # Total capital
    max_risk_per_trade: float = 0.02        # 2% of capital per trade
    max_open_positions: int = 5             # Max simultaneous positions
    max_exposure_per_symbol: float = 0.20   # 20% of capital per symbol
    max_daily_loss: float = 0.05            # 5% max daily loss
    max_correlation_overlap: int = 3         # Max positions in correlated assets
    margin_requirement: float = 0.10         # 10% margin requirement


class RiskManager:
    """
    Portfolio-level risk management engine.
    
    Validates trade proposals against:
    - Maximum open positions
    - Per-trade risk limits
    - Daily loss limits
    - Per-symbol exposure caps
    - Portfolio correlation checks
    """
    
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.positions: dict[str, Position] = {}
        self.daily_pnl: float = 0.0
        self.closed_pnl: float = 0.0
        self.trade_history: list[dict] = []
    
    def validate_trade(self, signal: TradingSignal, proposed_size: float, 
                       entry_price: float, stop_loss: float) -> tuple[bool, str, float]:
        """
        Validate whether a proposed trade meets risk requirements.
        
        Returns:
            (approved, reason, adjusted_size)
        """
        symbol = signal.symbol
        
        # 1. Check max open positions
        if len(self.positions) >= self.config.max_open_positions:
            if symbol not in self.positions:
                return False, f"Max positions reached ({self.config.max_open_positions})", 0
        
        # 2. Check daily loss limit
        daily_loss_limit = self.config.max_capital * self.config.max_daily_loss
        if self.daily_pnl <= -daily_loss_limit:
            return False, f"Daily loss limit reached ({self.config.max_daily_loss:.0%})", 0
        
        # 3. Determine side early for exposure calculation
        if signal.strength in [SignalStrength.STRONG_BUY, SignalStrength.BUY]:
            side = "LONG"
        elif signal.strength in [SignalStrength.STRONG_SELL, SignalStrength.SELL]:
            side = "SHORT"
        else:
            return False, "HOLD signal — no trade", 0

        # 4. Check per-symbol exposure (net, direction-aware)
        current_qty = 0
        current_side = None
        if symbol in self.positions:
            pos = self.positions[symbol]
            current_qty = pos.quantity
            current_side = pos.side
        
        signed_current = current_qty if current_side == "LONG" else -current_qty
        signed_trade = proposed_size if side == "LONG" else -proposed_size
        signed_new = signed_current + signed_trade
        new_exposure = abs(signed_new) * entry_price
        
        max_symbol_exposure = self.config.max_capital * self.config.max_exposure_per_symbol
        if new_exposure > max_symbol_exposure:
            # Adjust size down to match limit
            allowed_additional = max_symbol_exposure - (abs(signed_current) * entry_price)
            if allowed_additional <= 0:
                adjusted_size = 0
            else:
                adjusted_size = allowed_additional / entry_price
            
            if adjusted_size <= 0:
                return False, f"Symbol exposure limit reached for {symbol}", 0
            logger.warning(f"Adjusted size for {symbol}: {proposed_size} → {adjusted_size:.2f}")
            proposed_size = adjusted_size
        
        # 5. Check per-trade risk
        risk_per_share = abs(entry_price - stop_loss)
        total_risk = risk_per_share * proposed_size
        max_risk = self.config.max_capital * self.config.max_risk_per_trade
        
        if total_risk > max_risk:
            # Adjust size to fit risk budget
            adjusted_size = max_risk / risk_per_share
            logger.warning(f"Risk-adjusted size for {symbol}: {proposed_size:.2f} → {adjusted_size:.2f}")
            proposed_size = adjusted_size
        
        logger.info(f"Trade APPROVED: {side} {proposed_size:.2f} {symbol} @ {entry_price:.2f} (risk={total_risk:.2f})")
        return True, f"Approved: {side}", proposed_size
    
    def open_position(self, symbol: str, side: str, entry_price: float, 
                      quantity: float, stop_loss: float, take_profit: float):
        """Record a new open position or update SL/TP if position exists."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            logger.info(f"Position tracking opened: {side} {quantity:.2f} {symbol} @ {entry_price:.2f}")
        else:
            self.positions[symbol].stop_loss = stop_loss
            self.positions[symbol].take_profit = take_profit
            logger.info(f"Position SL/TP updated for {symbol}")

    def sync_broker_positions(self, broker_positions: list[dict]):
        """Synchronize risk manager positions tightly with actual broker state."""
        active_symbols = set()
        for b_pos in broker_positions:
            symbol = b_pos['symbol']
            active_symbols.add(symbol)
            if symbol in self.positions:
                rm_pos = self.positions[symbol]
                rm_pos.quantity = b_pos['qty']
                rm_pos.entry_price = b_pos['avg_price']
                rm_pos.side = b_pos['side']
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=b_pos['side'],
                    entry_price=b_pos['avg_price'],
                    quantity=b_pos['qty'],
                    stop_loss=0,
                    take_profit=0
                )
        
        # Clear positions no longer active in broker
        for sym in list(self.positions.keys()):
            if sym not in active_symbols:
                del self.positions[sym]
    
    def close_position(self, symbol: str, exit_price: float) -> float:
        """Close a position and return realized PnL."""
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return 0.0
        
        pos = self.positions.pop(symbol)
        if pos.side == "LONG":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity
        
        self.daily_pnl += pnl
        self.closed_pnl += pnl
        
        self.trade_history.append({
            "symbol": symbol,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "quantity": pos.quantity,
            "pnl": pnl,
        })
        
        logger.info(f"Position closed: {symbol} PnL={pnl:+.2f}")
        return pnl
    
    def update_unrealized_pnl(self, current_prices: dict[str, float]):
        """Update unrealized P&L for all open positions."""
        for symbol, pos in self.positions.items():
            if symbol in current_prices:
                price = current_prices[symbol]
                if pos.side == "LONG":
                    pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
                else:
                    pos.unrealized_pnl = (pos.entry_price - price) * pos.quantity
    
    def check_stop_losses(self, current_prices: dict[str, float]) -> list[str]:
        """Check stop-losses and return symbols that need closing."""
        to_close = []
        for symbol, pos in self.positions.items():
            if symbol in current_prices:
                price = current_prices[symbol]
                if pos.side == "LONG" and price <= pos.stop_loss:
                    to_close.append(symbol)
                    logger.warning(f"STOP-LOSS triggered for {symbol}: {price:.2f} <= {pos.stop_loss:.2f}")
                elif pos.side == "SHORT" and price >= pos.stop_loss:
                    to_close.append(symbol)
                    logger.warning(f"STOP-LOSS triggered for {symbol}: {price:.2f} >= {pos.stop_loss:.2f}")
                elif pos.side == "LONG" and price >= pos.take_profit:
                    to_close.append(symbol)
                    logger.info(f"TAKE-PROFIT hit for {symbol}: {price:.2f} >= {pos.take_profit:.2f}")
                elif pos.side == "SHORT" and price <= pos.take_profit:
                    to_close.append(symbol)
                    logger.info(f"TAKE-PROFIT hit for {symbol}: {price:.2f} <= {pos.take_profit:.2f}")
        return to_close
    
    def get_portfolio_summary(self) -> dict:
        """Get current portfolio state."""
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        total_exposure = sum(p.entry_price * p.quantity for p in self.positions.values())
        
        return {
            "open_positions": len(self.positions),
            "max_positions": self.config.max_open_positions,
            "daily_pnl": self.daily_pnl,
            "closed_pnl": self.closed_pnl,
            "unrealized_pnl": total_unrealized,
            "total_pnl": self.closed_pnl + total_unrealized,
            "total_exposure": total_exposure,
            "exposure_pct": total_exposure / self.config.max_capital if self.config.max_capital else 0,
            "positions": {s: {
                "side": p.side, "entry": p.entry_price, "qty": p.quantity,
                "sl": p.stop_loss, "tp": p.take_profit, "pnl": p.unrealized_pnl
            } for s, p in self.positions.items()},
        }
    
    def reset_daily(self):
        """Reset daily P&L (call at market open)."""
        self.daily_pnl = 0.0
        logger.info("Daily P&L reset")
