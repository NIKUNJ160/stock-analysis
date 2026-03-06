import asyncio
from typing import Optional

from services.execution_service.broker_api import (
    PaperTradingBroker, Order, create_order, OrderSide, OrderType
)
from src.signal_engine.signal_generator import TradingSignal, SignalStrength
from src.risk_management.risk_manager import RiskManager
from src.risk_management.stoploss import StopLossEngine
from src.risk_management.position_sizing import PositionSizer
from src.utils.logger import get_logger

logger = get_logger("Execution.OrderManager")


class OrderManager:
    """
    Central order management system. 
    
    Coordinates between Signal Engine → Risk Engine → Broker API.
    Handles the full order lifecycle:
        Signal → Validate Risk → Size Position → Calculate SL/TP → Place Order → Track
    """
    
    def __init__(self, broker: PaperTradingBroker, risk_manager: RiskManager,
                 capital: float = 1_000_000):
        self.broker = broker
        self.risk_manager = risk_manager
        self.capital = capital
        self.sizer = PositionSizer()
        self.sl_engine = StopLossEngine()
        self.order_history: list[dict] = []
    
    async def process_signal(self, signal: TradingSignal, current_price: float,
                             atr_value: float = 0) -> Optional[Order]:
        """
        Process a trading signal through the full lifecycle.
        
        Returns the placed Order if approved, None if rejected.
        """
        symbol = signal.symbol
        
        # 1. Determine side
        if signal.strength in [SignalStrength.STRONG_BUY, SignalStrength.BUY]:
            side = "LONG"
        elif signal.strength in [SignalStrength.STRONG_SELL, SignalStrength.SELL]:
            side = "SHORT"
        else:
            logger.debug(f"Signal is HOLD for {symbol}, no trade")
            return None
        
        # 2. Calculate stop-loss and take-profit
        if atr_value > 0:
            stop_loss, take_profit = self.sl_engine.atr_stop(
                current_price, atr_value, multiplier=1.5, side=side
            )
        else:
            stop_loss, take_profit = self.sl_engine.percentage_stop(
                current_price, stop_pct=0.02, tp_pct=0.04, side=side
            )
        
        # 3. Calculate position size
        position_size = self.sizer.fixed_fraction(
            capital=self.capital,
            risk_fraction=0.02,
            entry_price=current_price,
            stop_loss=stop_loss,
        )
        
        if position_size <= 0:
            logger.warning(f"Position size is 0 for {symbol}")
            return None
        
        # 4. Validate through risk manager
        approved, reason, adjusted_size = self.risk_manager.validate_trade(
            signal, position_size, current_price, stop_loss
        )
        
        if not approved or adjusted_size <= 0:
            logger.warning(f"Trade REJECTED for {symbol}: {reason} (Size: {adjusted_size})")
            return None
        
        # 5. Place the order
        self.broker.set_current_price(symbol, current_price)
        
        order_side = "BUY" if side == "LONG" else "SELL"
        order = create_order(
            symbol=symbol,
            side=order_side,
            quantity=adjusted_size,
            order_type="MARKET",
        )
        
        filled_order = self.broker.place_order(order)
        
        # 6. Update risk manager with new position
        if filled_order.status in [OrderStatus.FILLED, OrderStatus.PARTIAL_FILL]:
            self.risk_manager.open_position(
                symbol=symbol,
                side=side,
                entry_price=filled_order.filled_price,
                quantity=adjusted_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            
            self.order_history.append(filled_order.to_dict())
            logger.info(f"Order executed: {order_side} {adjusted_size} {symbol} @ {filled_order.filled_price:.2f}")
        
        self.risk_manager.sync_broker_positions(self.broker.get_positions())
        
        return filled_order
    
    async def check_exits(self, current_prices: dict[str, float]) -> list[str]:
        """
        Check all open positions for stop-loss / take-profit triggers.
        Returns list of symbols that were closed.
        """
        # Update broker prices
        for symbol, price in current_prices.items():
            self.broker.set_current_price(symbol, price)
        
        # Check risk manager stops
        self.risk_manager.update_unrealized_pnl(current_prices)
        symbols_to_close = self.risk_manager.check_stop_losses(current_prices)
        
        closed = []
        for symbol in symbols_to_close:
            price = current_prices.get(symbol, 0)
            if price > 0:
                # Place exit order
                pos = self.risk_manager.positions.get(symbol)
                if pos:
                    exit_side = "SELL" if pos.side == "LONG" else "BUY"
                    order = create_order(symbol=symbol, side=exit_side, quantity=pos.quantity)
                    filled_order = self.broker.place_order(order)
                    if filled_order.status in [OrderStatus.FILLED, OrderStatus.PARTIAL_FILL]:
                        self.risk_manager.close_position(symbol, price)
                        closed.append(symbol)
                    else:
                        logger.warning(f"Exit order for {symbol} not immediately filled")
        
        # Also check broker pending orders
        self.broker.check_pending_orders()
        
        # Run sync to clear out completely closed items and update quantities
        self.risk_manager.sync_broker_positions(self.broker.get_positions())
        
        return closed
    
    def get_order_book(self) -> list[dict]:
        """Get all orders."""
        return [o.to_dict() for o in self.broker.orders.values()]
    
    def get_trade_log(self) -> list[dict]:
        """Get broker trade log."""
        return self.broker.trade_log
