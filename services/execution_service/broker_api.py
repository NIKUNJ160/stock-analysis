from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid
import time
import os

try:
    from breeze_connect import BreezeConnect
except ImportError:
    BreezeConnect = None

from src.utils.logger import get_logger

logger = get_logger("Execution.BrokerAPI")


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None          # For LIMIT orders
    stop_price: Optional[float] = None     # For STOP orders
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0
    filled_price: float = 0
    created_at: str = ""
    filled_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "created_at": self.created_at,
            "filled_at": self.filled_at,
        }


class BrokerAPI(ABC):
    """Abstract base class for broker integrations."""
    
    @abstractmethod
    def place_order(self, order: Order) -> Order:
        """Submit an order to the broker."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current status of an order."""
        pass
    
    @abstractmethod
    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        pass
    
    @abstractmethod
    def get_balance(self) -> dict:
        """Get account balance."""
        pass


class PaperTradingBroker(BrokerAPI):
    """
    Paper trading broker for simulated execution.
    Fills all MARKET orders instantly at current price.
    Tracks positions and P&L in-memory.
    """
    
    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, dict] = {}  # symbol -> {qty, avg_price, side}
        self.orders: dict[str, Order] = {}
        self.trade_log: list[dict] = []
        self.current_prices: dict[str, float] = {}
    
    def set_current_price(self, symbol: str, price: float):
        """Update the latest price for a symbol (used for paper fills)."""
        self.current_prices[symbol] = price
    
    def place_order(self, order: Order) -> Order:
        """Instantly fill MARKET orders; queue LIMIT/STOP orders."""
        self.orders[order.order_id] = order
        
        if order.order_type == OrderType.MARKET:
            fill_price = self.current_prices.get(order.symbol, order.price or 0)
            if fill_price <= 0:
                order.status = OrderStatus.REJECTED
                logger.error(f"Order rejected: no price available for {order.symbol}")
                return order
            
            self._fill_order(order, fill_price)
        else:
            order.status = OrderStatus.SUBMITTED
            logger.info(f"Order submitted: {order.order_id} (LIMIT/STOP)")
        
        return order
    
    def _fill_order(self, order: Order, fill_price: float):
        """Execute a fill on paper."""
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now().isoformat()
        
        cost = fill_price * order.quantity
        
        if order.side == OrderSide.BUY:
            if order.symbol in self.positions and self.positions[order.symbol]['side'] == 'SHORT':
                # Covering a short
                pos = self.positions[order.symbol]
                cover_qty = min(order.quantity, pos['qty'])
                pnl = (pos['avg_price'] - fill_price) * cover_qty
                
                self.cash -= fill_price * cover_qty
                pos['qty'] -= cover_qty
                
                self.trade_log.append({
                    "symbol": order.symbol,
                    "side": "BUY_TO_COVER",
                    "price": fill_price,
                    "quantity": cover_qty,
                    "pnl": pnl,
                    "timestamp": order.filled_at,
                })
                
                if pos['qty'] <= 0:
                    del self.positions[order.symbol]
                
                # If there's remaining qty, it opens a long position
                remaining = order.quantity - cover_qty
                if remaining > 0:
                    cost_rem = fill_price * remaining
                    if cost_rem > self.cash:
                        order.status = OrderStatus.PARTIAL_FILL
                        order.filled_quantity = cover_qty
                        logger.warning(f"Partial fill/insufficient funds for remainder {order.symbol}")
                        return
                    self.cash -= cost_rem
                    self.positions[order.symbol] = {
                        'qty': remaining,
                        'avg_price': fill_price,
                        'side': 'LONG',
                    }
            else:
                # Opening/adding to LONG
                if cost > self.cash:
                    order.status = OrderStatus.REJECTED
                    logger.warning(f"Insufficient funds for {order.symbol}: need {cost:.2f}, have {self.cash:.2f}")
                    return
                
                self.cash -= cost
                
                if order.symbol in self.positions:
                    pos = self.positions[order.symbol]
                    total_qty = pos['qty'] + order.quantity
                    pos['avg_price'] = (pos['avg_price'] * pos['qty'] + fill_price * order.quantity) / total_qty
                    pos['qty'] = total_qty
                else:
                    self.positions[order.symbol] = {
                        'qty': order.quantity,
                        'avg_price': fill_price,
                        'side': 'LONG',
                    }
        else:  # SELL
            if order.symbol in self.positions and self.positions[order.symbol]['side'] == 'LONG':
                # Selling a long position
                pos = self.positions[order.symbol]
                sell_qty = min(order.quantity, pos['qty'])
                pnl = (fill_price - pos['avg_price']) * sell_qty
                
                self.cash += fill_price * sell_qty
                pos['qty'] -= sell_qty
                
                self.trade_log.append({
                    "symbol": order.symbol,
                    "side": "SELL",
                    "price": fill_price,
                    "quantity": sell_qty,
                    "pnl": pnl,
                    "timestamp": order.filled_at,
                })
                
                if pos['qty'] <= 0:
                    del self.positions[order.symbol]
                
                # If remaining qty, open short position
                remaining = order.quantity - sell_qty
                if remaining > 0:
                    self.cash += fill_price * remaining # Short sale proceeds
                    self.positions[order.symbol] = {
                        'qty': remaining,
                        'avg_price': fill_price,
                        'side': 'SHORT',
                    }
            else:
                # Opening/adding to SHORT
                self.cash += cost
                if order.symbol in self.positions:
                    pos = self.positions[order.symbol]
                    total_qty = pos['qty'] + order.quantity
                    pos['avg_price'] = (pos['avg_price'] * pos['qty'] + fill_price * order.quantity) / total_qty
                    pos['qty'] = total_qty
                else:
                    self.positions[order.symbol] = {
                        'qty': order.quantity,
                        'avg_price': fill_price,
                        'side': 'SHORT',
                    }
        
        logger.info(f"Order FILLED: {order.side.value} {order.quantity} {order.symbol} @ {fill_price:.2f}")
    
    def check_pending_orders(self):
        """Check and fill any pending LIMIT/STOP orders."""
        for oid, order in list(self.orders.items()):
            if order.status != OrderStatus.SUBMITTED:
                continue
            
            price = self.current_prices.get(order.symbol)
            if price is None:
                continue
            
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and price <= order.price:
                    self._fill_order(order, price)
                elif order.side == OrderSide.SELL and price >= order.price:
                    self._fill_order(order, price)
            elif order.order_type == OrderType.STOP_MARKET:
                if order.side == OrderSide.SELL and price <= order.stop_price:
                    self._fill_order(order, price)
                elif order.side == OrderSide.BUY and price >= order.stop_price:
                    self._fill_order(order, price)
            elif order.order_type == OrderType.STOP_LIMIT:
                if order.side == OrderSide.SELL and price <= order.stop_price and price >= (order.price or 0):
                    self._fill_order(order, price)
                elif order.side == OrderSide.BUY and price >= order.stop_price and price <= (order.price or float('inf')):
                    self._fill_order(order, price)
    
    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                order.status = OrderStatus.CANCELLED
                return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)
    
    def get_positions(self) -> list[dict]:
        result = []
        for symbol, pos in self.positions.items():
            current_price = self.current_prices.get(symbol, pos['avg_price'])
            unrealized_pnl = (current_price - pos['avg_price']) * pos['qty']
            if pos['side'] == 'SHORT':
                unrealized_pnl = -unrealized_pnl
            result.append({
                "symbol": symbol,
                "qty": pos['qty'],
                "avg_price": pos['avg_price'],
                "side": pos['side'],
                "current_price": current_price,
                "unrealized_pnl": unrealized_pnl,
            })
        return result
    
    def get_balance(self) -> dict:
        equity = self.cash
        for pos_info in self.get_positions():
            if pos_info['side'] == 'LONG':
                equity += (pos_info['current_price'] * pos_info['qty'])
            else:
                equity -= (pos_info['current_price'] * pos_info['qty'])
        
        return {
            "cash": self.cash,
            "equity": equity,
            "initial_capital": self.initial_capital,
            "total_return": (equity - self.initial_capital) / self.initial_capital,
            "open_positions": len(self.positions),
        }


class ICICIBroker(BrokerAPI):
    """
    Live broker integration for ICICI Direct using breeze-connect.
    """
    
    def __init__(self, api_key: str, api_secret: str, session_token: str):
        if BreezeConnect is None:
            raise ImportError("breeze-connect is not installed.")
        
        self.breeze = BreezeConnect(api_key=api_key)
        self.breeze.generate_session(api_secret=api_secret, session_token=session_token)
        logger.info("ICICIBroker session successfully initialized.")
        
        self.orders: dict[str, Order] = {}

    def _format_symbol(self, symbol: str) -> tuple[str, str]:
        """Convert standard format 'RELIANCE.NS' to breeze format ('RELIANCE', 'NSE')."""
        parts = symbol.split('.')
        exchange = "NSE"
        if len(parts) > 1 and parts[1].upper() == "BO":
            exchange = "BSE"
        return parts[0], exchange

    def place_order(self, order: Order) -> Order:
        stock_code, exchange_code = self._format_symbol(order.symbol)
        
        action = "buy" if order.side == OrderSide.BUY else "sell"
        order_type_map = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_MARKET: "stoploss",
            OrderType.STOP_LIMIT: "stoploss" 
        }
        
        try:
            resp = self.breeze.place_order(
                stock_code=stock_code,
                exchange_code=exchange_code,
                product="cash",  # Intraday cash. Alternative: 'margin', 'delivery'
                action=action,
                order_type=order_type_map[order.order_type],
                stoploss=str(order.stop_price) if order.stop_price else "0",
                quantity=str(int(order.quantity)),
                price=str(order.price) if order.price else "0",
                validity="day"
            )
            
            if resp.get('Status') == 200:
                broker_order_id = resp.get('Success', {}).get('order_id', order.order_id)
                order.order_id = str(broker_order_id)
                order.status = OrderStatus.SUBMITTED
                self.orders[order.order_id] = order
                logger.info(f"ICICIBroker order placed: {order.order_id}")
            else:
                order.status = OrderStatus.REJECTED
                logger.error(f"ICICIBroker placement failed: {resp.get('Error')}")
                
        except Exception as e:
            logger.error(f"ICICIBroker exception during place_order: {e}")
            order.status = OrderStatus.REJECTED
            
        return order

    def cancel_order(self, order_id: str) -> bool:
        if order_id not in self.orders:
            return False
            
        order = self.orders[order_id]
        stock_code, exchange_code = self._format_symbol(order.symbol)
        
        try:
            resp = self.breeze.cancel_order(
                exchange_code=exchange_code,
                order_id=order_id
            )
            if resp.get('Status') == 200:
                order.status = OrderStatus.CANCELLED
                return True
            logger.error(f"Cancel failed: {resp.get('Error')}")
            return False
        except Exception as e:
            logger.error(f"Exception during cancel_order: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        if order_id not in self.orders:
            return None
            
        order = self.orders[order_id]
        stock_code, exchange_code = self._format_symbol(order.symbol)
        
        try:
            resp = self.breeze.get_trade_detail(
                exchange_code=exchange_code,
                order_id=order_id
            )
            if resp.get('Status') == 200 and resp.get('Success'):
                details = resp['Success'][0]
                api_status = details.get('status', '').lower()
                
                if api_status in ['executed', 'traded']:
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = float(details.get('executed_quantity', order.quantity))
                    order.filled_price = float(details.get('average_price', 0))
                elif api_status == 'cancelled':
                    order.status = OrderStatus.CANCELLED
                elif api_status == 'rejected':
                    order.status = OrderStatus.REJECTED
                elif api_status == 'partially executed':
                    order.status = OrderStatus.PARTIAL_FILL
                    
        except Exception as e:
            logger.error(f"Exception fetching status for {order_id}: {e}")
            
        return order

    def get_positions(self) -> list[dict]:
        try:
            resp = self.breeze.get_portfolio_positions()
            if resp.get('Status') == 200 and resp.get('Success'):
                positions = []
                for p in resp['Success']:
                    qty = int(p.get('quantity', 0))
                    if qty == 0:
                        continue
                    positions.append({
                        "symbol": f"{p.get('stock_code')}.NS", # Extrapolating NS for simplicity
                        "qty": abs(qty),
                        "avg_price": float(p.get('average_price', 0)),
                        "side": "LONG" if qty > 0 else "SHORT",
                        "current_price": float(p.get('ltp', 0)),
                        "unrealized_pnl": float(p.get('unrealized_pnl', 0)),
                    })
                return positions
            return []
        except Exception as e:
            logger.error(f"Exception fetching positions: {e}")
            return []

    def get_balance(self) -> dict:
        try:
            resp = self.breeze.get_funds()
            if resp.get('Status') == 200 and resp.get('Success'):
                funds = resp['Success']
                available_margin = float(funds.get('available_margin', 0))
                return {
                    "cash": available_margin,
                    "equity": available_margin, 
                    "initial_capital": available_margin, 
                    "total_return": 0.0,
                    "open_positions": len(self.get_positions()),
                }
            return {"cash": 0, "equity": 0}
        except Exception as e:
            logger.error(f"Exception fetching balance: {e}")
            return {"cash": 0, "equity": 0}


def create_order(symbol: str, side: str, quantity: float, order_type: str = "MARKET",
                 price: float = None, stop_price: float = None) -> Order:
    """Helper to create an Order object."""
    return Order(
        order_id=str(uuid.uuid4())[:8],
        symbol=symbol,
        side=OrderSide(side),
        order_type=OrderType(order_type),
        quantity=quantity,
        price=price,
        stop_price=stop_price,
    )
