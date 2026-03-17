"""
Paper Trader for Quant Research Lab.

Simulates trading without real money:
    - Order execution simulation
    - Position tracking
    - P&L calculation
    - Slippage and commission modeling
    - Realistic fills

Paper trading is essential for:
    - Strategy testing without risk
    - System integration testing
    - Performance benchmarking
    - Debugging trading logic

Example usage:
    ```python
    from execution.paper_trader import PaperTrader, PaperTradingConfig

    config = PaperTradingConfig(
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0001
    )

    trader = PaperTrader(config)
    trader.update_market_price('BTCUSDT', 40000)

    order = trader.place_order(
        symbol='BTCUSDT',
        side='BUY',
        order_type='LIMIT',
        quantity=0.1,
        price=39500
    )
    ```
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from execution.exchange_gateway import (
    Order, Position, AccountInfo, Ticker,
    OrderSide, OrderType, OrderStatus, TimeInForce, PositionSide
)


@dataclass
class PaperTradingConfig:
    """
    Paper trading configuration.

    Attributes:
        initial_capital: Starting capital
        commission_rate: Trading commission rate
        slippage_rate: Base slippage rate
        market_impact: Market impact coefficient
        partial_fills: Allow partial fills
        min_fill_pct: Minimum fill percentage for partial fills
        latency_ms: Simulated latency in milliseconds
        price_source: Price source ('mid', 'bid', 'ask')
        spread_multiplier: Spread multiplier for fills
    """
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0001
    market_impact: float = 0.0
    partial_fills: bool = True
    min_fill_pct: float = 0.1
    latency_ms: int = 100
    price_source: str = 'mid'
    spread_multiplier: float = 1.0


@dataclass
class SimulatedPosition:
    """Simulated position."""
    symbol: str
    quantity: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    market_value: float = 0.0
    trades: List[Dict] = field(default_factory=list)


class PaperTrader:
    """
    Paper Trading Simulator.

    Simulates a trading exchange with:
        - Order placement and matching
        - Position tracking
        - P&L calculation
        - Realistic execution modeling

    Attributes:
        config: Paper trading configuration
    """

    def __init__(self, config: Optional[PaperTradingConfig] = None):
        """
        Initialize Paper Trader.

        Args:
            config: Paper trading configuration
        """
        self.config = config or PaperTradingConfig()
        self.logger = get_logger('paper_trader')

        # State
        self._capital = self.config.initial_capital
        self._initial_capital = self.config.initial_capital
        self._positions: Dict[str, SimulatedPosition] = {}
        self._orders: Dict[str, Order] = {}
        self._market_prices: Dict[str, float] = {}
        self._tickers: Dict[str, Ticker] = {}

        # Callbacks
        self._order_callbacks: List[Callable] = []
        self._position_callbacks: List[Callable] = []

        # Statistics
        self._total_trades = 0
        self._total_commission = 0.0
        self._total_slippage = 0.0
        self._winning_trades = 0
        self._losing_trades = 0

    async def connect(self) -> bool:
        """Connect (no-op for paper trading)."""
        self.logger.info("Paper trader initialized")
        return True

    async def disconnect(self) -> None:
        """Disconnect (no-op for paper trading)."""
        pass

    def update_market_price(
        self,
        symbol: str,
        price: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Update market price for a symbol.

        Args:
            symbol: Trading symbol
            price: Current price
            bid: Optional bid price
            ask: Optional ask price
            timestamp: Timestamp
        """
        self._market_prices[symbol] = price

        # Update ticker
        self._tickers[symbol] = Ticker(
            symbol=symbol,
            bid=bid or price * 0.9999,
            ask=ask or price * 1.0001,
            last=price,
            timestamp=timestamp or datetime.utcnow()
        )

        # Update positions
        if symbol in self._positions:
            self._update_position_pnl(symbol, price)

        # Check pending orders
        self._check_pending_orders(symbol, price)

    def place_order(
        self,
        symbol: str,
        side: Union[OrderSide, str],
        order_type: Union[OrderType, str],
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: Optional[TimeInForce] = None,
        position_side: Optional[PositionSide] = None,
        **kwargs
    ) -> Order:
        """
        Place an order.

        Args:
            symbol: Trading symbol
            side: Order side
            order_type: Order type
            quantity: Order quantity
            price: Limit price
            stop_price: Stop price
            time_in_force: Time in force
            position_side: Position side
            **kwargs: Additional parameters

        Returns:
            Order object
        """
        # Normalize inputs
        if isinstance(side, str):
            side = OrderSide(side.upper())
        if isinstance(order_type, str):
            order_type = OrderType(order_type.upper())

        # Create order
        order_id = str(uuid.uuid4())[:8]
        client_order_id = f"paper_{order_id}"

        order = Order(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            status=OrderStatus.NEW,
            quantity=quantity,
            price=price or 0,
            stop_price=stop_price or 0,
            time_in_force=time_in_force or TimeInForce.GTC,
            create_time=datetime.utcnow(),
            position_side=position_side or PositionSide.BOTH
        )

        self._orders[order_id] = order

        # Simulate latency
        if self.config.latency_ms > 0:
            time.sleep(self.config.latency_ms / 1000)

        # Try to fill immediately for market orders
        if order_type == OrderType.MARKET:
            self._fill_market_order(order)
        elif stop_price:
            # Stop order - keep pending until triggered
            self.logger.info(f"Stop order {order_id} pending at {stop_price}")
        elif order_type == OrderType.LIMIT:
            # Check if can fill immediately
            self._check_limit_order_fill(order)

        return order

    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None
    ) -> bool:
        """
        Cancel an order.

        Args:
            symbol: Trading symbol
            order_id: Order ID
            client_order_id: Client order ID

        Returns:
            True if cancelled
        """
        if order_id and order_id in self._orders:
            order = self._orders[order_id]
            if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                order.status = OrderStatus.CANCELED
                order.update_time = datetime.utcnow()
                self.logger.info(f"Order {order_id} cancelled")
                return True

        return False

    def get_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None
    ) -> Optional[Order]:
        """Get order by ID."""
        if order_id:
            return self._orders.get(order_id)
        return None

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders."""
        orders = [
            o for o in self._orders.values()
            if o.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]
        ]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_positions(self) -> List[Position]:
        """Get all positions."""
        positions = []
        for sym, pos in self._positions.items():
            if pos.quantity != 0:
                price = self._market_prices.get(sym, 0)
                positions.append(Position(
                    symbol=sym,
                    quantity=pos.quantity,
                    entry_price=pos.entry_price,
                    mark_price=price,
                    unrealized_pnl=pos.unrealized_pnl,
                    update_time=datetime.utcnow()
                ))
        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        if symbol in self._positions:
            pos = self._positions[symbol]
            if pos.quantity != 0:
                price = self._market_prices.get(symbol, 0)
                return Position(
                    symbol=symbol,
                    quantity=pos.quantity,
                    entry_price=pos.entry_price,
                    mark_price=price,
                    unrealized_pnl=pos.unrealized_pnl,
                    update_time=datetime.utcnow()
                )
        return None

    def get_account_info(self) -> AccountInfo:
        """Get account information."""
        # Calculate total equity
        total_equity = self._capital
        unrealized_pnl = 0.0

        for sym, pos in self._positions.items():
            if pos.quantity != 0:
                unrealized_pnl += pos.unrealized_pnl
                total_equity += pos.market_value

        return AccountInfo(
            total_balance=total_equity,
            available_balance=self._capital,
            unrealized_pnl=unrealized_pnl,
            margin_balance=total_equity,
            positions=self.get_positions(),
            update_time=datetime.utcnow()
        )

    def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """Get ticker for a symbol."""
        return self._tickers.get(symbol)

    def get_statistics(self) -> Dict[str, Any]:
        """Get trading statistics."""
        total_equity = self._capital
        unrealized_pnl = 0.0

        for pos in self._positions.values():
            if pos.quantity != 0:
                unrealized_pnl += pos.unrealized_pnl
                total_equity += pos.market_value

        realized_pnl = total_equity - self._initial_capital - unrealized_pnl

        return {
            'initial_capital': self._initial_capital,
            'current_capital': self._capital,
            'total_equity': total_equity,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_return': (total_equity - self._initial_capital) / self._initial_capital,
            'total_trades': self._total_trades,
            'winning_trades': self._winning_trades,
            'losing_trades': self._losing_trades,
            'win_rate': self._winning_trades / self._total_trades if self._total_trades > 0 else 0,
            'total_commission': self._total_commission,
            'total_slippage': self._total_slippage
        }

    def reset(self) -> None:
        """Reset paper trader to initial state."""
        self._capital = self.config.initial_capital
        self._positions.clear()
        self._orders.clear()
        self._total_trades = 0
        self._total_commission = 0.0
        self._total_slippage = 0.0
        self._winning_trades = 0
        self._losing_trades = 0
        self.logger.info("Paper trader reset")

    def on_order_update(self, callback: Callable) -> None:
        """Register order update callback."""
        self._order_callbacks.append(callback)

    def on_position_update(self, callback: Callable) -> None:
        """Register position update callback."""
        self._position_callbacks.append(callback)

    # Private methods

    def _fill_market_order(self, order: Order) -> None:
        """Fill a market order."""
        symbol = order.symbol
        if symbol not in self._market_prices:
            order.status = OrderStatus.REJECTED
            return

        base_price = self._market_prices[symbol]

        # Calculate fill price with slippage
        slippage = self._calculate_slippage(order)
        if order.side == OrderSide.BUY:
            fill_price = base_price * (1 + slippage)
        else:
            fill_price = base_price * (1 - slippage)

        self._execute_order(order, fill_price, order.quantity)

    def _check_limit_order_fill(self, order: Order) -> None:
        """Check if limit order can be filled."""
        symbol = order.symbol
        if symbol not in self._tickers:
            return

        ticker = self._tickers[symbol]

        if order.side == OrderSide.BUY:
            # Buy limit fills when ask <= limit price
            if ticker.ask <= order.price:
                self._execute_order(order, min(ticker.ask, order.price), order.quantity)
        else:
            # Sell limit fills when bid >= limit price
            if ticker.bid >= order.price:
                self._execute_order(order, max(ticker.bid, order.price), order.quantity)

    def _check_pending_orders(self, symbol: str, price: float) -> None:
        """Check pending orders for fills."""
        for order in self._orders.values():
            if order.symbol != symbol or order.status != OrderStatus.NEW:
                continue

            # Check stop orders
            if order.stop_price > 0:
                if order.side == OrderSide.BUY and price >= order.stop_price:
                    # Stop buy triggered
                    self._execute_order(order, price, order.quantity)
                elif order.side == OrderSide.SELL and price <= order.stop_price:
                    # Stop sell triggered
                    self._execute_order(order, price, order.quantity)

            # Check limit orders
            elif order.order_type == OrderType.LIMIT:
                self._check_limit_order_fill(order)

    def _execute_order(self, order: Order, fill_price: float, fill_qty: float) -> None:
        """Execute an order fill."""
        symbol = order.symbol

        # Calculate commission
        commission = fill_price * fill_qty * self.config.commission_rate

        # Update capital
        if order.side == OrderSide.BUY:
            cost = fill_price * fill_qty + commission
            self._capital -= cost
        else:
            revenue = fill_price * fill_qty - commission
            self._capital += revenue

        # Update position
        if symbol not in self._positions:
            self._positions[symbol] = SimulatedPosition(symbol=symbol)

        pos = self._positions[symbol]
        trade_pnl = 0.0

        if order.side == OrderSide.BUY:
            # Adding to position
            if pos.quantity >= 0:
                # Long position - add
                total_cost = pos.entry_price * pos.quantity + fill_price * fill_qty
                pos.quantity += fill_qty
                pos.entry_price = total_cost / pos.quantity if pos.quantity > 0 else 0
            else:
                # Short position - closing
                if fill_qty <= abs(pos.quantity):
                    trade_pnl = (pos.entry_price - fill_price) * fill_qty - commission
                    pos.quantity += fill_qty
                else:
                    # Flip to long
                    close_qty = abs(pos.quantity)
                    trade_pnl = (pos.entry_price - fill_price) * close_qty
                    remaining = fill_qty - close_qty
                    pos.quantity = remaining
                    pos.entry_price = fill_price
        else:
            # Selling
            if pos.quantity <= 0:
                # Short position - add
                total_cost = pos.entry_price * abs(pos.quantity) + fill_price * fill_qty
                pos.quantity -= fill_qty
                pos.entry_price = total_cost / abs(pos.quantity) if pos.quantity < 0 else 0
            else:
                # Long position - closing
                if fill_qty <= pos.quantity:
                    trade_pnl = (fill_price - pos.entry_price) * fill_qty - commission
                    pos.quantity -= fill_qty
                else:
                    # Flip to short
                    close_qty = pos.quantity
                    trade_pnl = (fill_price - pos.entry_price) * close_qty
                    remaining = fill_qty - close_qty
                    pos.quantity = -remaining
                    pos.entry_price = fill_price

        # Update order
        order.filled_quantity += fill_qty
        order.avg_price = fill_price
        order.commission += commission
        order.status = OrderStatus.FILLED if order.filled_quantity >= order.quantity else OrderStatus.PARTIALLY_FILLED
        order.update_time = datetime.utcnow()

        # Update statistics
        self._total_trades += 1
        self._total_commission += commission
        if trade_pnl > 0:
            self._winning_trades += 1
        elif trade_pnl < 0:
            self._losing_trades += 1

        # Record trade
        pos.trades.append({
            'timestamp': datetime.utcnow(),
            'side': order.side.value,
            'quantity': fill_qty,
            'price': fill_price,
            'commission': commission,
            'pnl': trade_pnl
        })

        # Update position P&L
        current_price = self._market_prices.get(symbol, fill_price)
        self._update_position_pnl(symbol, current_price)

        # Notify callbacks
        for callback in self._order_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"Order callback error: {e}")

        self.logger.info(
            f"Order {order.order_id} filled: {order.side.value} {fill_qty} {symbol} @ {fill_price:.4f}"
        )

    def _calculate_slippage(self, order: Order) -> float:
        """Calculate slippage for an order."""
        base_slippage = self.config.slippage_rate

        # Add market impact
        if self.config.market_impact > 0:
            # Simple linear impact model
            market_price = self._market_prices.get(order.symbol, 1)
            order_value = order.quantity * market_price
            impact = self.config.market_impact * (order_value / 100000)
            base_slippage += impact

        return base_slippage

    def _update_position_pnl(self, symbol: str, price: float) -> None:
        """Update position P&L."""
        if symbol not in self._positions:
            return

        pos = self._positions[symbol]
        if pos.quantity == 0:
            pos.unrealized_pnl = 0
            pos.market_value = 0
            return

        pos.market_value = pos.quantity * price

        if pos.quantity > 0:
            pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
        else:
            pos.unrealized_pnl = (pos.entry_price - price) * abs(pos.quantity)

        # Notify callbacks
        for callback in self._position_callbacks:
            try:
                callback(Position(
                    symbol=symbol,
                    quantity=pos.quantity,
                    entry_price=pos.entry_price,
                    mark_price=price,
                    unrealized_pnl=pos.unrealized_pnl
                ))
            except Exception as e:
                self.logger.error(f"Position callback error: {e}")


__all__ = ['PaperTradingConfig', 'PaperTrader']
