"""
Execution Simulator for Quant Research Lab.

Provides realistic order execution simulation with:
    - Market, limit, and stop orders
    - Slippage modeling
    - Transaction costs
    - Order book simulation
    - Latency simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import uuid
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.math_utils import safe_divide


class OrderType(Enum):
    """Order types."""
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'
    TRAILING_STOP = 'trailing_stop'
    ICEBERG = 'iceberg'


class OrderSide(Enum):
    """Order side."""
    BUY = 'buy'
    SELL = 'sell'


class OrderStatus(Enum):
    """Order status."""
    PENDING = 'pending'
    OPEN = 'open'
    PARTIALLY_FILLED = 'partially_filled'
    FILLED = 'filled'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'
    EXPIRED = 'expired'


class TimeInForce(Enum):
    """Time in force."""
    GTC = 'good_till_cancelled'
    IOC = 'immediate_or_cancel'
    FOK = 'fill_or_kill'
    DAY = 'day'


@dataclass
class Order:
    """Order data structure."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit/stop orders
    stop_price: Optional[float] = None  # For stop orders
    time_in_force: TimeInForce = TimeInForce.GTC
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    parent_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def remaining_quantity(self) -> float:
        """Get remaining unfilled quantity."""
        return self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]

    @property
    def is_complete(self) -> bool:
        """Check if order is complete."""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'commission': self.commission,
            'slippage': self.slippage,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class Fill:
    """Fill/execution data structure."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float
    timestamp: datetime = field(default_factory=datetime.now)
    fill_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: Dict = field(default_factory=dict)

    @property
    def value(self) -> float:
        """Get fill value."""
        return self.quantity * self.price

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'fill_id': self.fill_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'commission': self.commission,
            'slippage': self.slippage,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class Position:
    """Position data structure."""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        """Check if no position."""
        return self.quantity == 0

    def update(self, fill: Fill) -> float:
        """
        Update position with fill.

        Args:
            fill: Fill to apply

        Returns:
            Realized P&L from the fill
        """
        realized_pnl = 0.0

        if fill.side == OrderSide.BUY:
            if self.is_short:
                # Closing short position
                close_qty = min(abs(self.quantity), fill.quantity)
                realized_pnl = close_qty * (self.avg_price - fill.price)
                self.realized_pnl += realized_pnl

                if fill.quantity >= abs(self.quantity):
                    # Fully closed
                    self.quantity = fill.quantity - abs(self.quantity)
                    self.avg_price = fill.price if self.quantity > 0 else 0
                else:
                    self.quantity += fill.quantity
            else:
                # Adding to long position
                new_qty = self.quantity + fill.quantity
                self.avg_price = (
                    (self.quantity * self.avg_price + fill.quantity * fill.price) / new_qty
                    if new_qty != 0 else 0
                )
                self.quantity = new_qty
        else:  # SELL
            if self.is_long:
                # Closing long position
                close_qty = min(self.quantity, fill.quantity)
                realized_pnl = close_qty * (fill.price - self.avg_price)
                self.realized_pnl += realized_pnl

                if fill.quantity >= self.quantity:
                    # Fully closed
                    self.quantity = -(fill.quantity - self.quantity)
                    self.avg_price = fill.price if self.quantity < 0 else 0
                else:
                    self.quantity -= fill.quantity
            else:
                # Adding to short position
                new_qty = self.quantity - fill.quantity
                self.avg_price = (
                    (abs(self.quantity) * self.avg_price + fill.quantity * fill.price) / abs(new_qty)
                    if new_qty != 0 else 0
                )
                self.quantity = new_qty

        return realized_pnl

    def mark_to_market(self, current_price: float) -> float:
        """
        Mark position to market.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L
        """
        self.unrealized_pnl = self.quantity * (current_price - self.avg_price)
        return self.unrealized_pnl


class SlippageModel:
    """
    Slippage model for realistic execution simulation.

    Models slippage based on:
        - Order size relative to volume
        - Market volatility
        - Bid-ask spread
        - Market impact
    """

    def __init__(
        self,
        base_slippage: float = 0.0001,
        volume_impact_coefficient: float = 0.1,
        volatility_impact_coefficient: float = 0.5,
        spread_impact_coefficient: float = 0.5,
        use_random_slippage: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Slippage Model.

        Args:
            base_slippage: Base slippage rate
            volume_impact_coefficient: Impact of order size on slippage
            volatility_impact_coefficient: Impact of volatility on slippage
            spread_impact_coefficient: Impact of spread on slippage
            use_random_slippage: Add randomness to slippage
            random_seed: Random seed for reproducibility
        """
        self.base_slippage = base_slippage
        self.volume_impact_coefficient = volume_impact_coefficient
        self.volatility_impact_coefficient = volatility_impact_coefficient
        self.spread_impact_coefficient = spread_impact_coefficient
        self.use_random_slippage = use_random_slippage

        if random_seed is not None:
            np.random.seed(random_seed)

    def calculate_slippage(
        self,
        order: Order,
        market_price: float,
        volume: float = 1000000,
        volatility: float = 0.02,
        spread: float = 0.0001,
        avg_daily_volume: float = 10000000
    ) -> float:
        """
        Calculate slippage for an order.

        Args:
            order: Order to calculate slippage for
            market_price: Current market price
            volume: Current market volume
            volatility: Current volatility (as decimal)
            spread: Current bid-ask spread (as decimal)
            avg_daily_volume: Average daily volume

        Returns:
            Slippage as decimal
        """
        # Base slippage
        slippage = self.base_slippage

        # Volume impact (larger orders = more slippage)
        order_value = order.quantity * market_price
        volume_ratio = order_value / volume
        volume_impact = self.volume_impact_coefficient * volume_ratio
        slippage += volume_impact

        # Market impact (based on participation rate)
        participation_rate = order_value / avg_daily_volume
        market_impact = 0.1 * np.sqrt(participation_rate)  # Square root model
        slippage += market_impact

        # Volatility impact
        volatility_impact = self.volatility_impact_coefficient * volatility
        slippage += volatility_impact

        # Spread impact
        spread_impact = self.spread_impact_coefficient * spread
        slippage += spread_impact

        # Add randomness
        if self.use_random_slippage:
            random_factor = np.random.uniform(0.8, 1.2)
            slippage *= random_factor

        return slippage


class CommissionModel:
    """
    Commission model for transaction costs.

    Supports:
        - Fixed commission
        - Per-share commission
        - Percentage-based commission
        - Minimum commission
    """

    def __init__(
        self,
        commission_type: str = 'percentage',
        commission_rate: float = 0.001,
        per_share_rate: float = 0.005,
        fixed_commission: float = 0.0,
        min_commission: float = 0.0,
        max_commission: Optional[float] = None
    ):
        """
        Initialize Commission Model.

        Args:
            commission_type: 'percentage', 'per_share', or 'fixed'
            commission_rate: Commission rate for percentage model
            per_share_rate: Commission per share
            fixed_commission: Fixed commission per trade
            min_commission: Minimum commission
            max_commission: Maximum commission
        """
        self.commission_type = commission_type
        self.commission_rate = commission_rate
        self.per_share_rate = per_share_rate
        self.fixed_commission = fixed_commission
        self.min_commission = min_commission
        self.max_commission = max_commission

    def calculate_commission(
        self,
        quantity: float,
        price: float,
        side: OrderSide
    ) -> float:
        """
        Calculate commission for a trade.

        Args:
            quantity: Trade quantity
            price: Trade price
            side: Trade side

        Returns:
            Commission amount
        """
        if self.commission_type == 'percentage':
            commission = quantity * price * self.commission_rate
        elif self.commission_type == 'per_share':
            commission = quantity * self.per_share_rate
        else:  # fixed
            commission = self.fixed_commission

        # Apply min/max
        commission = max(commission, self.min_commission)
        if self.max_commission is not None:
            commission = min(commission, self.max_commission)

        return commission


class ExecutionSimulator:
    """
    Order Execution Simulator.

    Simulates realistic order execution with:
        - Order book simulation
        - Slippage modeling
        - Commission calculation
        - Latency simulation
        - Partial fills

    Attributes:
        slippage_model: Slippage model
        commission_model: Commission model
        latency_ms: Simulated latency in milliseconds
    """

    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        commission_model: Optional[CommissionModel] = None,
        latency_ms: float = 100.0,
        enable_partial_fills: bool = True,
        partial_fill_probability: float = 0.1
    ):
        """
        Initialize Execution Simulator.

        Args:
            slippage_model: Slippage model instance
            commission_model: Commission model instance
            latency_ms: Simulated latency
            enable_partial_fills: Allow partial fills
            partial_fill_probability: Probability of partial fill
        """
        self.logger = get_logger('execution_simulator')
        self.slippage_model = slippage_model or SlippageModel()
        self.commission_model = commission_model or CommissionModel()
        self.latency_ms = latency_ms
        self.enable_partial_fills = enable_partial_fills
        self.partial_fill_probability = partial_fill_probability

        # Order management
        self._orders: Dict[str, Order] = {}
        self._fills: List[Fill] = []
        self._open_orders: Dict[str, List[Order]] = {}  # symbol -> orders

        # Market data cache
        self._last_prices: Dict[str, float] = {}
        self._last_volumes: Dict[str, float] = {}
        self._volatilities: Dict[str, float] = {}
        self._spreads: Dict[str, float] = {}

    def update_market(
        self,
        symbol: str,
        price: float,
        volume: float = 0,
        volatility: float = 0.02,
        spread: float = 0.0001,
        bid: Optional[float] = None,
        ask: Optional[float] = None
    ) -> None:
        """
        Update market data for a symbol.

        Args:
            symbol: Trading symbol
            price: Current price
            volume: Current volume
            volatility: Current volatility
            spread: Current spread
            bid: Best bid (optional)
            ask: Best ask (optional)
        """
        self._last_prices[symbol] = price
        self._last_volumes[symbol] = volume
        self._volatilities[symbol] = volatility
        self._spreads[symbol] = spread

        if bid is not None and ask is not None:
            self._spreads[symbol] = (ask - bid) / price

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        timestamp: Optional[datetime] = None
    ) -> Order:
        """
        Submit a new order.

        Args:
            symbol: Trading symbol
            side: Order side
            order_type: Order type
            quantity: Order quantity
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force
            timestamp: Order timestamp

        Returns:
            Created order
        """
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            timestamp=timestamp or datetime.now()
        )

        self._orders[order.order_id] = order

        # Track open orders
        if symbol not in self._open_orders:
            self._open_orders[symbol] = []
        self._open_orders[symbol].append(order)

        self.logger.debug(
            f"Submitted {order_type.value} {side.value} order: {quantity} {symbol} @ {price}"
        )

        return order

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]

        if not order.is_active:
            return False

        order.status = OrderStatus.CANCELLED

        # Remove from open orders
        if order.symbol in self._open_orders:
            self._open_orders[order.symbol] = [
                o for o in self._open_orders[order.symbol] if o.order_id != order_id
            ]

        self.logger.debug(f"Cancelled order {order_id}")
        return True

    def process_orders(
        self,
        current_time: datetime
    ) -> List[Fill]:
        """
        Process all open orders and generate fills.

        Args:
            current_time: Current simulation time

        Returns:
            List of fills generated
        """
        fills = []

        for symbol, orders in list(self._open_orders.items()):
            if symbol not in self._last_prices:
                continue

            current_price = self._last_prices[symbol]
            volume = self._last_volumes.get(symbol, 1000000)
            volatility = self._volatilities.get(symbol, 0.02)
            spread = self._spreads.get(symbol, 0.0001)

            for order in orders[:]:  # Copy list to allow modification
                if not order.is_active:
                    orders.remove(order)
                    continue

                # Check if order can be filled
                fill_result = self._try_fill_order(
                    order,
                    current_price,
                    volume,
                    volatility,
                    spread,
                    current_time
                )

                if fill_result:
                    fill, fully_filled = fill_result
                    fills.append(fill)
                    self._fills.append(fill)

                    # Update order status
                    order.filled_quantity += fill.quantity
                    order.avg_fill_price = (
                        (order.avg_fill_price * (order.filled_quantity - fill.quantity) +
                         fill.price * fill.quantity) / order.filled_quantity
                    )
                    order.commission += fill.commission
                    order.slippage += fill.slippage

                    if fully_filled:
                        order.status = OrderStatus.FILLED
                        orders.remove(order)
                    else:
                        order.status = OrderStatus.PARTIALLY_FILLED

        return fills

    def _try_fill_order(
        self,
        order: Order,
        current_price: float,
        volume: float,
        volatility: float,
        spread: float,
        current_time: datetime
    ) -> Optional[Tuple[Fill, bool]]:
        """
        Try to fill an order.

        Args:
            order: Order to fill
            current_price: Current market price
            volume: Current volume
            volatility: Current volatility
            spread: Current spread
            current_time: Current time

        Returns:
            Tuple of (fill, fully_filled) if fillable, None otherwise
        """
        execution_price = None
        should_execute = False

        if order.order_type == OrderType.MARKET:
            # Market orders always execute
            should_execute = True

            # Calculate execution price with slippage
            slippage = self.slippage_model.calculate_slippage(
                order, current_price, volume, volatility, spread
            )

            if order.side == OrderSide.BUY:
                execution_price = current_price * (1 + slippage)
            else:
                execution_price = current_price * (1 - slippage)

        elif order.order_type == OrderType.LIMIT:
            # Limit orders execute if price is favorable
            if order.side == OrderSide.BUY and current_price <= order.price:
                should_execute = True
                execution_price = min(current_price, order.price)
            elif order.side == OrderSide.SELL and current_price >= order.price:
                should_execute = True
                execution_price = max(current_price, order.price)

        elif order.order_type == OrderType.STOP:
            # Stop orders trigger when price crosses stop price
            if order.stop_price is not None:
                if order.side == OrderSide.BUY and current_price >= order.stop_price:
                    should_execute = True
                    slippage = self.slippage_model.calculate_slippage(
                        order, current_price, volume, volatility, spread
                    )
                    execution_price = current_price * (1 + slippage)
                elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                    should_execute = True
                    slippage = self.slippage_model.calculate_slippage(
                        order, current_price, volume, volatility, spread
                    )
                    execution_price = current_price * (1 - slippage)

        elif order.order_type == OrderType.STOP_LIMIT:
            # Stop-limit: trigger becomes limit order
            if order.stop_price is not None and order.price is not None:
                # Check stop trigger
                if (order.side == OrderSide.BUY and current_price >= order.stop_price) or \
                   (order.side == OrderSide.SELL and current_price <= order.stop_price):
                    # Convert to limit order logic
                    if order.side == OrderSide.BUY and current_price <= order.price:
                        should_execute = True
                        execution_price = min(current_price, order.price)
                    elif order.side == OrderSide.SELL and current_price >= order.price:
                        should_execute = True
                        execution_price = max(current_price, order.price)

        if not should_execute or execution_price is None:
            return None

        # Determine fill quantity
        fill_quantity = order.remaining_quantity
        fully_filled = True

        if self.enable_partial_fills and np.random.random() < self.partial_fill_probability:
            fill_quantity = order.remaining_quantity * np.random.uniform(0.3, 0.9)
            fully_filled = False

        # Calculate commission
        commission = self.commission_model.calculate_commission(
            fill_quantity, execution_price, order.side
        )

        # Calculate slippage
        slippage = abs(execution_price - current_price) / current_price

        # Create fill
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=execution_price,
            commission=commission,
            slippage=slippage,
            timestamp=current_time
        )

        return fill, fully_filled

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get open orders.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of open orders
        """
        if symbol:
            return self._open_orders.get(symbol, [])
        else:
            all_orders = []
            for orders in self._open_orders.values():
                all_orders.extend(orders)
            return all_orders

    def get_fills(self) -> List[Fill]:
        """Get all fills."""
        return self._fills.copy()

    def clear_fills(self) -> None:
        """Clear fill history."""
        self._fills.clear()

    def reset(self) -> None:
        """Reset simulator state."""
        self._orders.clear()
        self._fills.clear()
        self._open_orders.clear()
        self._last_prices.clear()
        self._last_volumes.clear()
        self._volatilities.clear()
        self._spreads.clear()


def create_execution_simulator(
    commission_rate: float = 0.001,
    base_slippage: float = 0.0001,
    **kwargs
) -> ExecutionSimulator:
    """
    Convenience function to create an execution simulator.

    Args:
        commission_rate: Commission rate
        base_slippage: Base slippage rate
        **kwargs: Additional arguments

    Returns:
        Configured ExecutionSimulator
    """
    slippage_model = SlippageModel(base_slippage=base_slippage)
    commission_model = CommissionModel(commission_rate=commission_rate)

    return ExecutionSimulator(
        slippage_model=slippage_model,
        commission_model=commission_model,
        **kwargs
    )
