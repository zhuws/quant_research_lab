"""
Order Manager for Quant Research Lab.

Manages order lifecycle and execution:
    - Order submission and tracking
    - Order state machine
    - Execution analytics
    - Risk checks integration
    - Retry and error handling

Order management is essential for:
    - Reliable order execution
    - Trade tracking and audit
    - Performance analysis
    - Risk compliance

Example usage:
    ```python
    from execution.order_manager import OrderManager

    manager = OrderManager()

    # Submit order
    order_id = manager.submit_order({
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'type': 'LIMIT',
        'quantity': 0.1,
        'price': 40000
    })

    # Check status
    status = manager.get_order_status(order_id)
    ```
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from execution.exchange_gateway import (
    Order, OrderSide, OrderType, OrderStatus, TimeInForce
)


class OrderState(Enum):
    """Order state for state machine."""
    PENDING = 'pending'
    SUBMITTED = 'submitted'
    ACKNOWLEDGED = 'acknowledged'
    PARTIALLY_FILLED = 'partially_filled'
    FILLED = 'filled'
    CANCELING = 'canceling'
    CANCELED = 'canceled'
    REJECTED = 'rejected'
    EXPIRED = 'expired'
    FAILED = 'failed'


@dataclass
class OrderRequest:
    """
    Order request data.

    Attributes:
        symbol: Trading symbol
        side: Order side
        order_type: Order type
        quantity: Order quantity
        price: Limit price
        stop_price: Stop price
        time_in_force: Time in force
        client_order_id: Client order ID
        metadata: Additional metadata
    """
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_order(self) -> Order:
        """Convert to Order object."""
        return Order(
            order_id='',
            client_order_id=self.client_order_id or str(uuid.uuid4())[:8],
            symbol=self.symbol,
            side=self.side,
            order_type=self.order_type,
            status=OrderStatus.NEW,
            quantity=self.quantity,
            price=self.price or 0,
            stop_price=self.stop_price or 0,
            time_in_force=self.time_in_force
        )


@dataclass
class OrderRecord:
    """
    Complete order record with state history.

    Attributes:
        order: Order object
        state: Current state
        state_history: History of state transitions
        submit_time: When order was submitted
        fill_time: When order was filled
        error_message: Error if any
        retry_count: Number of retry attempts
    """
    order: Order
    state: OrderState = OrderState.PENDING
    state_history: List[tuple] = field(default_factory=list)
    submit_time: Optional[datetime] = None
    fill_time: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0

    def __post_init__(self):
        if not self.state_history:
            self.state_history = [(self.state, datetime.utcnow())]


class OrderManager:
    """
    Order Lifecycle Manager.

    Manages:
        - Order submission
        - State tracking
        - Execution monitoring
        - Retry logic
        - Analytics

    Attributes:
        max_retries: Maximum retry attempts for failed orders
        retry_delay: Delay between retries in seconds
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        order_timeout: int = 86400  # 24 hours
    ):
        """
        Initialize Order Manager.

        Args:
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
            order_timeout: Order timeout in seconds
        """
        self.logger = get_logger('order_manager')
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.order_timeout = order_timeout

        # Order storage
        self._orders: Dict[str, OrderRecord] = {}
        self._client_order_map: Dict[str, str] = {}

        # Callbacks
        self._state_callbacks: List[Callable] = []
        self._fill_callbacks: List[Callable] = []

        # Statistics
        self._total_submitted = 0
        self._total_filled = 0
        self._total_canceled = 0
        self._total_rejected = 0

        # Gateway reference (set externally)
        self._gateway = None

    def set_gateway(self, gateway: Any) -> None:
        """
        Set the exchange gateway for order execution.

        Args:
            gateway: Exchange gateway instance
        """
        self._gateway = gateway

    def submit_order(
        self,
        request: Union[OrderRequest, Dict[str, Any]],
        gateway: Optional[Any] = None
    ) -> str:
        """
        Submit an order.

        Args:
            request: Order request (OrderRequest or dict)
            gateway: Optional gateway override

        Returns:
            Client order ID
        """
        # Normalize request
        if isinstance(request, dict):
            request = self._dict_to_request(request)

        # Create order
        order = request.to_order()
        client_order_id = order.client_order_id

        # Create record
        record = OrderRecord(order=order)
        self._orders[client_order_id] = record
        self._client_order_map[order.order_id] = client_order_id

        # Update statistics
        self._total_submitted += 1

        # Set state
        self._set_state(record, OrderState.SUBMITTED)
        record.submit_time = datetime.utcnow()

        # Submit to gateway
        gateway = gateway or self._gateway
        if gateway:
            asyncio.create_task(self._submit_to_gateway(record, gateway))
        else:
            self.logger.warning(f"No gateway set for order {client_order_id}")

        self.logger.info(
            f"Order submitted: {client_order_id} {request.side.value} "
            f"{request.quantity} {request.symbol} @ {request.price or 'MARKET'}"
        )

        return client_order_id

    def cancel_order(
        self,
        client_order_id: str,
        gateway: Optional[Any] = None
    ) -> bool:
        """
        Cancel an order.

        Args:
            client_order_id: Client order ID
            gateway: Optional gateway override

        Returns:
            True if cancel request sent
        """
        if client_order_id not in self._orders:
            self.logger.warning(f"Order not found: {client_order_id}")
            return False

        record = self._orders[client_order_id]

        if record.state in [OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED]:
            self.logger.warning(f"Cannot cancel order in state {record.state.value}")
            return False

        self._set_state(record, OrderState.CANCELING)

        # Submit cancel to gateway
        gateway = gateway or self._gateway
        if gateway:
            asyncio.create_task(self._cancel_on_gateway(record, gateway))

        return True

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            Number of cancel requests sent
        """
        count = 0
        for client_order_id, record in self._orders.items():
            if record.state in [OrderState.SUBMITTED, OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED]:
                if symbol is None or record.order.symbol == symbol:
                    if self.cancel_order(client_order_id):
                        count += 1

        return count

    def get_order(self, client_order_id: str) -> Optional[Order]:
        """Get order by client order ID."""
        if client_order_id in self._orders:
            return self._orders[client_order_id].order
        return None

    def get_order_state(self, client_order_id: str) -> Optional[OrderState]:
        """Get order state by client order ID."""
        if client_order_id in self._orders:
            return self._orders[client_order_id].state
        return None

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders."""
        orders = []
        for record in self._orders.values():
            if record.state in [OrderState.SUBMITTED, OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED]:
                if symbol is None or record.order.symbol == symbol:
                    orders.append(record.order)
        return orders

    def get_orders_by_state(self, state: OrderState) -> List[Order]:
        """Get orders by state."""
        return [r.order for r in self._orders.values() if r.state == state]

    def update_order(self, order: Order) -> None:
        """
        Update order from gateway callback.

        Args:
            order: Updated order from exchange
        """
        # Find record
        client_order_id = self._client_order_map.get(order.order_id) or order.client_order_id

        if client_order_id not in self._orders:
            self.logger.warning(f"Received update for unknown order: {order.order_id}")
            return

        record = self._orders[client_order_id]

        # Update order data
        record.order = order

        # Map exchange status to state
        state_map = {
            OrderStatus.NEW: OrderState.ACKNOWLEDGED,
            OrderStatus.PARTIALLY_FILLED: OrderState.PARTIALLY_FILLED,
            OrderStatus.FILLED: OrderState.FILLED,
            OrderStatus.CANCELED: OrderState.CANCELED,
            OrderStatus.REJECTED: OrderState.REJECTED,
            OrderStatus.EXPIRED: OrderState.EXPIRED
        }

        new_state = state_map.get(order.status, record.state)

        # Check for fill
        if new_state == OrderState.FILLED:
            record.fill_time = datetime.utcnow()
            self._total_filled += 1
            self._notify_fill(record)

        elif new_state == OrderState.CANCELED:
            self._total_canceled += 1

        elif new_state == OrderState.REJECTED:
            self._total_rejected += 1

        self._set_state(record, new_state)

    def on_state_change(self, callback: Callable) -> None:
        """Register state change callback."""
        self._state_callbacks.append(callback)

    def on_fill(self, callback: Callable) -> None:
        """Register fill callback."""
        self._fill_callbacks.append(callback)

    def get_statistics(self) -> Dict[str, Any]:
        """Get order statistics."""
        states = defaultdict(int)
        for record in self._orders.values():
            states[record.state.value] += 1

        return {
            'total_submitted': self._total_submitted,
            'total_filled': self._total_filled,
            'total_canceled': self._total_canceled,
            'total_rejected': self._total_rejected,
            'fill_rate': self._total_filled / self._total_submitted if self._total_submitted > 0 else 0,
            'open_orders': len(self.get_open_orders()),
            'states': dict(states)
        }

    def cleanup_expired(self) -> int:
        """Remove expired orders from tracking."""
        expired = []
        cutoff = datetime.utcnow() - timedelta(seconds=self.order_timeout)

        for client_order_id, record in self._orders.items():
            if record.state in [OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED, OrderState.EXPIRED]:
                if record.fill_time and record.fill_time < cutoff:
                    expired.append(client_order_id)
                elif record.submit_time and record.submit_time < cutoff:
                    expired.append(client_order_id)

        for oid in expired:
            del self._orders[oid]

        if expired:
            self.logger.info(f"Cleaned up {len(expired)} expired orders")

        return len(expired)

    # Private methods

    def _dict_to_request(self, data: Dict[str, Any]) -> OrderRequest:
        """Convert dictionary to OrderRequest."""
        side = data.get('side', 'BUY')
        if isinstance(side, str):
            side = OrderSide(side.upper())

        order_type = data.get('type', data.get('order_type', 'LIMIT'))
        if isinstance(order_type, str):
            order_type = OrderType(order_type.upper())

        tif = data.get('time_in_force', 'GTC')
        if isinstance(tif, str):
            tif = TimeInForce(tif.upper())

        return OrderRequest(
            symbol=data.get('symbol', ''),
            side=side,
            order_type=order_type,
            quantity=float(data.get('quantity', 0)),
            price=float(data.get('price', 0)) if data.get('price') else None,
            stop_price=float(data.get('stop_price', 0)) if data.get('stop_price') else None,
            time_in_force=tif,
            client_order_id=data.get('client_order_id'),
            metadata=data.get('metadata', {})
        )

    def _set_state(self, record: OrderRecord, new_state: OrderState) -> None:
        """Set order state and record history."""
        if record.state != new_state:
            record.state = new_state
            record.state_history.append((new_state, datetime.utcnow()))

            # Notify callbacks
            for callback in self._state_callbacks:
                try:
                    callback(record.order, new_state)
                except Exception as e:
                    self.logger.error(f"State callback error: {e}")

    async def _submit_to_gateway(self, record: OrderRecord, gateway: Any) -> None:
        """Submit order to gateway."""
        try:
            order = await gateway.place_order(
                symbol=record.order.symbol,
                side=record.order.side,
                order_type=record.order.order_type,
                quantity=record.order.quantity,
                price=record.order.price if record.order.price > 0 else None,
                stop_price=record.order.stop_price if record.order.stop_price > 0 else None,
                time_in_force=record.order.time_in_force,
                client_order_id=record.order.client_order_id
            )

            # Update with exchange order ID
            record.order.order_id = order.order_id
            self._client_order_map[order.order_id] = record.order.client_order_id

            self._set_state(record, OrderState.ACKNOWLEDGED)

        except Exception as e:
            self.logger.error(f"Order submission failed: {e}")
            record.error_message = str(e)
            record.retry_count += 1

            if record.retry_count < self.max_retries:
                await asyncio.sleep(self.retry_delay)
                await self._submit_to_gateway(record, gateway)
            else:
                self._set_state(record, OrderState.FAILED)

    async def _cancel_on_gateway(self, record: OrderRecord, gateway: Any) -> None:
        """Cancel order on gateway."""
        try:
            success = await gateway.cancel_order(
                symbol=record.order.symbol,
                order_id=record.order.order_id,
                client_order_id=record.order.client_order_id
            )

            if success:
                self._set_state(record, OrderState.CANCELED)
            else:
                self.logger.warning(f"Cancel failed for order {record.order.client_order_id}")

        except Exception as e:
            self.logger.error(f"Cancel error: {e}")
            record.error_message = str(e)

    def _notify_fill(self, record: OrderRecord) -> None:
        """Notify fill callbacks."""
        for callback in self._fill_callbacks:
            try:
                callback(record.order)
            except Exception as e:
                self.logger.error(f"Fill callback error: {e}")


__all__ = [
    'OrderState',
    'OrderRequest',
    'OrderRecord',
    'OrderManager'
]
