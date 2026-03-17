"""
Exchange Gateway for Quant Research Lab.

Abstract interface for exchange connectivity:
    - Unified API for multiple exchanges
    - Order placement and management
    - Position tracking
    - Account information
    - Market data streaming

Supported exchanges:
    - Binance (Spot, Futures)
    - Bybit (Spot, Futures)

Example usage:
    ```python
    from execution.exchange_gateway import ExchangeConfig
    from execution.binance_gateway import BinanceGateway

    config = ExchangeConfig(
        api_key='your_key',
        api_secret='your_secret',
        testnet=True
    )

    gateway = BinanceGateway(config)
    await gateway.connect()

    # Place order
    order = await gateway.place_order(
        symbol='BTCUSDT',
        side='BUY',
        order_type='LIMIT',
        quantity=0.001,
        price=40000
    )
    ```
"""

import asyncio
import hashlib
import hmac
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger


class OrderSide(Enum):
    """Order side."""
    BUY = 'BUY'
    SELL = 'SELL'


class OrderType(Enum):
    """Order type."""
    MARKET = 'MARKET'
    LIMIT = 'LIMIT'
    STOP_MARKET = 'STOP_MARKET'
    STOP_LIMIT = 'STOP_LIMIT'
    TAKE_PROFIT = 'TAKE_PROFIT'
    TAKE_PROFIT_LIMIT = 'TAKE_PROFIT_LIMIT'


class OrderStatus(Enum):
    """Order status."""
    NEW = 'NEW'
    PARTIALLY_FILLED = 'PARTIALLY_FILLED'
    FILLED = 'FILLED'
    CANCELED = 'CANCELED'
    REJECTED = 'REJECTED'
    EXPIRED = 'EXPIRED'


class TimeInForce(Enum):
    """Time in force."""
    GTC = 'GTC'  # Good Till Cancel
    IOC = 'IOC'  # Immediate or Cancel
    FOK = 'FOK'  # Fill or Kill
    GTX = 'GTX'  # Good Till Crossing (Post Only)


class PositionSide(Enum):
    """Position side (for hedge mode)."""
    LONG = 'LONG'
    SHORT = 'SHORT'
    BOTH = 'BOTH'


@dataclass
class ExchangeConfig:
    """
    Exchange configuration.

    Attributes:
        api_key: API key
        api_secret: API secret
        testnet: Use testnet/sandbox
        rate_limit: Requests per second
        timeout: Request timeout in seconds
        recv_window: Time window for request validity
        use_futures: Use futures API
        hedge_mode: Use hedge mode (separate long/short positions)
    """
    api_key: str = ''
    api_secret: str = ''
    testnet: bool = True
    rate_limit: int = 50
    timeout: int = 30
    recv_window: int = 5000
    use_futures: bool = True
    hedge_mode: bool = False


@dataclass
class Order:
    """
    Order data structure.

    Attributes:
        order_id: Exchange order ID
        client_order_id: Client order ID
        symbol: Trading symbol
        side: Order side
        order_type: Order type
        status: Current status
        quantity: Order quantity
        price: Order price
        filled_quantity: Filled quantity
        avg_price: Average fill price
        commission: Commission paid
        commission_asset: Commission asset
        time_in_force: Time in force
        create_time: Order creation time
        update_time: Last update time
        position_side: Position side for hedge mode
        stop_price: Stop price for stop orders
    """
    order_id: str = ''
    client_order_id: str = ''
    symbol: str = ''
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.LIMIT
    status: OrderStatus = OrderStatus.NEW
    quantity: float = 0.0
    price: float = 0.0
    filled_quantity: float = 0.0
    avg_price: float = 0.0
    commission: float = 0.0
    commission_asset: str = ''
    time_in_force: TimeInForce = TimeInForce.GTC
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    position_side: PositionSide = PositionSide.BOTH
    stop_price: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'status': self.status.value,
            'quantity': self.quantity,
            'price': self.price,
            'filled_quantity': self.filled_quantity,
            'avg_price': self.avg_price,
            'commission': self.commission,
            'commission_asset': self.commission_asset,
            'time_in_force': self.time_in_force.value,
            'create_time': self.create_time.isoformat() if self.create_time else None,
            'update_time': self.update_time.isoformat() if self.update_time else None,
            'position_side': self.position_side.value,
            'stop_price': self.stop_price
        }


@dataclass
class Position:
    """
    Position data structure.

    Attributes:
        symbol: Trading symbol
        position_side: Position side
        quantity: Position quantity
        entry_price: Average entry price
        mark_price: Current mark price
        unrealized_pnl: Unrealized profit/loss
        liquidation_price: Estimated liquidation price
        leverage: Position leverage
        margin_type: Margin type (ISOLATED, CROSSED)
        update_time: Last update time
    """
    symbol: str = ''
    position_side: PositionSide = PositionSide.BOTH
    quantity: float = 0.0
    entry_price: float = 0.0
    mark_price: float = 0.0
    unrealized_pnl: float = 0.0
    liquidation_price: float = 0.0
    leverage: int = 1
    margin_type: str = 'CROSSED'
    update_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'position_side': self.position_side.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'mark_price': self.mark_price,
            'unrealized_pnl': self.unrealized_pnl,
            'liquidation_price': self.liquidation_price,
            'leverage': self.leverage,
            'margin_type': self.margin_type,
            'update_time': self.update_time.isoformat() if self.update_time else None
        }


@dataclass
class AccountInfo:
    """
    Account information.

    Attributes:
        total_balance: Total account balance
        available_balance: Available for trading
        unrealized_pnl: Total unrealized P&L
        margin_balance: Margin balance
        positions: List of positions
        update_time: Last update time
    """
    total_balance: float = 0.0
    available_balance: float = 0.0
    unrealized_pnl: float = 0.0
    margin_balance: float = 0.0
    positions: List[Position] = field(default_factory=list)
    update_time: Optional[datetime] = None


@dataclass
class Ticker:
    """
    Market ticker data.

    Attributes:
        symbol: Trading symbol
        bid: Best bid price
        ask: Best ask price
        last: Last traded price
        volume: 24h volume
        quote_volume: 24h quote volume
        high: 24h high
        low: 24h low
        timestamp: Last update time
    """
    symbol: str = ''
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: float = 0.0
    quote_volume: float = 0.0
    high: float = 0.0
    low: float = 0.0
    timestamp: Optional[datetime] = None


class ExchangeGateway(ABC):
    """
    Abstract Exchange Gateway Interface.

    Defines the unified API for all exchange implementations.
    Subclasses implement exchange-specific logic.

    Features:
        - Order placement and management
        - Position tracking
        - Account information
        - Market data
        - WebSocket streaming
    """

    def __init__(self, config: ExchangeConfig):
        """
        Initialize Exchange Gateway.

        Args:
            config: Exchange configuration
        """
        self.config = config
        self.logger = get_logger(f'{self.__class__.__name__}')
        self._connected = False
        self._callbacks: Dict[str, List[Callable]] = {}
        self._last_request_time = 0

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to exchange.

        Returns:
            True if connected successfully
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        pass

    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """
        Get account information.

        Returns:
            AccountInfo with balance and positions
        """
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """
        Get all positions.

        Returns:
            List of Position objects
        """
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position or None
        """
        pass

    @abstractmethod
    async def place_order(
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
            side: Order side (BUY/SELL)
            order_type: Order type (MARKET/LIMIT/etc)
            quantity: Order quantity
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force
            position_side: Position side (for hedge mode)
            **kwargs: Additional exchange-specific parameters

        Returns:
            Order object
        """
        pass

    @abstractmethod
    async def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None
    ) -> bool:
        """
        Cancel an order.

        Args:
            symbol: Trading symbol
            order_id: Exchange order ID
            client_order_id: Client order ID

        Returns:
            True if cancelled successfully
        """
        pass

    @abstractmethod
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all orders.

        Args:
            symbol: Optional symbol to cancel orders for

        Returns:
            Number of orders cancelled
        """
        pass

    @abstractmethod
    async def get_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None
    ) -> Optional[Order]:
        """
        Get order by ID.

        Args:
            symbol: Trading symbol
            order_id: Exchange order ID
            client_order_id: Client order ID

        Returns:
            Order or None
        """
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders
        """
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """
        Get current ticker.

        Args:
            symbol: Trading symbol

        Returns:
            Ticker object
        """
        pass

    @abstractmethod
    async def set_leverage(
        self,
        symbol: str,
        leverage: int
    ) -> bool:
        """
        Set leverage for a symbol.

        Args:
            symbol: Trading symbol
            leverage: Leverage value

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def subscribe_ticker(
        self,
        symbol: str,
        callback: Callable[[Ticker], None]
    ) -> None:
        """
        Subscribe to ticker updates.

        Args:
            symbol: Trading symbol
            callback: Callback function
        """
        pass

    @abstractmethod
    async def subscribe_orders(
        self,
        callback: Callable[[Order], None]
    ) -> None:
        """
        Subscribe to order updates.

        Args:
            callback: Callback function
        """
        pass

    @abstractmethod
    async def subscribe_positions(
        self,
        callback: Callable[[Position], None]
    ) -> None:
        """
        Subscribe to position updates.

        Args:
            callback: Callback function
        """
        pass

    def is_connected(self) -> bool:
        """Check if connected to exchange."""
        return self._connected

    def _generate_signature(
        self,
        query_string: str,
        secret: str
    ) -> str:
        """
        Generate HMAC signature.

        Args:
            query_string: Query string to sign
            secret: API secret

        Returns:
            Hex signature
        """
        return hmac.new(
            secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds."""
        return int(time.time() * 1000)

    def _rate_limit_wait(self) -> None:
        """Wait for rate limit."""
        min_interval = 1.0 / self.config.rate_limit
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    def on(self, event: str, callback: Callable) -> None:
        """
        Register event callback.

        Args:
            event: Event name
            callback: Callback function
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def _emit(self, event: str, data: Any) -> None:
        """
        Emit event to callbacks.

        Args:
            event: Event name
            data: Event data
        """
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Callback error for {event}: {e}")


__all__ = [
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'TimeInForce',
    'PositionSide',
    'ExchangeConfig',
    'Order',
    'Position',
    'AccountInfo',
    'Ticker',
    'ExchangeGateway'
]
