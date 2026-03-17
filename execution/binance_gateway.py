"""
Binance Gateway for Quant Research Lab.

Implements exchange gateway for Binance:
    - Spot and Futures trading
    - Order management
    - Position tracking
    - WebSocket streaming
    - Testnet support

Example usage:
    ```python
    from execution.binance_gateway import BinanceGateway
    from execution.exchange_gateway import ExchangeConfig

    config = ExchangeConfig(
        api_key='your_key',
        api_secret='your_secret',
        testnet=True,
        use_futures=True
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
import json
import time
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from execution.exchange_gateway import (
    ExchangeGateway, ExchangeConfig, Order, Position, AccountInfo, Ticker,
    OrderSide, OrderType, OrderStatus, TimeInForce, PositionSide
)

# Try to import optional dependencies
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


class BinanceGateway(ExchangeGateway):
    """
    Binance Exchange Gateway.

    Implements ExchangeGateway for Binance exchange with:
        - REST API for order management
        - WebSocket for real-time updates
        - Spot and Futures support
        - Testnet support for testing

    Attributes:
        config: Exchange configuration
    """

    # API endpoints
    SPOT_URL = "https://api.binance.com"
    SPOT_TESTNET_URL = "https://testnet.binance.vision"
    FUTURES_URL = "https://fapi.binance.com"
    FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"

    # WebSocket endpoints
    SPOT_WS_URL = "wss://stream.binance.com:9443/ws"
    SPOT_TESTNET_WS_URL = "wss://testnet.binance.vision/ws"
    FUTURES_WS_URL = "wss://fstream.binance.com/ws"
    FUTURES_TESTNET_WS_URL = "wss://stream.binancefuture.com/ws"

    def __init__(self, config: ExchangeConfig):
        """
        Initialize Binance Gateway.

        Args:
            config: Exchange configuration
        """
        super().__init__(config)
        self._session = None
        self._ws = None
        self._ws_task = None
        self._listen_key = None
        self._subscriptions: Dict[str, List[Callable]] = {}

        # Set URLs based on config
        if config.use_futures:
            self.base_url = self.FUTURES_TESTNET_URL if config.testnet else self.FUTURES_URL
            self.ws_url = self.FUTURES_TESTNET_WS_URL if config.testnet else self.FUTURES_WS_URL
        else:
            self.base_url = self.SPOT_TESTNET_URL if config.testnet else self.SPOT_URL
            self.ws_url = self.SPOT_TESTNET_WS_URL if config.testnet else self.SPOT_WS_URL

    async def connect(self) -> bool:
        """
        Connect to Binance.

        Returns:
            True if connected successfully
        """
        if not HAS_AIOHTTP:
            self.logger.error("aiohttp not installed. Run: pip install aiohttp")
            return False

        try:
            self._session = aiohttp.ClientSession(
                headers={'X-MBX-APIKEY': self.config.api_key}
            )

            # Test connection
            await self._make_request('GET', '/fapi/v1/ping')

            # Get listen key for user stream
            if self.config.api_key:
                self._listen_key = await self._get_listen_key()

            self._connected = True
            self.logger.info(f"Connected to Binance {'testnet' if self.config.testnet else 'mainnet'}")
            return True

        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Binance."""
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()
        self._connected = False
        self.logger.info("Disconnected from Binance")

    async def get_account_info(self) -> AccountInfo:
        """
        Get account information.

        Returns:
            AccountInfo with balance and positions
        """
        if self.config.use_futures:
            data = await self._make_signed_request('GET', '/fapi/v2/account')
            return self._parse_futures_account(data)
        else:
            data = await self._make_signed_request('GET', '/api/v3/account')
            return self._parse_spot_account(data)

    async def get_positions(self) -> List[Position]:
        """
        Get all positions.

        Returns:
            List of Position objects
        """
        if not self.config.use_futures:
            return []  # No positions in spot

        data = await self._make_signed_request('GET', '/fapi/v2/positionRisk')
        positions = []

        for pos_data in data:
            qty = float(pos_data.get('positionAmt', 0))
            if qty != 0:
                positions.append(Position(
                    symbol=pos_data['symbol'],
                    position_side=PositionSide.LONG if qty > 0 else PositionSide.SHORT,
                    quantity=abs(qty),
                    entry_price=float(pos_data.get('entryPrice', 0)),
                    mark_price=float(pos_data.get('markPrice', 0)),
                    unrealized_pnl=float(pos_data.get('unRealizedProfit', 0)),
                    liquidation_price=float(pos_data.get('liquidationPrice', 0)),
                    leverage=int(pos_data.get('leverage', 1)),
                    update_time=datetime.fromtimestamp(pos_data.get('updateTime', 0) / 1000)
                ))

        return positions

    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position or None
        """
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

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
            side: Order side
            order_type: Order type
            quantity: Order quantity
            price: Limit price
            stop_price: Stop price
            time_in_force: Time in force
            position_side: Position side for hedge mode
            **kwargs: Additional parameters

        Returns:
            Order object
        """
        # Normalize inputs
        if isinstance(side, str):
            side = OrderSide(side.upper())
        if isinstance(order_type, str):
            order_type = OrderType(order_type.upper())
        if isinstance(time_in_force, str):
            time_in_force = TimeInForce(time_in_force.upper())
        if isinstance(position_side, str):
            position_side = PositionSide(position_side.upper())

        # Build request
        params = {
            'symbol': symbol,
            'side': side.value,
            'type': order_type.value,
            'quantity': quantity
        }

        if price is not None and order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            params['price'] = price
            params['timeInForce'] = (time_in_force or TimeInForce.GTC).value

        if stop_price is not None:
            params['stopPrice'] = stop_price

        if position_side and self.config.hedge_mode:
            params['positionSide'] = position_side.value

        # Add any extra params
        params.update(kwargs)

        # Make request
        endpoint = '/fapi/v1/order' if self.config.use_futures else '/api/v3/order'
        data = await self._make_signed_request('POST', endpoint, params)

        return self._parse_order(data)

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
        params = {'symbol': symbol}

        if order_id:
            params['orderId'] = order_id
        elif client_order_id:
            params['origClientOrderId'] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id required")

        endpoint = '/fapi/v1/order' if self.config.use_futures else '/api/v3/order'
        await self._make_signed_request('DELETE', endpoint, params)

        return True

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            Number of orders cancelled
        """
        if symbol:
            endpoint = '/fapi/v1/allOpenOrders' if self.config.use_futures else '/api/v3/openOrders'
            params = {'symbol': symbol}
            await self._make_signed_request('DELETE', endpoint, params)
            return 1  # Binance returns OK, not count
        else:
            # Need to cancel per symbol
            open_orders = await self.get_open_orders()
            symbols = set(o.symbol for o in open_orders)
            count = 0
            for sym in symbols:
                try:
                    endpoint = '/fapi/v1/allOpenOrders' if self.config.use_futures else '/api/v3/openOrders'
                    params = {'symbol': sym}
                    await self._make_signed_request('DELETE', endpoint, params)
                    count += 1
                except Exception as e:
                    self.logger.error(f"Error cancelling orders for {sym}: {e}")
            return count

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
        params = {'symbol': symbol}

        if order_id:
            params['orderId'] = order_id
        elif client_order_id:
            params['origClientOrderId'] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id required")

        endpoint = '/fapi/v1/order' if self.config.use_futures else '/api/v3/order'
        data = await self._make_signed_request('GET', endpoint, params)

        return self._parse_order(data)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders
        """
        params = {}
        if symbol:
            params['symbol'] = symbol

        endpoint = '/fapi/v1/openOrders' if self.config.use_futures else '/api/v3/openOrders'
        data = await self._make_signed_request('GET', endpoint, params)

        return [self._parse_order(o) for o in data]

    async def get_ticker(self, symbol: str) -> Ticker:
        """
        Get current ticker.

        Args:
            symbol: Trading symbol

        Returns:
            Ticker object
        """
        endpoint = '/fapi/v1/ticker/bookTicker' if self.config.use_futures else '/api/v3/ticker/bookTicker'
        data = await self._make_request('GET', endpoint, {'symbol': symbol})

        endpoint_24h = '/fapi/v1/ticker/24hr' if self.config.use_futures else '/api/v3/ticker/24hr'
        data_24h = await self._make_request('GET', endpoint_24h, {'symbol': symbol})

        return Ticker(
            symbol=symbol,
            bid=float(data.get('bidPrice', 0)),
            ask=float(data.get('askPrice', 0)),
            last=float(data_24h.get('lastPrice', 0)),
            volume=float(data_24h.get('volume', 0)),
            quote_volume=float(data_24h.get('quoteVolume', 0)),
            high=float(data_24h.get('highPrice', 0)),
            low=float(data_24h.get('lowPrice', 0)),
            timestamp=datetime.utcnow()
        )

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
        if not self.config.use_futures:
            self.logger.warning("Leverage only available for futures")
            return False

        await self._make_signed_request(
            'POST',
            '/fapi/v1/leverage',
            {'symbol': symbol, 'leverage': leverage}
        )
        return True

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
        stream = f"{symbol.lower()}@bookTicker"
        await self._subscribe_stream(stream, callback)

    async def subscribe_orders(
        self,
        callback: Callable[[Order], None]
    ) -> None:
        """
        Subscribe to order updates.

        Args:
            callback: Callback function
        """
        if not self._listen_key:
            self._listen_key = await self._get_listen_key()

        await self._subscribe_stream(self._listen_key, callback)

    async def subscribe_positions(
        self,
        callback: Callable[[Position], None]
    ) -> None:
        """
        Subscribe to position updates.

        Args:
            callback: Callback function
        """
        # Position updates come through the user stream
        if not self._listen_key:
            self._listen_key = await self._get_listen_key()

        await self._subscribe_stream(self._listen_key, callback)

    # Private methods

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Any:
        """Make public API request."""
        self._rate_limit_wait()
        url = f"{self.base_url}{endpoint}"

        async with self._session.request(method, url, params=params) as response:
            if response.status == 429:
                # Rate limited
                retry_after = int(response.headers.get('Retry-After', 60))
                await asyncio.sleep(retry_after)
                return await self._make_request(method, endpoint, params)

            response.raise_for_status()
            return await response.json()

    async def _make_signed_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Any:
        """Make signed API request."""
        params = params or {}
        params['timestamp'] = self._get_timestamp()
        params['recvWindow'] = self.config.recv_window

        query_string = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
        signature = self._generate_signature(query_string, self.config.api_secret)
        params['signature'] = signature

        return await self._make_request(method, endpoint, params)

    async def _get_listen_key(self) -> str:
        """Get listen key for user data stream."""
        endpoint = '/fapi/v1/listenKey' if self.config.use_futures else '/api/v3/userDataStream'
        data = await self._make_signed_request('POST', endpoint)
        return data.get('listenKey', '')

    async def _subscribe_stream(
        self,
        stream: str,
        callback: Callable
    ) -> None:
        """Subscribe to WebSocket stream."""
        if stream not in self._subscriptions:
            self._subscriptions[stream] = []
        self._subscriptions[stream].append(callback)

        # Start WebSocket if not running
        if not self._ws_task:
            self._ws_task = asyncio.create_task(self._run_websocket())

    async def _run_websocket(self) -> None:
        """Run WebSocket connection."""
        if not HAS_WEBSOCKETS:
            self.logger.error("websockets not installed. Run: pip install websockets")
            return

        streams = '/'.join(self._subscriptions.keys())
        url = f"{self.ws_url}/{streams}"

        try:
            async with websockets.connect(url) as ws:
                self._ws = ws
                self.logger.info("WebSocket connected")

                async for message in ws:
                    data = json.loads(message)

                    # Determine stream and call callbacks
                    if 'stream' in data:
                        stream = data['stream']
                        for callback in self._subscriptions.get(stream, []):
                            try:
                                callback(data['data'])
                            except Exception as e:
                                self.logger.error(f"Callback error: {e}")
                    else:
                        # User data stream
                        for callback in self._subscriptions.get(self._listen_key, []):
                            try:
                                callback(data)
                            except Exception as e:
                                self.logger.error(f"Callback error: {e}")

        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
            # Reconnect
            await asyncio.sleep(5)
            self._ws_task = asyncio.create_task(self._run_websocket())

    def _parse_order(self, data: Dict) -> Order:
        """Parse order from API response."""
        return Order(
            order_id=str(data.get('orderId', '')),
            client_order_id=data.get('clientOrderId', ''),
            symbol=data.get('symbol', ''),
            side=OrderSide(data.get('side', 'BUY')),
            order_type=OrderType(data.get('type', 'LIMIT')),
            status=OrderStatus(data.get('status', 'NEW')),
            quantity=float(data.get('origQty', 0)),
            price=float(data.get('price', 0)),
            filled_quantity=float(data.get('executedQty', 0)),
            avg_price=float(data.get('avgPrice', 0)),
            commission=float(data.get('commission', 0)),
            commission_asset=data.get('commissionAsset', ''),
            time_in_force=TimeInForce(data.get('timeInForce', 'GTC')),
            create_time=datetime.fromtimestamp(data.get('time', 0) / 1000) if data.get('time') else None,
            update_time=datetime.fromtimestamp(data.get('updateTime', 0) / 1000) if data.get('updateTime') else None,
            position_side=PositionSide(data.get('positionSide', 'BOTH')),
            stop_price=float(data.get('stopPrice', 0))
        )

    def _parse_futures_account(self, data: Dict) -> AccountInfo:
        """Parse futures account info."""
        positions = []
        for pos_data in data.get('positions', []):
            qty = float(pos_data.get('positionAmt', 0))
            if qty != 0:
                positions.append(Position(
                    symbol=pos_data['symbol'],
                    position_side=PositionSide.LONG if qty > 0 else PositionSide.SHORT,
                    quantity=abs(qty),
                    entry_price=float(pos_data.get('entryPrice', 0)),
                    unrealized_pnl=float(pos_data.get('unRealizedProfit', 0)),
                    leverage=int(pos_data.get('leverage', 1))
                ))

        return AccountInfo(
            total_balance=float(data.get('totalWalletBalance', 0)),
            available_balance=float(data.get('availableBalance', 0)),
            unrealized_pnl=float(data.get('totalUnrealizedProfit', 0)),
            margin_balance=float(data.get('totalMarginBalance', 0)),
            positions=positions,
            update_time=datetime.fromtimestamp(data.get('updateTime', 0) / 1000)
        )

    def _parse_spot_account(self, data: Dict) -> AccountInfo:
        """Parse spot account info."""
        total_balance = sum(
            float(b.get('free', 0)) + float(b.get('locked', 0))
            for b in data.get('balances', [])
        )
        available_balance = sum(
            float(b.get('free', 0))
            for b in data.get('balances', [])
        )

        return AccountInfo(
            total_balance=total_balance,
            available_balance=available_balance,
            unrealized_pnl=0,
            margin_balance=total_balance,
            positions=[],
            update_time=datetime.fromtimestamp(data.get('updateTime', 0) / 1000)
        )


__all__ = ['BinanceGateway']
