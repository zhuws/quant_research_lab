"""
Bybit Gateway for Quant Research Lab.

Implements exchange gateway for Bybit:
    - Spot and Futures trading
    - Order management
    - Position tracking
    - WebSocket streaming
    - Testnet support

Example usage:
    ```python
    from execution.bybit_gateway import BybitGateway
    from execution.exchange_gateway import ExchangeConfig

    config = ExchangeConfig(
        api_key='your_key',
        api_secret='your_secret',
        testnet=True,
        use_futures=True
    )

    gateway = BybitGateway(config)
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


class BybitGateway(ExchangeGateway):
    """
    Bybit Exchange Gateway.

    Implements ExchangeGateway for Bybit exchange with:
        - REST API for order management
        - WebSocket for real-time updates
        - V5 API support
        - Testnet support for testing

    Attributes:
        config: Exchange configuration
    """

    # API endpoints
    MAINNET_URL = "https://api.bybit.com"
    TESTNET_URL = "https://api-testnet.bybit.com"

    # WebSocket endpoints
    MAINNET_WS_URL = "wss://stream.bybit.com/v5/private"
    TESTNET_WS_URL = "wss://stream-testnet.bybit.com/v5/private"

    PUBLIC_WS_MAINNET = "wss://stream.bybit.com/v5/public/linear"
    PUBLIC_WS_TESTNET = "wss://stream-testnet.bybit.com/v5/public/linear"

    def __init__(self, config: ExchangeConfig):
        """
        Initialize Bybit Gateway.

        Args:
            config: Exchange configuration
        """
        super().__init__(config)
        self._session = None
        self._ws = None
        self._ws_task = None
        self._subscriptions: Dict[str, List[Callable]] = {}

        # Set URLs based on config
        self.base_url = self.TESTNET_URL if config.testnet else self.MAINNET_URL
        self.ws_url = self.TESTNET_WS_URL if config.testnet else self.MAINNET_WS_URL
        self.public_ws_url = self.PUBLIC_WS_TESTNET if config.testnet else self.PUBLIC_WS_MAINNET

    async def connect(self) -> bool:
        """
        Connect to Bybit.

        Returns:
            True if connected successfully
        """
        if not HAS_AIOHTTP:
            self.logger.error("aiohttp not installed. Run: pip install aiohttp")
            return False

        try:
            self._session = aiohttp.ClientSession()

            # Test connection
            await self._make_request('GET', '/v5/market/time')

            self._connected = True
            self.logger.info(f"Connected to Bybit {'testnet' if self.config.testnet else 'mainnet'}")
            return True

        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Bybit."""
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()
        self._connected = False
        self.logger.info("Disconnected from Bybit")

    async def get_account_info(self) -> AccountInfo:
        """
        Get account information.

        Returns:
            AccountInfo with balance and positions
        """
        data = await self._make_signed_request('GET', '/v5/account/wallet-balance', {
            'accountType': 'UNIFIED' if self.config.use_futures else 'SPOT'
        })

        if data.get('retCode') != 0:
            raise Exception(data.get('retMsg', 'Unknown error'))

        result = data.get('result', {}).get('list', [])
        if not result:
            return AccountInfo()

        account = result[0]
        coin_data = account.get('coin', [])

        total_balance = sum(float(c.get('walletBalance', 0)) for c in coin_data)
        available_balance = sum(float(c.get('availableToWithdraw', 0)) for c in coin_data)
        unrealized_pnl = sum(float(c.get('unrealisedPnl', 0)) for c in coin_data)

        positions = await self.get_positions()

        return AccountInfo(
            total_balance=total_balance,
            available_balance=available_balance,
            unrealized_pnl=unrealized_pnl,
            margin_balance=total_balance,
            positions=positions,
            update_time=datetime.utcnow()
        )

    async def get_positions(self) -> List[Position]:
        """
        Get all positions.

        Returns:
            List of Position objects
        """
        if not self.config.use_futures:
            return []

        data = await self._make_signed_request('GET', '/v5/position/list', {
            'category': 'linear',
            'settleCoin': 'USDT'
        })

        if data.get('retCode') != 0:
            self.logger.error(f"Error getting positions: {data.get('retMsg')}")
            return []

        positions = []
        for pos_data in data.get('result', {}).get('list', []):
            qty = float(pos_data.get('size', 0))
            if qty > 0:
                side = pos_data.get('side', '')
                positions.append(Position(
                    symbol=pos_data.get('symbol', ''),
                    position_side=PositionSide.LONG if side == 'Buy' else PositionSide.SHORT,
                    quantity=qty,
                    entry_price=float(pos_data.get('avgPrice', 0)),
                    mark_price=float(pos_data.get('markPrice', 0)),
                    unrealized_pnl=float(pos_data.get('unrealisedPnl', 0)),
                    liquidation_price=float(pos_data.get('liqPrice', 0)),
                    leverage=int(float(pos_data.get('leverage', 1))),
                    update_time=datetime.utcnow()
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
        if not self.config.use_futures:
            return None

        data = await self._make_signed_request('GET', '/v5/position/list', {
            'category': 'linear',
            'symbol': symbol
        })

        if data.get('retCode') != 0:
            return None

        positions = data.get('result', {}).get('list', [])
        if not positions:
            return None

        pos_data = positions[0]
        qty = float(pos_data.get('size', 0))
        if qty == 0:
            return None

        side = pos_data.get('side', '')
        return Position(
            symbol=symbol,
            position_side=PositionSide.LONG if side == 'Buy' else PositionSide.SHORT,
            quantity=qty,
            entry_price=float(pos_data.get('avgPrice', 0)),
            mark_price=float(pos_data.get('markPrice', 0)),
            unrealized_pnl=float(pos_data.get('unrealisedPnl', 0)),
            liquidation_price=float(pos_data.get('liqPrice', 0)),
            leverage=int(float(pos_data.get('leverage', 1))),
            update_time=datetime.utcnow()
        )

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

        # Build request
        params = {
            'category': 'linear' if self.config.use_futures else 'spot',
            'symbol': symbol,
            'side': 'Buy' if side == OrderSide.BUY else 'Sell',
            'qty': str(quantity)
        }

        # Map order type
        type_map = {
            OrderType.MARKET: 'Market',
            OrderType.LIMIT: 'Limit',
        }
        params['orderType'] = type_map.get(order_type, 'Limit')

        if price is not None and order_type == OrderType.LIMIT:
            params['price'] = str(price)
            params['timeInForce'] = (time_in_force or TimeInForce.GTC).value

        if stop_price is not None:
            params['triggerPrice'] = str(stop_price)

        # Handle position side for hedge mode
        if self.config.hedge_mode and position_side:
            params['positionIdx'] = 1 if position_side == PositionSide.LONG else 2

        params.update(kwargs)

        data = await self._make_signed_request('POST', '/v5/order/create', params)

        if data.get('retCode') != 0:
            raise Exception(data.get('retMsg', 'Order failed'))

        result = data.get('result', {})

        return Order(
            order_id=result.get('orderId', ''),
            client_order_id=result.get('orderLinkId', ''),
            symbol=symbol,
            side=side,
            order_type=order_type,
            status=OrderStatus.NEW,
            quantity=quantity,
            price=price or 0,
            create_time=datetime.utcnow()
        )

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
        params = {
            'category': 'linear' if self.config.use_futures else 'spot',
            'symbol': symbol
        }

        if order_id:
            params['orderId'] = order_id
        elif client_order_id:
            params['orderLinkId'] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id required")

        data = await self._make_signed_request('POST', '/v5/order/cancel', params)

        return data.get('retCode') == 0

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            Number of orders cancelled
        """
        params = {
            'category': 'linear' if self.config.use_futures else 'spot'
        }
        if symbol:
            params['symbol'] = symbol

        data = await self._make_signed_request('POST', '/v5/order/cancel-all', params)

        if data.get('retCode') != 0:
            return 0

        return data.get('result', {}).get('cancelled', 0)

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
        params = {
            'category': 'linear' if self.config.use_futures else 'spot',
            'symbol': symbol
        }

        if order_id:
            params['orderId'] = order_id
        elif client_order_id:
            params['orderLinkId'] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id required")

        data = await self._make_signed_request('GET', '/v5/order/realtime', params)

        if data.get('retCode') != 0:
            return None

        orders = data.get('result', {}).get('list', [])
        if not orders:
            return None

        return self._parse_order(orders[0])

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders
        """
        params = {
            'category': 'linear' if self.config.use_futures else 'spot',
            'openOnly': 0
        }
        if symbol:
            params['symbol'] = symbol

        data = await self._make_signed_request('GET', '/v5/order/realtime', params)

        if data.get('retCode') != 0:
            return []

        orders = data.get('result', {}).get('list', [])
        return [self._parse_order(o) for o in orders]

    async def get_ticker(self, symbol: str) -> Ticker:
        """
        Get current ticker.

        Args:
            symbol: Trading symbol

        Returns:
            Ticker object
        """
        data = await self._make_request('GET', '/v5/market/tickers', {
            'category': 'linear' if self.config.use_futures else 'spot',
            'symbol': symbol
        })

        if data.get('retCode') != 0:
            raise Exception(data.get('retMsg'))

        tickers = data.get('result', {}).get('list', [])
        if not tickers:
            raise Exception(f"No ticker data for {symbol}")

        t = tickers[0]
        return Ticker(
            symbol=symbol,
            bid=float(t.get('bid1Price', 0)),
            ask=float(t.get('ask1Price', 0)),
            last=float(t.get('lastPrice', 0)),
            volume=float(t.get('volume24h', 0)),
            quote_volume=float(t.get('turnover24h', 0)),
            high=float(t.get('highPrice24h', 0)),
            low=float(t.get('lowPrice24h', 0)),
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

        data = await self._make_signed_request('POST', '/v5/position/set-leverage', {
            'category': 'linear',
            'symbol': symbol,
            'buyLeverage': str(leverage),
            'sellLeverage': str(leverage)
        })

        return data.get('retCode') == 0

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
        # Bybit uses public WebSocket for tickers
        stream = f"public.{symbol.lower()}"
        await self._subscribe_public_stream(stream, callback)

    async def subscribe_orders(
        self,
        callback: Callable[[Order], None]
    ) -> None:
        """
        Subscribe to order updates.

        Args:
            callback: Callback function
        """
        await self._subscribe_private_stream('order', callback)

    async def subscribe_positions(
        self,
        callback: Callable[[Position], None]
    ) -> None:
        """
        Subscribe to position updates.

        Args:
            callback: Callback function
        """
        await self._subscribe_private_stream('position', callback)

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
        timestamp = str(int(time.time() * 1000))

        # Build signature string
        if method.upper() == 'POST':
            body = json.dumps(params)
            sign_string = timestamp + self.config.api_key + self.config.recv_window + body
            headers = {
                'X-BAPI-API-KEY': self.config.api_key,
                'X-BAPI-TIMESTAMP': timestamp,
                'X-BAPI-SIGN': self._generate_signature(sign_string, self.config.api_secret),
                'X-BAPI-RECV-WINDOW': str(self.config.recv_window),
                'Content-Type': 'application/json'
            }
            url = f"{self.base_url}{endpoint}"
            async with self._session.post(url, data=body, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        else:
            query_string = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
            sign_string = timestamp + self.config.api_key + self.config.recv_window + query_string
            headers = {
                'X-BAPI-API-KEY': self.config.api_key,
                'X-BAPI-TIMESTAMP': timestamp,
                'X-BAPI-SIGN': self._generate_signature(sign_string, self.config.api_secret),
                'X-BAPI-RECV-WINDOW': str(self.config.recv_window)
            }
            url = f"{self.base_url}{endpoint}?{query_string}"
            async with self._session.get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.json()

    async def _subscribe_public_stream(
        self,
        stream: str,
        callback: Callable
    ) -> None:
        """Subscribe to public WebSocket stream."""
        if stream not in self._subscriptions:
            self._subscriptions[stream] = []
        self._subscriptions[stream].append(callback)

    async def _subscribe_private_stream(
        self,
        stream: str,
        callback: Callable
    ) -> None:
        """Subscribe to private WebSocket stream."""
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

        try:
            async with websockets.connect(self.ws_url) as ws:
                self._ws = ws
                self.logger.info("WebSocket connected")

                # Authenticate
                auth_msg = {
                    'op': 'auth',
                    'args': [
                        self.config.api_key,
                        str(int(time.time() * 1000)),
                        self._generate_signature(
                            f"GET/realtime{int(time.time() * 1000)}",
                            self.config.api_secret
                        )
                    ]
                }
                await ws.send(json.dumps(auth_msg))

                # Subscribe
                sub_msg = {
                    'op': 'subscribe',
                    'args': list(self._subscriptions.keys())
                }
                await ws.send(json.dumps(sub_msg))

                async for message in ws:
                    data = json.loads(message)

                    if 'topic' in data:
                        topic = data['topic']
                        for callback in self._subscriptions.get(topic, []):
                            try:
                                callback(data.get('data', data))
                            except Exception as e:
                                self.logger.error(f"Callback error: {e}")

        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
            await asyncio.sleep(5)
            self._ws_task = asyncio.create_task(self._run_websocket())

    def _parse_order(self, data: Dict) -> Order:
        """Parse order from API response."""
        status_map = {
            'New': OrderStatus.NEW,
            'PartiallyFilled': OrderStatus.PARTIALLY_FILLED,
            'Filled': OrderStatus.FILLED,
            'Cancelled': OrderStatus.CANCELED,
            'Rejected': OrderStatus.REJECTED
        }

        side = OrderSide.BUY if data.get('side') == 'Buy' else OrderSide.SELL
        order_type = OrderType.MARKET if data.get('orderType') == 'Market' else OrderType.LIMIT

        return Order(
            order_id=data.get('orderId', ''),
            client_order_id=data.get('orderLinkId', ''),
            symbol=data.get('symbol', ''),
            side=side,
            order_type=order_type,
            status=status_map.get(data.get('orderStatus', ''), OrderStatus.NEW),
            quantity=float(data.get('qty', 0)),
            price=float(data.get('price', 0)),
            filled_quantity=float(data.get('cumExecQty', 0)),
            avg_price=float(data.get('avgPrice', 0)),
            create_time=datetime.fromtimestamp(int(data.get('createdTime', 0)) / 1000) if data.get('createdTime') else None,
            update_time=datetime.fromtimestamp(int(data.get('updatedTime', 0)) / 1000) if data.get('updatedTime') else None
        )


__all__ = ['BybitGateway']
