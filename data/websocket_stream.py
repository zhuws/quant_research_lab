"""
WebSocket Stream Module for Quant Research Lab.
Handles real-time data streaming from exchanges.
"""

import asyncio
import json
import threading
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from collections import deque
import websockets

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.time_utils import from_timestamp


class WebSocketMessage:
    """WebSocket message container."""

    def __init__(
        self,
        exchange: str,
        channel: str,
        data: Dict[str, Any],
        timestamp: datetime = None
    ):
        """
        Initialize WebSocket message.

        Args:
            exchange: Exchange name
            channel: Data channel
            data: Message data
            timestamp: Message timestamp
        """
        self.exchange = exchange
        self.channel = channel
        self.data = data
        self.timestamp = timestamp or datetime.utcnow()


class BinanceWebSocketStream:
    """
    Binance WebSocket stream handler.

    Handles real-time data streaming for OHLCV, trades, and orderbook.
    """

    SPOT_WS_URL = "wss://stream.binance.com:9443/ws"
    FUTURES_WS_URL = "wss://fstream.binance.com/ws"

    def __init__(
        self,
        symbol: str,
        use_futures: bool = True,
        on_message: Optional[Callable[[WebSocketMessage], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None
    ):
        """
        Initialize Binance WebSocket stream.

        Args:
            symbol: Trading symbol (lowercase)
            use_futures: Use futures WebSocket
            on_message: Message callback
            on_error: Error callback
        """
        self.symbol = symbol.lower()
        self.use_futures = use_futures
        self.on_message = on_message
        self.on_error = on_error

        self.ws_url = self.FUTURES_WS_URL if use_futures else self.SPOT_WS_URL
        self.logger = get_logger('binance_ws')

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Message buffer
        self._message_buffer: deque = deque(maxlen=10000)

    def _get_stream_name(self, channel: str) -> str:
        """Get stream name for channel."""
        return f"{self.symbol}@{channel}"

    async def connect(
        self,
        channels: List[str] = None
    ) -> None:
        """
        Connect to WebSocket.

        Args:
            channels: List of channels to subscribe
                     (kline_1m, aggTrade, depth, bookTicker)
        """
        channels = channels or ['kline_1m', 'aggTrade', 'bookTicker']

        # Build combined stream URL
        streams = []
        for ch in channels:
            streams.append(self._get_stream_name(ch))

        stream_url = f"{self.ws_url}/{'/'.join(streams)}"

        self._running = True

        try:
            self._ws = await websockets.connect(stream_url)
            self.logger.info(f"Connected to Binance WebSocket: {self.symbol}")

            await self._listen()

        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
            if self.on_error:
                self.on_error(e)

    async def _listen(self) -> None:
        """Listen for incoming messages."""
        while self._running:
            try:
                message = await self._ws.recv()
                data = json.loads(message)

                # Handle combined stream format
                if 'stream' in data:
                    stream = data['stream']
                    event_data = data['data']
                    channel = stream.split('@')[1]
                else:
                    event_data = data
                    channel = data.get('e', 'unknown')

                # Parse message
                msg = self._parse_message(channel, event_data)

                if msg and self.on_message:
                    self._message_buffer.append(msg)
                    self.on_message(msg)

            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("WebSocket connection closed, reconnecting...")
                await asyncio.sleep(1)
                await self.connect()

            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                if self.on_error:
                    self.on_error(e)

    def _parse_message(
        self,
        channel: str,
        data: Dict[str, Any]
    ) -> Optional[WebSocketMessage]:
        """Parse WebSocket message."""
        try:
            if 'kline' in channel:
                return self._parse_kline(data)
            elif 'aggTrade' in channel:
                return self._parse_trade(data)
            elif 'bookTicker' in channel:
                return self._parse_book_ticker(data)
            elif 'depth' in channel:
                return self._parse_depth(data)
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error parsing message: {e}")
            return None

    def _parse_kline(self, data: Dict) -> WebSocketMessage:
        """Parse kline/candlestick data."""
        k = data['k']
        return WebSocketMessage(
            exchange='binance',
            channel='kline',
            data={
                'symbol': k['s'],
                'timeframe': k['i'],
                'open_time': from_timestamp(k['t']),
                'close_time': from_timestamp(k['T']),
                'open': float(k['o']),
                'high': float(k['h']),
                'low': float(k['l']),
                'close': float(k['c']),
                'volume': float(k['v']),
                'trades': k['n'],
                'is_closed': k['x']
            },
            timestamp=datetime.utcnow()
        )

    def _parse_trade(self, data: Dict) -> WebSocketMessage:
        """Parse aggregated trade data."""
        return WebSocketMessage(
            exchange='binance',
            channel='trade',
            data={
                'symbol': data['s'],
                'trade_id': data['a'],
                'price': float(data['p']),
                'quantity': float(data['q']),
                'timestamp': from_timestamp(data['T']),
                'side': 'sell' if data['m'] else 'buy'
            },
            timestamp=datetime.utcnow()
        )

    def _parse_book_ticker(self, data: Dict) -> WebSocketMessage:
        """Parse book ticker data."""
        return WebSocketMessage(
            exchange='binance',
            channel='bookTicker',
            data={
                'symbol': data['s'],
                'bid_price': float(data['b']),
                'bid_quantity': float(data['B']),
                'ask_price': float(data['a']),
                'ask_quantity': float(data['A'])
            },
            timestamp=datetime.utcnow()
        )

    def _parse_depth(self, data: Dict) -> WebSocketMessage:
        """Parse orderbook depth data."""
        return WebSocketMessage(
            exchange='binance',
            channel='depth',
            data={
                'symbol': data.get('s', self.symbol.upper()),
                'bids': [[float(p), float(q)] for p, q in data.get('b', [])],
                'asks': [[float(p), float(q)] for p, q in data.get('a', [])]
            },
            timestamp=datetime.utcnow()
        )

    async def subscribe(self, channels: List[str]) -> None:
        """
        Subscribe to additional channels.

        Args:
            channels: List of channels to subscribe
        """
        params = [self._get_stream_name(ch) for ch in channels]
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": params,
            "id": int(datetime.now().timestamp() * 1000)
        }

        await self._ws.send(json.dumps(subscribe_msg))
        self.logger.info(f"Subscribed to: {params}")

    async def unsubscribe(self, channels: List[str]) -> None:
        """
        Unsubscribe from channels.

        Args:
            channels: List of channels to unsubscribe
        """
        params = [self._get_stream_name(ch) for ch in channels]
        unsubscribe_msg = {
            "method": "UNSUBSCRIBE",
            "params": params,
            "id": int(datetime.now().timestamp() * 1000)
        }

        await self._ws.send(json.dumps(unsubscribe_msg))
        self.logger.info(f"Unsubscribed from: {params}")

    async def close(self) -> None:
        """Close WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
        self.logger.info("Closed Binance WebSocket")

    def get_buffered_messages(self) -> List[WebSocketMessage]:
        """Get all buffered messages."""
        return list(self._message_buffer)


class BybitWebSocketStream:
    """
    Bybit WebSocket stream handler.

    Handles real-time data streaming for OHLCV, trades, and orderbook.
    """

    PUBLIC_WS_URL = "wss://stream.bybit.com/v5/public/linear"
    PRIVATE_WS_URL = "wss://stream.bybit.com/v5/private"

    def __init__(
        self,
        symbol: str,
        category: str = 'linear',
        on_message: Optional[Callable[[WebSocketMessage], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None
    ):
        """
        Initialize Bybit WebSocket stream.

        Args:
            symbol: Trading symbol
            category: Market category (linear, inverse, spot)
            on_message: Message callback
            on_error: Error callback
        """
        self.symbol = symbol
        self.category = category
        self.on_message = on_message
        self.on_error = on_error

        self.logger = get_logger('bybit_ws')
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False

        # Message buffer
        self._message_buffer: deque = deque(maxlen=10000)

    async def connect(
        self,
        channels: List[str] = None
    ) -> None:
        """
        Connect to WebSocket.

        Args:
            channels: List of channels to subscribe
                     (kline.1, publicTrade, orderbook.50, tickers)
        """
        channels = channels or ['kline.1', 'publicTrade', 'orderbook.50']

        # Determine URL based on category
        if self.category == 'spot':
            ws_url = "wss://stream.bybit.com/v5/public/spot"
        else:
            ws_url = self.PUBLIC_WS_URL

        self._running = True

        try:
            self._ws = await websockets.connect(ws_url)
            self.logger.info(f"Connected to Bybit WebSocket: {self.symbol}")

            # Subscribe to channels
            await self._subscribe(channels)

            # Start ping task
            asyncio.create_task(self._ping_loop())

            # Listen for messages
            await self._listen()

        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
            if self.on_error:
                self.on_error(e)

    async def _subscribe(self, channels: List[str]) -> None:
        """Subscribe to channels."""
        topics = [f"{ch}.{self.symbol}" for ch in channels]

        subscribe_msg = {
            "op": "subscribe",
            "args": topics
        }

        await self._ws.send(json.dumps(subscribe_msg))
        self.logger.info(f"Subscribed to: {topics}")

    async def _ping_loop(self) -> None:
        """Send ping to keep connection alive."""
        while self._running:
            try:
                await self._ws.send(json.dumps({"op": "ping"}))
                await asyncio.sleep(20)  # Ping every 20 seconds
            except Exception:
                break

    async def _listen(self) -> None:
        """Listen for incoming messages."""
        while self._running:
            try:
                message = await self._ws.recv()
                data = json.loads(message)

                # Handle pong
                if data.get('op') == 'pong':
                    continue

                # Handle subscription confirmation
                if data.get('op') == 'subscribe':
                    continue

                # Parse data message
                topic = data.get('topic', '')

                if 'kline' in topic:
                    msg = self._parse_kline(data)
                elif 'publicTrade' in topic:
                    msg = self._parse_trade(data)
                elif 'orderbook' in topic:
                    msg = self._parse_orderbook(data)
                else:
                    continue

                if msg and self.on_message:
                    self._message_buffer.append(msg)
                    self.on_message(msg)

            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("WebSocket connection closed, reconnecting...")
                await asyncio.sleep(1)
                # Reconnect
                channels = ['kline.1', 'publicTrade', 'orderbook.50']
                await self.connect(channels)

            except Exception as e:
                self.logger.error(f"Error processing message: {e}")

    def _parse_kline(self, data: Dict) -> WebSocketMessage:
        """Parse kline data."""
        kline_data = data['data'][0]
        return WebSocketMessage(
            exchange='bybit',
            channel='kline',
            data={
                'symbol': self.symbol,  # Use stored symbol instead of from data
                'timeframe': data['topic'].split('.')[1],
                'open_time': from_timestamp(kline_data['start']),
                'close_time': from_timestamp(kline_data['end']),
                'open': float(kline_data['open']),
                'high': float(kline_data['high']),
                'low': float(kline_data['low']),
                'close': float(kline_data['close']),
                'volume': float(kline_data['volume']),
                'is_closed': kline_data['confirm']
            },
            timestamp=datetime.utcnow()
        )

    def _parse_trade(self, data: Dict) -> WebSocketMessage:
        """Parse trade data."""
        trades = data['data']
        return WebSocketMessage(
            exchange='bybit',
            channel='trade',
            data={
                'symbol': self.symbol,
                'trades': [{
                    'trade_id': t['i'],
                    'price': float(t['p']),
                    'quantity': float(t['v']),
                    'timestamp': from_timestamp(t['T']),
                    'side': t['S'].lower()
                } for t in trades]
            },
            timestamp=datetime.utcnow()
        )

    def _parse_orderbook(self, data: Dict) -> WebSocketMessage:
        """Parse orderbook data."""
        ob_data = data['data']
        return WebSocketMessage(
            exchange='bybit',
            channel='orderbook',
            data={
                'symbol': self.symbol,
                'bids': [[float(p), float(q)] for p, q in ob_data.get('b', [])],
                'asks': [[float(p), float(q)] for p, q in ob_data.get('a', [])],
                'update_id': ob_data.get('u', 0),
                'sequence': ob_data.get('seq', 0)
            },
            timestamp=datetime.utcnow()
        )

    async def close(self) -> None:
        """Close WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
        self.logger.info("Closed Bybit WebSocket")


class MultiExchangeStreamManager:
    """
    Manager for multiple WebSocket streams from different exchanges.
    """

    def __init__(
        self,
        symbols: List[str] = None,
        exchanges: List[str] = None
    ):
        """
        Initialize multi-exchange stream manager.

        Args:
            symbols: List of trading symbols
            exchanges: List of exchanges
        """
        self.symbols = symbols or ['ETHUSDT']
        self.exchanges = exchanges or ['binance', 'bybit']

        self.logger = get_logger('stream_manager')
        self._streams: Dict[str, Any] = {}
        self._message_handlers: List[Callable] = []

    def add_message_handler(self, handler: Callable) -> None:
        """
        Add message handler.

        Args:
            handler: Message handler function
        """
        self._message_handlers.append(handler)

    def _on_message(self, msg: WebSocketMessage) -> None:
        """Handle incoming message from any stream."""
        for handler in self._message_handlers:
            try:
                handler(msg)
            except Exception as e:
                self.logger.error(f"Handler error: {e}")

    async def start_all(self) -> None:
        """Start all WebSocket streams."""
        tasks = []

        for symbol in self.symbols:
            if 'binance' in self.exchanges:
                stream = BinanceWebSocketStream(
                    symbol=symbol,
                    on_message=self._on_message
                )
                self._streams[f'binance_{symbol}'] = stream
                tasks.append(stream.connect())

            if 'bybit' in self.exchanges:
                stream = BybitWebSocketStream(
                    symbol=symbol,
                    on_message=self._on_message
                )
                self._streams[f'bybit_{symbol}'] = stream
                tasks.append(stream.connect())

        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all(self) -> None:
        """Stop all WebSocket streams."""
        for stream in self._streams.values():
            await stream.close()

        self._streams.clear()
        self.logger.info("Stopped all WebSocket streams")

    def get_stream(self, exchange: str, symbol: str) -> Optional[Any]:
        """Get specific stream."""
        key = f"{exchange}_{symbol}"
        return self._streams.get(key)
