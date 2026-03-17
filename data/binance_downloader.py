"""
Binance Data Downloader for Quant Research Lab.
Handles historical and real-time data collection from Binance exchange.
"""

import requests
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import time
import asyncio
import aiohttp

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.time_utils import to_timestamp, from_timestamp


class BinanceDownloader:
    """
    Binance exchange data downloader.

    Downloads historical OHLCV, orderbook, trades, and funding rate data
    from Binance REST API.
    """

    BASE_URL = "https://api.binance.com"
    FUTURES_URL = "https://fapi.binance.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        rate_limit: int = 1200,  # requests per minute
        use_futures: bool = True
    ):
        """
        Initialize Binance downloader.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            rate_limit: Maximum requests per minute
            use_futures: Use futures API (default True)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.rate_limit = rate_limit
        self.use_futures = use_futures

        self.base_url = self.FUTURES_URL if use_futures else self.BASE_URL
        self.logger = get_logger('binance_downloader')
        self._last_request_time = 0
        self._request_interval = 60.0 / rate_limit

        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'X-MBX-APIKEY': api_key})

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make API request with rate limiting.

        Args:
            endpoint: API endpoint
            params: Request parameters

        Returns:
            JSON response
        """
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_interval:
            time.sleep(self._request_interval - elapsed)

        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            self._last_request_time = time.time()

            if response.status_code == 429:
                # Rate limit hit, wait and retry
                retry_after = int(response.headers.get('Retry-After', 60))
                self.logger.warning(f"Rate limit hit, waiting {retry_after}s")
                time.sleep(retry_after)
                return self._make_request(endpoint, params)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise

    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange information.

        Returns:
            Exchange info dictionary
        """
        if self.use_futures:
            endpoint = "/fapi/v1/exchangeInfo"
        else:
            endpoint = "/api/v3/exchangeInfo"

        return self._make_request(endpoint)

    def get_server_time(self) -> datetime:
        """
        Get server time.

        Returns:
            Server datetime
        """
        endpoint = "/api/v3/time"
        result = self._make_request(endpoint)
        return from_timestamp(result['serverTime'])

    def get_symbols(self) -> List[str]:
        """
        Get list of available trading symbols.

        Returns:
            List of symbol strings
        """
        info = self.get_exchange_info()
        symbols = [s['symbol'] for s in info.get('symbols', [])]
        return symbols

    def download_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1m',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1500
    ) -> pd.DataFrame:
        """
        Download OHLCV (candlestick) data.

        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            timeframe: Candlestick interval (1m, 5m, 15m, 1h, 4h, 1d)
            start_time: Start datetime
            end_time: End datetime
            limit: Maximum candles per request (max 1500)

        Returns:
            DataFrame with OHLCV data
        """
        if self.use_futures:
            endpoint = "/fapi/v1/klines"
        else:
            endpoint = "/api/v3/klines"

        all_data = []
        current_start = start_time
        current_end = end_time or datetime.utcnow()

        while True:
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': limit
            }

            if current_start:
                params['startTime'] = to_timestamp(current_start)
            if end_time:
                params['endTime'] = to_timestamp(end_time)

            try:
                klines = self._make_request(endpoint, params)

                if not klines:
                    break

                all_data.extend(klines)

                # Check if we've reached the end
                if len(klines) < limit:
                    break

                # Update start time for next batch
                last_timestamp = klines[-1][0]
                current_start = from_timestamp(last_timestamp + 1)

                if end_time and current_start >= end_time:
                    break

            except Exception as e:
                self.logger.error(f"Error downloading OHLCV: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        self.logger.info(
            f"Downloaded {len(df)} candles for {symbol} {timeframe}"
        )

        return df

    def download_recent_trades(
        self,
        symbol: str,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Download recent trades.

        Args:
            symbol: Trading symbol
            limit: Maximum trades to download (max 1000)

        Returns:
            DataFrame with trade data
        """
        if self.use_futures:
            endpoint = "/fapi/v1/aggTrades"
        else:
            endpoint = "/api/v3/aggTrades"

        params = {
            'symbol': symbol,
            'limit': min(limit, 1000)
        }

        trades = self._make_request(endpoint, params)

        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)
        df.columns = ['trade_id', 'price', 'quantity', 'first_trade_id',
                      'last_trade_id', 'timestamp', 'is_buyer_maker', 'is_best_match']

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['price'] = df['price'].astype(float)
        df['quantity'] = df['quantity'].astype(float)
        df['side'] = df['is_buyer_maker'].apply(lambda x: 'sell' if x else 'buy')

        df = df[['trade_id', 'timestamp', 'price', 'quantity', 'side']]

        self.logger.info(f"Downloaded {len(df)} trades for {symbol}")

        return df

    def get_orderbook(
        self,
        symbol: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get current orderbook snapshot.

        Args:
            symbol: Trading symbol
            limit: Depth limit (5, 10, 20, 50, 100, 500, 1000)

        Returns:
            Orderbook dictionary with bids and asks
        """
        if self.use_futures:
            endpoint = "/fapi/v1/depth"
        else:
            endpoint = "/api/v3/depth"

        params = {
            'symbol': symbol,
            'limit': limit
        }

        result = self._make_request(endpoint, params)

        orderbook = {
            'symbol': symbol,
            'timestamp': datetime.utcnow(),
            'bids': [[float(p), float(q)] for p, q in result.get('bids', [])],
            'asks': [[float(p), float(q)] for p, q in result.get('asks', [])]
        }

        return orderbook

    def get_funding_rate(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get historical funding rates.

        Args:
            symbol: Trading symbol
            start_time: Start datetime
            end_time: End datetime
            limit: Maximum records

        Returns:
            DataFrame with funding rate data
        """
        if not self.use_futures:
            self.logger.warning("Funding rates only available for futures")
            return pd.DataFrame()

        endpoint = "/fapi/v1/fundingRate"

        params = {
            'symbol': symbol,
            'limit': limit
        }

        if start_time:
            params['startTime'] = to_timestamp(start_time)
        if end_time:
            params['endTime'] = to_timestamp(end_time)

        result = self._make_request(endpoint, params)

        if not result:
            return pd.DataFrame()

        df = pd.DataFrame(result)
        df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df['funding_rate'] = df['fundingRate'].astype(float)
        df['symbol'] = symbol

        df = df[['symbol', 'timestamp', 'funding_rate']]

        self.logger.info(f"Downloaded {len(df)} funding rates for {symbol}")

        return df

    def get_ticker(
        self,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get current ticker information.

        Args:
            symbol: Trading symbol (optional, returns all if not provided)

        Returns:
            Ticker information dictionary
        """
        if self.use_futures:
            endpoint = "/fapi/v1/ticker/24hr"
        else:
            endpoint = "/api/v3/ticker/24hr"

        params = {}
        if symbol:
            params['symbol'] = symbol

        result = self._make_request(endpoint, params)

        if isinstance(result, list) and len(result) == 1:
            return result[0]
        return result

    def get_open_interest(
        self,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Get open interest for futures symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Open interest information
        """
        if not self.use_futures:
            self.logger.warning("Open interest only available for futures")
            return {}

        endpoint = "/fapi/v1/openInterest"

        params = {'symbol': symbol}
        result = self._make_request(endpoint, params)

        return {
            'symbol': symbol,
            'open_interest': float(result.get('openInterest', 0)),
            'timestamp': datetime.utcnow()
        }

    def download_historical_trades(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Download historical trades with pagination.

        Args:
            symbol: Trading symbol
            start_time: Start datetime
            end_time: End datetime
            limit: Maximum trades per request

        Returns:
            DataFrame with trade data
        """
        all_trades = []
        from_id = None

        while True:
            try:
                trades = self._download_trades_page(symbol, from_id, limit)

                if not trades:
                    break

                # Filter by time range
                for trade in trades:
                    trade_time = from_timestamp(trade['time'])
                    if start_time and trade_time < start_time:
                        continue
                    if end_time and trade_time > end_time:
                        return self._process_trades(all_trades)

                    all_trades.append({
                        'trade_id': trade['a'],
                        'timestamp': trade_time,
                        'price': float(trade['p']),
                        'quantity': float(trade['q']),
                        'side': 'sell' if trade['m'] else 'buy'
                    })

                # Update from_id for pagination
                from_id = trades[-1]['a'] - 1

                if len(trades) < limit:
                    break

            except Exception as e:
                self.logger.error(f"Error downloading historical trades: {e}")
                break

        return self._process_trades(all_trades)

    def _download_trades_page(
        self,
        symbol: str,
        from_id: Optional[int] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """Download a single page of historical trades."""
        if self.use_futures:
            endpoint = "/fapi/v1/aggTrades"
        else:
            endpoint = "/api/v3/aggTrades"

        params = {
            'symbol': symbol,
            'limit': limit
        }

        if from_id:
            params['fromId'] = from_id

        return self._make_request(endpoint, params)

    def _process_trades(self, trades: List[Dict]) -> pd.DataFrame:
        """Process trades list into DataFrame."""
        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)
        df = df.drop_duplicates(subset=['trade_id'])
        df = df.sort_values('timestamp')
        df = df.reset_index(drop=True)

        return df

    def close(self) -> None:
        """Close the session."""
        self.session.close()


class BinanceAsyncDownloader:
    """
    Async version of Binance downloader for high-performance data collection.
    """

    BASE_URL = "https://api.binance.com"
    FUTURES_URL = "https://fapi.binance.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_futures: bool = True
    ):
        """Initialize async downloader."""
        self.api_key = api_key
        self.use_futures = use_futures
        self.base_url = self.FUTURES_URL if use_futures else self.BASE_URL
        self.logger = get_logger('binance_async_downloader')
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {}
            if self.api_key:
                headers['X-MBX-APIKEY'] = self.api_key
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def download_ohlcv_multi(
        self,
        symbols: List[str],
        timeframe: str = '1m',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1500
    ) -> Dict[str, pd.DataFrame]:
        """
        Download OHLCV for multiple symbols concurrently.

        Args:
            symbols: List of trading symbols
            timeframe: Candlestick interval
            start_time: Start datetime
            end_time: End datetime
            limit: Maximum candles per request

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        tasks = [
            self._download_single_ohlcv(
                symbol, timeframe, start_time, end_time, limit
            )
            for symbol in symbols
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error downloading {symbol}: {result}")
            else:
                data[symbol] = result

        return data

    async def _download_single_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        limit: int
    ) -> pd.DataFrame:
        """Download OHLCV for a single symbol."""
        session = await self._get_session()

        endpoint = "/fapi/v1/klines" if self.use_futures else "/api/v3/klines"

        params = {
            'symbol': symbol,
            'interval': timeframe,
            'limit': limit
        }

        if start_time:
            params['startTime'] = to_timestamp(start_time)
        if end_time:
            params['endTime'] = to_timestamp(end_time)

        url = f"{self.base_url}{endpoint}"

        async with session.get(url, params=params) as response:
            response.raise_for_status()
            klines = await response.json()

        if not klines:
            return pd.DataFrame()

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    async def close(self) -> None:
        """Close the async session."""
        if self._session and not self._session.closed:
            await self._session.close()
