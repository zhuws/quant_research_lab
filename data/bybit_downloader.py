"""
Bybit Data Downloader for Quant Research Lab.
Handles historical and real-time data collection from Bybit exchange.
"""

import requests
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import time
import hmac
import hashlib
import asyncio
import aiohttp

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.time_utils import to_timestamp, from_timestamp


class BybitDownloader:
    """
    Bybit exchange data downloader.

    Downloads historical OHLCV, orderbook, trades, and funding rate data
    from Bybit REST API.
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        rate_limit: int = 100,  # requests per second
        category: str = 'linear'  # linear, inverse, spot
    ):
        """
        Initialize Bybit downloader.

        Args:
            api_key: Bybit API key
            api_secret: Bybit API secret
            rate_limit: Maximum requests per second
            category: Market category (linear, inverse, spot)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.rate_limit = rate_limit
        self.category = category

        self.logger = get_logger('bybit_downloader')
        self._last_request_time = 0
        self._request_interval = 1.0 / rate_limit

        self.session = requests.Session()

    def _generate_signature(self, params: Dict[str, Any], timestamp: int) -> str:
        """
        Generate API signature.

        Args:
            params: Request parameters
            timestamp: Request timestamp

        Returns:
            Signature string
        """
        if not self.api_key or not self.api_secret:
            return ""

        param_str = str(timestamp) + self.api_key + "5000" + "&".join(
            f"{k}={v}" for k, v in sorted(params.items())
        )

        return hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False
    ) -> Dict[str, Any]:
        """
        Make API request with rate limiting.

        Args:
            endpoint: API endpoint
            params: Request parameters
            signed: Whether request requires signature

        Returns:
            JSON response
        """
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_interval:
            time.sleep(self._request_interval - elapsed)

        if params is None:
            params = {}

        url = f"{self.BASE_URL}{endpoint}"
        headers = {}

        if signed and self.api_key:
            timestamp = int(time.time() * 1000)
            params['api_key'] = self.api_key
            params['timestamp'] = timestamp
            params['recv_window'] = 5000
            signature = self._generate_signature(params, timestamp)
            params['sign'] = signature

        try:
            response = self.session.get(url, params=params, timeout=30)
            self._last_request_time = time.time()

            response.raise_for_status()
            data = response.json()

            if data.get('retCode', 0) != 0:
                raise Exception(f"API error: {data.get('retMsg', 'Unknown error')}")

            return data.get('result', {})

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise

    def get_server_time(self) -> datetime:
        """
        Get server time.

        Returns:
            Server datetime
        """
        result = self._make_request("/v5/public/time")
        return from_timestamp(result['timeSecond'] * 1000)

    def get_symbols(self) -> List[str]:
        """
        Get list of available trading symbols.

        Returns:
            List of symbol strings
        """
        params = {'category': self.category}
        result = self._make_request("/v5/public/instruments-info", params)

        symbols = [item['symbol'] for item in result.get('list', [])]
        return symbols

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol information.

        Args:
            symbol: Trading symbol

        Returns:
            Symbol info dictionary
        """
        params = {
            'category': self.category,
            'symbol': symbol
        }
        result = self._make_request("/v5/public/instruments-info", params)

        if result.get('list'):
            return result['list'][0]
        return {}

    def download_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Download OHLCV (kline) data.

        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            timeframe: Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            start_time: Start datetime
            end_time: End datetime
            limit: Maximum candles per request (max 1000)

        Returns:
            DataFrame with OHLCV data
        """
        endpoint = "/v5/market/kline"

        all_data = []
        current_end = end_time or datetime.utcnow()

        while True:
            params = {
                'category': self.category,
                'symbol': symbol,
                'interval': timeframe,
                'limit': limit
            }

            if start_time:
                params['start'] = to_timestamp(start_time)
            if current_end:
                params['end'] = to_timestamp(current_end)

            try:
                result = self._make_request(endpoint, params)
                klines = result.get('list', [])

                if not klines:
                    break

                # Bybit returns newest first, reverse to chronological
                klines = klines[::-1]
                all_data.extend(klines)

                if len(klines) < limit:
                    break

                # Update end time for next batch
                oldest_timestamp = int(klines[0][0])
                current_end = from_timestamp(oldest_timestamp - 1)

                if start_time and current_end < start_time:
                    break

            except Exception as e:
                self.logger.error(f"Error downloading OHLCV: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        # Bybit format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.drop_duplicates(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        self.logger.info(
            f"Downloaded {len(df)} candles for {symbol} {timeframe}"
        )

        return df

    def download_recent_trades(
        self,
        symbol: str,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Download recent trades.

        Args:
            symbol: Trading symbol
            limit: Maximum trades to download (max 500 for public API)

        Returns:
            DataFrame with trade data
        """
        endpoint = "/v5/public/recent-trade"

        params = {
            'category': self.category,
            'symbol': symbol,
            'limit': min(limit, 500)
        }

        result = self._make_request(endpoint, params)
        trades = result.get('list', [])

        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)
        df['timestamp'] = pd.to_datetime(df['time'].astype(int), unit='ms')
        df['price'] = df['price'].astype(float)
        df['quantity'] = df['size'].astype(float)
        df['trade_id'] = df['execId']
        df['side'] = df['side'].str.lower()

        df = df[['trade_id', 'timestamp', 'price', 'quantity', 'side']]

        self.logger.info(f"Downloaded {len(df)} trades for {symbol}")

        return df

    def get_orderbook(
        self,
        symbol: str,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get current orderbook snapshot.

        Args:
            symbol: Trading symbol
            limit: Depth limit (1, 50, 200, 500)

        Returns:
            Orderbook dictionary with bids and asks
        """
        endpoint = "/v5/public/orderbook"

        params = {
            'category': self.category,
            'symbol': symbol,
            'limit': limit
        }

        result = self._make_request(endpoint, params)

        orderbook = {
            'symbol': symbol,
            'timestamp': datetime.utcnow(),
            'bids': [[float(p), float(q)] for p, q in result.get('b', [])],
            'asks': [[float(p), float(q)] for p, q in result.get('a', [])]
        }

        return orderbook

    def get_funding_rate(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 200
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
        if self.category not in ['linear', 'inverse']:
            self.logger.warning("Funding rates only available for linear/inverse")
            return pd.DataFrame()

        endpoint = "/v5/public/funding/history-funding-rate"

        all_data = []

        while True:
            params = {
                'category': self.category,
                'symbol': symbol,
                'limit': limit
            }

            if start_time:
                params['startTime'] = to_timestamp(start_time)
            if end_time:
                params['endTime'] = to_timestamp(end_time)

            try:
                result = self._make_request(endpoint, params)
                rates = result.get('list', [])

                if not rates:
                    break

                all_data.extend(rates)

                if len(rates) < limit:
                    break

                # Update end time for next batch
                oldest_time = int(rates[-1]['fundingRateTimestamp'])
                end_time = from_timestamp(oldest_time - 1)

                if start_time and end_time < start_time:
                    break

            except Exception as e:
                self.logger.error(f"Error downloading funding rates: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['fundingRateTimestamp'].astype(int), unit='ms')
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
            symbol: Trading symbol (optional)

        Returns:
            Ticker information dictionary
        """
        endpoint = "/v5/public/tickers"

        params = {'category': self.category}
        if symbol:
            params['symbol'] = symbol

        result = self._make_request(endpoint, params)

        tickers = result.get('list', [])

        if symbol and tickers:
            ticker = tickers[0]
            return {
                'symbol': ticker['symbol'],
                'last_price': float(ticker['lastPrice']),
                'volume_24h': float(ticker['volume24h']),
                'high_24h': float(ticker['highPrice24h']),
                'low_24h': float(ticker['lowPrice24h']),
                'funding_rate': float(ticker.get('fundingRate', 0)),
                'open_interest': float(ticker.get('openInterest', 0))
            }

        return {'tickers': tickers}

    def get_open_interest(
        self,
        symbol: str,
        interval: str = '1h',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 200
    ) -> pd.DataFrame:
        """
        Get historical open interest.

        Args:
            symbol: Trading symbol
            interval: Time interval (5min, 15min, 30min, 1h, 4h, 1d)
            start_time: Start datetime
            end_time: End datetime
            limit: Maximum records

        Returns:
            DataFrame with open interest data
        """
        endpoint = "/v5/public/open-interest"

        params = {
            'category': self.category,
            'symbol': symbol,
            'intervalTime': interval,
            'limit': limit
        }

        if start_time:
            params['startTime'] = to_timestamp(start_time)
        if end_time:
            params['endTime'] = to_timestamp(end_time)

        result = self._make_request(endpoint, params)

        records = result.get('list', [])

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df['open_interest'] = df['openInterest'].astype(float)

        df = df[['timestamp', 'open_interest']]

        return df

    def get_long_short_ratio(
        self,
        symbol: str,
        period: str = '1h',
        limit: int = 50
    ) -> pd.DataFrame:
        """
        Get long/short ratio.

        Args:
            symbol: Trading symbol
            period: Time period (5min, 15min, 30min, 1h, 4h, 1d)
            limit: Maximum records

        Returns:
            DataFrame with long/short ratio data
        """
        endpoint = "/v5/public/account/ratio"

        params = {
            'category': self.category,
            'symbol': symbol,
            'period': period,
            'limit': limit
        }

        result = self._make_request(endpoint, params)
        records = result.get('list', [])

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df['long_ratio'] = df['longRatio'].astype(float)
        df['short_ratio'] = df['shortRatio'].astype(float)

        return df[['timestamp', 'long_ratio', 'short_ratio']]

    def close(self) -> None:
        """Close the session."""
        self.session.close()


class BybitAsyncDownloader:
    """
    Async version of Bybit downloader for high-performance data collection.
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        category: str = 'linear'
    ):
        """Initialize async downloader."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.category = category
        self.logger = get_logger('bybit_async_downloader')
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def download_ohlcv_multi(
        self,
        symbols: List[str],
        timeframe: str = '1',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> Dict[str, pd.DataFrame]:
        """
        Download OHLCV for multiple symbols concurrently.

        Args:
            symbols: List of trading symbols
            timeframe: Kline interval
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

        endpoint = "/v5/market/kline"

        params = {
            'category': self.category,
            'symbol': symbol,
            'interval': timeframe,
            'limit': limit
        }

        if start_time:
            params['start'] = to_timestamp(start_time)
        if end_time:
            params['end'] = to_timestamp(end_time)

        url = f"{self.BASE_URL}{endpoint}"

        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()

        if data.get('retCode', 0) != 0:
            raise Exception(f"API error: {data.get('retMsg')}")

        klines = data.get('result', {}).get('list', [])

        if not klines:
            return pd.DataFrame()

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    async def close(self) -> None:
        """Close the async session."""
        if self._session and not self._session.closed:
            await self._session.close()
