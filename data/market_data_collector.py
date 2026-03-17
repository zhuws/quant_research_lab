"""
Market Data Collector for Quant Research Lab.
Coordinates data collection from multiple exchanges.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger, TradingLogger
from utils.time_utils import get_utc_now, align_to_timeframe
from data.mysql_storage import MySQLStorage
from data.binance_downloader import BinanceDownloader
from data.bybit_downloader import BybitDownloader


class MarketDataCollector:
    """
    Centralized market data collection manager.

    Coordinates data collection from multiple exchanges and stores
    data in the database.
    """

    def __init__(
        self,
        storage: MySQLStorage,
        symbols: List[str] = None,
        exchanges: List[str] = None,
        timeframes: List[str] = None,
        binance_api_key: Optional[str] = None,
        binance_api_secret: Optional[str] = None,
        bybit_api_key: Optional[str] = None,
        bybit_api_secret: Optional[str] = None
    ):
        """
        Initialize market data collector.

        Args:
            storage: MySQL storage instance
            symbols: List of trading symbols
            exchanges: List of exchanges to collect from
            timeframes: List of timeframes to collect
            binance_api_key: Binance API key
            binance_api_secret: Binance API secret
            bybit_api_key: Bybit API key
            bybit_api_secret: Bybit API secret
        """
        self.storage = storage
        self.symbols = symbols or ['ETHUSDT']
        self.exchanges = exchanges or ['binance', 'bybit']
        self.timeframes = timeframes or ['1m']

        self.logger = get_logger('market_data_collector')
        self.trading_logger = TradingLogger('data_collection')

        # Initialize downloaders
        self.downloaders = {}
        if 'binance' in self.exchanges:
            self.downloaders['binance'] = BinanceDownloader(
                api_key=binance_api_key,
                api_secret=binance_api_secret
            )
        if 'bybit' in self.exchanges:
            self.downloaders['bybit'] = BybitDownloader(
                api_key=bybit_api_key,
                api_secret=bybit_api_secret
            )

        self._running = False
        self._collection_thread: Optional[threading.Thread] = None

    def download_historical_ohlcv(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Download historical OHLCV data.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Candle timeframe
            start_date: Start date
            end_date: End date (default: now)

        Returns:
            DataFrame with OHLCV data
        """
        if exchange not in self.downloaders:
            self.logger.error(f"Unknown exchange: {exchange}")
            return pd.DataFrame()

        downloader = self.downloaders[exchange]
        self.logger.info(
            f"Downloading {symbol} {timeframe} from {exchange} "
            f"from {start_date} to {end_date or 'now'}"
        )

        try:
            # Map timeframe for Bybit
            tf = timeframe
            if exchange == 'bybit':
                tf = self._map_timeframe_to_bybit(timeframe)

            df = downloader.download_ohlcv(
                symbol=symbol,
                timeframe=tf,
                start_time=start_date,
                end_time=end_date
            )

            if df.empty:
                self.logger.warning(f"No data returned for {symbol}")
                return df

            # Store in database
            self.storage.insert_ohlcv(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                data=df
            )

            self.logger.info(f"Stored {len(df)} candles for {symbol}")
            return df

        except Exception as e:
            self.logger.error(f"Error downloading data: {e}")
            return pd.DataFrame()

    def _map_timeframe_to_bybit(self, timeframe: str) -> str:
        """Map standard timeframe to Bybit format."""
        mapping = {
            '1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30',
            '1h': '60', '2h': '120', '4h': '240', '6h': '360', '12h': '720',
            '1d': 'D', '1w': 'W', '1M': 'M'
        }
        return mapping.get(timeframe, timeframe)

    def download_all_historical(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Download historical data for all symbols and exchanges.

        Args:
            start_date: Start date
            end_date: End date (default: now)

        Returns:
            Dictionary of DataFrames by symbol and exchange
        """
        results = {}

        for symbol in self.symbols:
            results[symbol] = {}

            for exchange in self.exchanges:
                for timeframe in self.timeframes:
                    df = self.download_historical_ohlcv(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )
                    results[symbol][f"{exchange}_{timeframe}"] = df

        return results

    def collect_orderbook_snapshot(
        self,
        symbol: str,
        exchange: str
    ) -> Dict[str, Any]:
        """
        Collect current orderbook snapshot.

        Args:
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Orderbook dictionary
        """
        if exchange not in self.downloaders:
            self.logger.error(f"Unknown exchange: {exchange}")
            return {}

        downloader = self.downloaders[exchange]

        try:
            orderbook = downloader.get_orderbook(symbol)

            # Store in database
            self.storage.insert_orderbook(
                exchange=exchange,
                symbol=symbol,
                timestamp=orderbook['timestamp'],
                bids=orderbook['bids'],
                asks=orderbook['asks']
            )

            return orderbook

        except Exception as e:
            self.logger.error(f"Error collecting orderbook: {e}")
            return {}

    def collect_recent_trades(
        self,
        symbol: str,
        exchange: str,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Collect recent trades.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            limit: Maximum trades to collect

        Returns:
            DataFrame with trades
        """
        if exchange not in self.downloaders:
            self.logger.error(f"Unknown exchange: {exchange}")
            return pd.DataFrame()

        downloader = self.downloaders[exchange]

        try:
            trades = downloader.download_recent_trades(symbol, limit)

            if not trades.empty:
                # Store in database
                trade_records = trades.to_dict('records')
                self.storage.insert_trades(
                    exchange=exchange,
                    symbol=symbol,
                    trades=trade_records
                )

            return trades

        except Exception as e:
            self.logger.error(f"Error collecting trades: {e}")
            return pd.DataFrame()

    def collect_funding_rates(
        self,
        symbol: str,
        exchange: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Collect funding rate history.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with funding rates
        """
        if exchange not in self.downloaders:
            self.logger.error(f"Unknown exchange: {exchange}")
            return pd.DataFrame()

        downloader = self.downloaders[exchange]

        try:
            funding = downloader.get_funding_rate(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time
            )

            if not funding.empty:
                for _, row in funding.iterrows():
                    self.storage.insert_funding_rate(
                        exchange=exchange,
                        symbol=symbol,
                        timestamp=row['timestamp'],
                        funding_rate=row['funding_rate'],
                        funding_time=row['timestamp']
                    )

            return funding

        except Exception as e:
            self.logger.error(f"Error collecting funding rates: {e}")
            return pd.DataFrame()

    def update_latest_data(self) -> None:
        """
        Update latest data for all symbols and exchanges.

        Fetches data from the last stored timestamp to now.
        """
        self.logger.info("Updating latest data for all symbols")

        for symbol in self.symbols:
            for exchange in self.exchanges:
                for timeframe in self.timeframes:
                    try:
                        # Get last stored timestamp
                        last_ts = self.storage.get_latest_timestamp(
                            table='ohlcv',
                            exchange=exchange,
                            symbol=symbol,
                            timeframe=timeframe
                        )

                        if last_ts:
                            start_time = last_ts + timedelta(minutes=1)
                        else:
                            # No existing data, get last 7 days
                            start_time = get_utc_now() - timedelta(days=7)

                        self.download_historical_ohlcv(
                            symbol=symbol,
                            exchange=exchange,
                            timeframe=timeframe,
                            start_date=start_time
                        )

                    except Exception as e:
                        self.logger.error(
                            f"Error updating {symbol} {exchange}: {e}"
                        )

    def start_continuous_collection(
        self,
        interval_seconds: int = 60
    ) -> None:
        """
        Start continuous data collection.

        Args:
            interval_seconds: Collection interval in seconds
        """
        if self._running:
            self.logger.warning("Collection already running")
            return

        self._running = True

        def collection_loop():
            while self._running:
                try:
                    self.update_latest_data()
                except Exception as e:
                    self.logger.error(f"Collection error: {e}")

                # Wait for next interval
                for _ in range(interval_seconds):
                    if not self._running:
                        break
                    time.sleep(1)

        self._collection_thread = threading.Thread(
            target=collection_loop,
            daemon=True
        )
        self._collection_thread.start()

        self.logger.info(f"Started continuous collection (interval: {interval_seconds}s)")

    def stop_continuous_collection(self) -> None:
        """Stop continuous data collection."""
        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        self.logger.info("Stopped continuous collection")

    def get_status(self) -> Dict[str, Any]:
        """
        Get collection status.

        Returns:
            Status dictionary
        """
        status = {
            'running': self._running,
            'symbols': self.symbols,
            'exchanges': self.exchanges,
            'timeframes': self.timeframes,
            'data_counts': {}
        }

        for symbol in self.symbols:
            status['data_counts'][symbol] = {}
            for exchange in self.exchanges:
                count = self.storage.get_data_count(
                    table='ohlcv',
                    exchange=exchange,
                    symbol=symbol,
                    timeframe='1m'
                )
                status['data_counts'][symbol][exchange] = count

        return status

    def collect_cross_exchange_data(
        self,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Collect data from multiple exchanges for cross-exchange analysis.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with data from all exchanges
        """
        data = {
            'symbol': symbol,
            'timestamp': get_utc_now(),
            'prices': {},
            'orderbooks': {}
        }

        for exchange in self.exchanges:
            try:
                # Get ticker
                ticker = self.downloaders[exchange].get_ticker(symbol)
                data['prices'][exchange] = ticker.get('last_price', 0)

                # Get orderbook
                orderbook = self.collect_orderbook_snapshot(symbol, exchange)
                data['orderbooks'][exchange] = orderbook

            except Exception as e:
                self.logger.error(f"Error collecting cross-exchange data: {e}")

        return data

    def calculate_cross_exchange_spread(
        self,
        symbol: str
    ) -> Dict[str, float]:
        """
        Calculate cross-exchange price spread.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with spread metrics
        """
        data = self.collect_cross_exchange_data(symbol)
        prices = data['prices']

        if len(prices) < 2:
            return {'spread': 0, 'spread_pct': 0}

        exchanges = list(prices.keys())
        price1 = prices[exchanges[0]]
        price2 = prices[exchanges[1]]

        spread = abs(price1 - price2)
        spread_pct = spread / min(price1, price2) * 100 if min(price1, price2) > 0 else 0

        return {
            'spread': spread,
            'spread_pct': spread_pct,
            'prices': prices
        }

    def close(self) -> None:
        """Close all connections."""
        self.stop_continuous_collection()

        for downloader in self.downloaders.values():
            downloader.close()

        self.logger.info("Closed market data collector")


class ScheduledDataCollector:
    """
    Scheduled data collector with configurable collection tasks.
    """

    def __init__(
        self,
        collector: MarketDataCollector,
        schedule_config: Dict[str, Any] = None
    ):
        """
        Initialize scheduled collector.

        Args:
            collector: Market data collector instance
            schedule_config: Schedule configuration
        """
        self.collector = collector
        self.schedule_config = schedule_config or {}
        self.logger = get_logger('scheduled_collector')
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None

    def add_task(
        self,
        task_name: str,
        task_func: callable,
        interval_seconds: int
    ) -> None:
        """
        Add a scheduled collection task.

        Args:
            task_name: Task identifier
            task_func: Task function
            interval_seconds: Execution interval
        """
        self.schedule_config[task_name] = {
            'func': task_func,
            'interval': interval_seconds,
            'last_run': None
        }

    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return

        self._running = True

        def scheduler_loop():
            while self._running:
                now = time.time()

                for task_name, task in self.schedule_config.items():
                    last_run = task.get('last_run', 0) or 0
                    interval = task['interval']

                    if now - last_run >= interval:
                        try:
                            task['func']()
                            task['last_run'] = now
                        except Exception as e:
                            self.logger.error(f"Task {task_name} failed: {e}")

                time.sleep(1)

        self._scheduler_thread = threading.Thread(
            target=scheduler_loop,
            daemon=True
        )
        self._scheduler_thread.start()

        self.logger.info("Started scheduled data collection")

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        self.logger.info("Stopped scheduled data collection")
