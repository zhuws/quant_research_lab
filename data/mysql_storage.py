"""
MySQL Storage Module for Quant Research Lab.
Provides database connection management and data persistence.
"""

import pymysql
from pymysql.cursors import DictCursor
from typing import Optional, List, Dict, Any, Union
from contextlib import contextmanager
import pandas as pd
import numpy as np
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger


class MySQLStorage:
    """
    MySQL database storage manager.

    Handles database connections, table creation, and data operations
    for the quantitative trading research platform.
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 3306,
        user: str = 'quant_user',
        password: str = 'quant_password',
        database: str = 'quant_research',
        charset: str = 'utf8mb4',
        autocommit: bool = True
    ):
        """
        Initialize MySQL storage.

        Args:
            host: Database host
            port: Database port
            user: Database user
            password: Database password
            database: Database name
            charset: Character set
            autocommit: Auto-commit transactions
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        self.autocommit = autocommit

        self.logger = get_logger('mysql_storage')
        self._connection: Optional[pymysql.Connection] = None

    def connect(self) -> None:
        """Establish database connection."""
        try:
            self._connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset=self.charset,
                autocommit=self.autocommit,
                cursorclass=DictCursor
            )
            self.logger.info(f"Connected to MySQL database: {self.database}")
        except pymysql.Error as e:
            self.logger.error(f"Failed to connect to MySQL: {e}")
            raise

    def disconnect(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            self.logger.info("Disconnected from MySQL database")

    @contextmanager
    def get_cursor(self):
        """
        Get database cursor context manager.

        Yields:
            Database cursor
        """
        if not self._connection:
            self.connect()

        cursor = self._connection.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def execute(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> int:
        """
        Execute a SQL query.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Number of affected rows
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount

    def execute_many(
        self,
        query: str,
        params_list: List[tuple]
    ) -> int:
        """
        Execute a SQL query with multiple parameter sets.

        Args:
            query: SQL query string
            params_list: List of parameter tuples

        Returns:
            Number of affected rows
        """
        with self.get_cursor() as cursor:
            cursor.executemany(query, params_list)
            return cursor.rowcount

    def fetch_one(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single row.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Row as dictionary or None
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()

    def fetch_all(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all rows.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of row dictionaries
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()

    def fetch_dataframe(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> pd.DataFrame:
        """
        Fetch query results as DataFrame.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            DataFrame with query results
        """
        results = self.fetch_all(query, params)
        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results)

    def create_tables(self) -> None:
        """Create all required tables."""
        tables = {
            'ohlcv': """
                CREATE TABLE IF NOT EXISTS ohlcv (
                    exchange VARCHAR(20) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    timestamp DATETIME(3) NOT NULL,
                    open DECIMAL(20, 8) NOT NULL,
                    high DECIMAL(20, 8) NOT NULL,
                    low DECIMAL(20, 8) NOT NULL,
                    close DECIMAL(20, 8) NOT NULL,
                    volume DECIMAL(30, 8) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (exchange, symbol, timeframe, timestamp),
                    INDEX idx_timestamp (timestamp),
                    INDEX idx_symbol_time (symbol, timestamp)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            'orderbook': """
                CREATE TABLE IF NOT EXISTS orderbook (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    exchange VARCHAR(20) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp DATETIME(3) NOT NULL,
                    bids JSON NOT NULL,
                    asks JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_symbol_time (symbol, timestamp),
                    INDEX idx_exchange_time (exchange, timestamp)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            'trades': """
                CREATE TABLE IF NOT EXISTS trades (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    exchange VARCHAR(20) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    trade_id VARCHAR(50) NOT NULL,
                    timestamp DATETIME(3) NOT NULL,
                    price DECIMAL(20, 8) NOT NULL,
                    quantity DECIMAL(30, 8) NOT NULL,
                    side ENUM('buy', 'sell') NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uk_trade (exchange, trade_id),
                    INDEX idx_symbol_time (symbol, timestamp)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            'funding_rates': """
                CREATE TABLE IF NOT EXISTS funding_rates (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    exchange VARCHAR(20) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp DATETIME(3) NOT NULL,
                    funding_rate DECIMAL(10, 8) NOT NULL,
                    funding_time DATETIME(3) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uk_funding (exchange, symbol, timestamp),
                    INDEX idx_symbol_time (symbol, timestamp)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            'features': """
                CREATE TABLE IF NOT EXISTS features (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp DATETIME(3) NOT NULL,
                    feature_name VARCHAR(100) NOT NULL,
                    feature_value DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uk_feature (symbol, timestamp, feature_name),
                    INDEX idx_symbol_time (symbol, timestamp)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            'signals': """
                CREATE TABLE IF NOT EXISTS signals (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    strategy VARCHAR(50) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp DATETIME(3) NOT NULL,
                    `signal` INT NOT NULL,
                    confidence DECIMAL(5, 4),
                    price DECIMAL(20, 8),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_strategy_time (strategy, timestamp),
                    INDEX idx_symbol_time (symbol, timestamp)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            'orders': """
                CREATE TABLE IF NOT EXISTS orders (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    exchange VARCHAR(20) NOT NULL,
                    order_id VARCHAR(100) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    side ENUM('buy', 'sell') NOT NULL,
                    order_type ENUM('market', 'limit', 'stop', 'stop_limit') NOT NULL,
                    quantity DECIMAL(30, 8) NOT NULL,
                    price DECIMAL(20, 8),
                    status ENUM('pending', 'open', 'filled', 'cancelled', 'rejected') NOT NULL,
                    filled_quantity DECIMAL(30, 8) DEFAULT 0,
                    average_price DECIMAL(20, 8),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY uk_order (exchange, order_id),
                    INDEX idx_symbol_time (symbol, created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            'positions': """
                CREATE TABLE IF NOT EXISTS positions (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    exchange VARCHAR(20) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    side ENUM('long', 'short') NOT NULL,
                    quantity DECIMAL(30, 8) NOT NULL,
                    entry_price DECIMAL(20, 8) NOT NULL,
                    current_price DECIMAL(20, 8),
                    unrealized_pnl DECIMAL(20, 8),
                    leverage DECIMAL(10, 2) DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY uk_position (exchange, symbol),
                    INDEX idx_symbol (symbol)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            'pnl': """
                CREATE TABLE IF NOT EXISTS pnl (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    strategy VARCHAR(50) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp DATETIME(3) NOT NULL,
                    realized_pnl DECIMAL(20, 8),
                    unrealized_pnl DECIMAL(20, 8),
                    total_pnl DECIMAL(20, 8),
                    drawdown DECIMAL(10, 6),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_strategy_time (strategy, timestamp),
                    INDEX idx_symbol_time (symbol, timestamp)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        }

        for table_name, create_sql in tables.items():
            try:
                self.execute(create_sql)
                self.logger.info(f"Created table: {table_name}")
            except pymysql.Error as e:
                self.logger.error(f"Failed to create table {table_name}: {e}")
                raise

    def drop_tables(self) -> None:
        """Drop all tables (use with caution)."""
        tables = [
            'pnl', 'positions', 'orders', 'signals', 'features',
            'funding_rates', 'trades', 'orderbook', 'ohlcv'
        ]

        for table in tables:
            try:
                self.execute(f"DROP TABLE IF EXISTS {table}")
                self.logger.info(f"Dropped table: {table}")
            except pymysql.Error as e:
                self.logger.error(f"Failed to drop table {table}: {e}")

    def insert_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame
    ) -> int:
        """
        Insert OHLCV data.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Timeframe
            data: DataFrame with columns: timestamp, open, high, low, close, volume

        Returns:
            Number of rows inserted
        """
        query = """
            INSERT INTO ohlcv
            (exchange, symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            open = VALUES(open),
            high = VALUES(high),
            low = VALUES(low),
            close = VALUES(close),
            volume = VALUES(volume)
        """

        params_list = []
        for _, row in data.iterrows():
            ts = row['timestamp']
            if isinstance(ts, str):
                ts = pd.to_datetime(ts)
            params_list.append((
                exchange, symbol, timeframe,
                ts,
                row['open'], row['high'], row['low'],
                row['close'], row['volume']
            ))

        return self.execute_many(query, params_list)

    def get_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV data.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Timeframe
            start_time: Start timestamp
            end_time: End timestamp
            limit: Maximum rows to return

        Returns:
            DataFrame with OHLCV data
        """
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE exchange = %s AND symbol = %s AND timeframe = %s
        """
        params = [exchange, symbol, timeframe]

        if start_time:
            query += " AND timestamp >= %s"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= %s"
            params.append(end_time)

        query += " ORDER BY timestamp ASC"

        if limit:
            query += f" LIMIT {limit}"

        df = self.fetch_dataframe(query, tuple(params))

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Convert Decimal columns to float for numerical operations
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

        return df

    def insert_orderbook(
        self,
        exchange: str,
        symbol: str,
        timestamp: datetime,
        bids: List[List[float]],
        asks: List[List[float]]
    ) -> int:
        """
        Insert orderbook snapshot.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timestamp: Snapshot timestamp
            bids: List of [price, quantity] for bids
            asks: List of [price, quantity] for asks

        Returns:
            Number of rows inserted
        """
        import json

        query = """
            INSERT INTO orderbook
            (exchange, symbol, timestamp, bids, asks)
            VALUES (%s, %s, %s, %s, %s)
        """

        return self.execute(query, (
            exchange, symbol, timestamp,
            json.dumps(bids), json.dumps(asks)
        ))

    def insert_trades(
        self,
        exchange: str,
        symbol: str,
        trades: List[Dict[str, Any]]
    ) -> int:
        """
        Insert trade data.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            trades: List of trade dictionaries

        Returns:
            Number of rows inserted
        """
        query = """
            INSERT INTO trades
            (exchange, symbol, trade_id, timestamp, price, quantity, side)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            price = VALUES(price),
            quantity = VALUES(quantity),
            side = VALUES(side)
        """

        params_list = []
        for trade in trades:
            params_list.append((
                exchange, symbol,
                trade['trade_id'],
                trade['timestamp'],
                trade['price'],
                trade['quantity'],
                trade['side']
            ))

        return self.execute_many(query, params_list)

    def insert_funding_rate(
        self,
        exchange: str,
        symbol: str,
        timestamp: datetime,
        funding_rate: float,
        funding_time: datetime
    ) -> int:
        """
        Insert funding rate data.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timestamp: Data timestamp
            funding_rate: Funding rate value
            funding_time: Funding settlement time

        Returns:
            Number of rows inserted
        """
        query = """
            INSERT INTO funding_rates
            (exchange, symbol, timestamp, funding_rate, funding_time)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            funding_rate = VALUES(funding_rate),
            funding_time = VALUES(funding_time)
        """

        return self.execute(query, (
            exchange, symbol, timestamp, funding_rate, funding_time
        ))

    def get_latest_timestamp(
        self,
        table: str,
        exchange: str,
        symbol: str,
        timeframe: Optional[str] = None
    ) -> Optional[datetime]:
        """
        Get latest timestamp from table.

        Args:
            table: Table name
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Timeframe (for ohlcv table)

        Returns:
            Latest timestamp or None
        """
        if table == 'ohlcv' and timeframe:
            query = """
                SELECT MAX(timestamp) as max_ts
                FROM ohlcv
                WHERE exchange = %s AND symbol = %s AND timeframe = %s
            """
            params = (exchange, symbol, timeframe)
        else:
            query = f"""
                SELECT MAX(timestamp) as max_ts
                FROM {table}
                WHERE exchange = %s AND symbol = %s
            """
            params = (exchange, symbol)

        result = self.fetch_one(query, params)

        if result and result['max_ts']:
            return result['max_ts']
        return None

    def get_data_count(
        self,
        table: str,
        exchange: str,
        symbol: str,
        timeframe: Optional[str] = None
    ) -> int:
        """
        Get count of records in table.

        Args:
            table: Table name
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Timeframe (for ohlcv table)

        Returns:
            Record count
        """
        if table == 'ohlcv' and timeframe:
            query = """
                SELECT COUNT(*) as cnt
                FROM ohlcv
                WHERE exchange = %s AND symbol = %s AND timeframe = %s
            """
            params = (exchange, symbol, timeframe)
        else:
            query = f"""
                SELECT COUNT(*) as cnt
                FROM {table}
                WHERE exchange = %s AND symbol = %s
            """
            params = (exchange, symbol)

        result = self.fetch_one(query, params)
        return result['cnt'] if result else 0

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
