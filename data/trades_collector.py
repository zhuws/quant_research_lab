"""
Trades Collector for Quant Research Lab.
Collects and processes trade data.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from collections import deque
import threading

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger


class Trade:
    """
    Single trade record.
    """

    def __init__(
        self,
        trade_id: str,
        symbol: str,
        exchange: str,
        timestamp: datetime,
        price: float,
        quantity: float,
        side: str  # 'buy' or 'sell'
    ):
        """
        Initialize trade record.

        Args:
            trade_id: Trade identifier
            symbol: Trading symbol
            exchange: Exchange name
            timestamp: Trade timestamp
            price: Trade price
            quantity: Trade quantity
            side: Trade side
        """
        self.trade_id = trade_id
        self.symbol = symbol
        self.exchange = exchange
        self.timestamp = timestamp
        self.price = price
        self.quantity = quantity
        self.side = side.lower()

    @property
    def value(self) -> float:
        """Get trade value."""
        return self.price * self.quantity

    @property
    def is_buy(self) -> bool:
        """Check if buy trade."""
        return self.side == 'buy'

    @property
    def is_sell(self) -> bool:
        """Check if sell trade."""
        return self.side == 'sell'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timestamp': self.timestamp,
            'price': self.price,
            'quantity': self.quantity,
            'side': self.side,
            'value': self.value
        }


class TradesCollector:
    """
    Collects and processes trade data.
    """

    def __init__(
        self,
        max_trades: int = 100000,
        storage: Optional[Any] = None
    ):
        """
        Initialize trades collector.

        Args:
            max_trades: Maximum trades to keep in memory
            storage: Optional database storage
        """
        self.max_trades = max_trades
        self.storage = storage

        self.logger = get_logger('trades_collector')
        self._trades: deque = deque(maxlen=max_trades)
        self._lock = threading.Lock()

    def add_trade(
        self,
        trade_id: str,
        symbol: str,
        exchange: str,
        timestamp: datetime,
        price: float,
        quantity: float,
        side: str
    ) -> Trade:
        """
        Add a trade to the collection.

        Args:
            trade_id: Trade identifier
            symbol: Trading symbol
            exchange: Exchange name
            timestamp: Trade timestamp
            price: Trade price
            quantity: Trade quantity
            side: Trade side

        Returns:
            Trade object
        """
        trade = Trade(
            trade_id=trade_id,
            symbol=symbol,
            exchange=exchange,
            timestamp=timestamp,
            price=price,
            quantity=quantity,
            side=side
        )

        with self._lock:
            self._trades.append(trade)

        return trade

    def add_trades_batch(self, trades: List[Trade]) -> None:
        """
        Add multiple trades at once.

        Args:
            trades: List of Trade objects
        """
        with self._lock:
            for trade in trades:
                self._trades.append(trade)

    def get_recent(
        self,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        limit: int = 1000
    ) -> List[Trade]:
        """
        Get recent trades.

        Args:
            symbol: Filter by symbol
            exchange: Filter by exchange
            limit: Maximum trades to return

        Returns:
            List of trades
        """
        with self._lock:
            trades = list(self._trades)

        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        if exchange:
            trades = [t for t in trades if t.exchange == exchange]

        return trades[-limit:]

    def get_trades_in_range(
        self,
        start_time: datetime,
        end_time: datetime,
        symbol: Optional[str] = None
    ) -> List[Trade]:
        """
        Get trades in time range.

        Args:
            start_time: Start timestamp
            end_time: End timestamp
            symbol: Filter by symbol

        Returns:
            List of trades
        """
        with self._lock:
            trades = list(self._trades)

        trades = [t for t in trades if start_time <= t.timestamp <= end_time]

        if symbol:
            trades = [t for t in trades if t.symbol == symbol]

        return trades

    def to_dataframe(
        self,
        symbol: Optional[str] = None,
        limit: int = 10000
    ) -> pd.DataFrame:
        """
        Convert trades to DataFrame.

        Args:
            symbol: Filter by symbol
            limit: Maximum trades to include

        Returns:
            DataFrame with trade data
        """
        trades = self.get_recent(symbol=symbol, limit=limit)

        if not trades:
            return pd.DataFrame()

        records = [t.to_dict() for t in trades]
        df = pd.DataFrame(records)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df


class TradeFlowAnalyzer:
    """
    Analyzes trade flow patterns.
    """

    def __init__(
        self,
        window_seconds: int = 60
    ):
        """
        Initialize trade flow analyzer.

        Args:
            window_seconds: Time window for analysis
        """
        self.window_seconds = window_seconds
        self.logger = get_logger('trade_flow_analyzer')

    def calculate_flow_metrics(
        self,
        trades: List[Trade]
    ) -> Dict[str, float]:
        """
        Calculate trade flow metrics.

        Args:
            trades: List of trades

        Returns:
            Dictionary of metrics
        """
        if not trades:
            return {}

        buy_trades = [t for t in trades if t.is_buy]
        sell_trades = [t for t in trades if t.is_sell]

        buy_volume = sum(t.quantity for t in buy_trades)
        sell_volume = sum(t.quantity for t in sell_trades)

        buy_value = sum(t.value for t in buy_trades)
        sell_value = sum(t.value for t in sell_trades)

        total_volume = buy_volume + sell_volume
        total_value = buy_value + sell_value

        vwap_buy = buy_value / buy_volume if buy_volume > 0 else 0
        vwap_sell = sell_value / sell_volume if sell_volume > 0 else 0

        delta = buy_volume - sell_volume

        return {
            'buy_count': len(buy_trades),
            'sell_count': len(sell_trades),
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'buy_value': buy_value,
            'sell_value': sell_value,
            'total_volume': total_volume,
            'total_value': total_value,
            'vwap_buy': vwap_buy,
            'vwap_sell': vwap_sell,
            'delta': delta,
            'delta_pct': delta / total_volume if total_volume > 0 else 0,
            'buy_ratio': buy_volume / total_volume if total_volume > 0 else 0.5
        }

    def detect_large_trades(
        self,
        trades: List[Trade],
        threshold_qty: float = 10.0,
        threshold_value: float = 50000.0
    ) -> List[Trade]:
        """
        Detect large trades.

        Args:
            trades: List of trades
            threshold_qty: Quantity threshold
            threshold_value: Value threshold

        Returns:
            List of large trades
        """
        large_trades = []

        for trade in trades:
            if trade.quantity >= threshold_qty or trade.value >= threshold_value:
                large_trades.append(trade)

        return large_trades

    def calculate_trade_pressure(
        self,
        trades: List[Trade]
    ) -> float:
        """
        Calculate trade pressure indicator.

        Args:
            trades: List of trades

        Returns:
            Pressure value (-1 to 1)
        """
        if not trades:
            return 0.0

        metrics = self.calculate_flow_metrics(trades)
        return metrics.get('delta_pct', 0)

    def detect_aggression(
        self,
        trades: List[Trade]
    ) -> Dict[str, Any]:
        """
        Detect aggressive trading activity.

        Args:
            trades: List of trades

        Returns:
            Aggression analysis
        """
        if not trades:
            return {'aggressive_side': None, 'aggression_score': 0}

        buy_volume = sum(t.quantity for t in trades if t.is_buy)
        sell_volume = sum(t.quantity for t in trades if t.is_sell)

        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return {'aggressive_side': None, 'aggression_score': 0}

        buy_ratio = buy_volume / total_volume
        sell_ratio = sell_volume / total_volume

        if buy_ratio > 0.6:
            aggressive_side = 'buy'
            aggression_score = buy_ratio - 0.5
        elif sell_ratio > 0.6:
            aggressive_side = 'sell'
            aggression_score = sell_ratio - 0.5
        else:
            aggressive_side = None
            aggression_score = 0

        return {
            'aggressive_side': aggressive_side,
            'aggression_score': aggression_score * 2,  # Scale to 0-1
            'buy_ratio': buy_ratio,
            'sell_ratio': sell_ratio
        }


class VolumeProfile:
    """
    Calculates volume profile from trades.
    """

    def __init__(
        self,
        price_bins: int = 50
    ):
        """
        Initialize volume profile calculator.

        Args:
            price_bins: Number of price bins
        """
        self.price_bins = price_bins
        self.logger = get_logger('volume_profile')

    def calculate(
        self,
        trades: List[Trade]
    ) -> pd.DataFrame:
        """
        Calculate volume profile.

        Args:
            trades: List of trades

        Returns:
            DataFrame with volume profile
        """
        if not trades:
            return pd.DataFrame()

        prices = [t.price for t in trades]
        volumes = [t.quantity for t in trades]
        sides = [t.side for t in trades]

        # Create price bins
        min_price = min(prices)
        max_price = max(prices)

        price_range = max_price - min_price
        bin_size = price_range / self.price_bins if price_range > 0 else 0.01

        bins = np.arange(min_price, max_price + bin_size, bin_size)

        # Calculate volume at each price level
        df = pd.DataFrame({
            'price': prices,
            'volume': volumes,
            'side': sides
        })

        df['price_bin'] = pd.cut(df['price'], bins=bins)

        profile = df.groupby('price_bin').agg({
            'volume': 'sum',
            'side': lambda x: (x == 'buy').sum() / len(x) if len(x) > 0 else 0.5
        }).reset_index()

        profile.columns = ['price_range', 'volume', 'buy_ratio']
        profile['price_mid'] = profile['price_range'].apply(
            lambda x: x.mid if pd.notna(x) else None
        )

        return profile.dropna()

    def find_poc(
        self,
        profile: pd.DataFrame
    ) -> Optional[float]:
        """
        Find Point of Control (price with highest volume).

        Args:
            profile: Volume profile DataFrame

        Returns:
            POC price
        """
        if profile.empty:
            return None

        poc_idx = profile['volume'].idxmax()
        return profile.loc[poc_idx, 'price_mid']

    def find_value_area(
        self,
        profile: pd.DataFrame,
        value_area_pct: float = 0.7
    ) -> Dict[str, float]:
        """
        Find Value Area (price range containing value_area_pct of volume).

        Args:
            profile: Volume profile DataFrame
            value_area_pct: Percentage of volume to include

        Returns:
            Dictionary with VAH, VAL, and POC
        """
        if profile.empty:
            return {}

        total_volume = profile['volume'].sum()
        target_volume = total_volume * value_area_pct

        # Sort by volume descending
        sorted_profile = profile.sort_values('volume', ascending=False)

        accumulated = 0
        va_levels = []

        for _, row in sorted_profile.iterrows():
            accumulated += row['volume']
            va_levels.append(row['price_mid'])

            if accumulated >= target_volume:
                break

        if not va_levels:
            return {}

        return {
            'vah': max(va_levels),  # Value Area High
            'val': min(va_levels),  # Value Area Low
            'poc': self.find_poc(profile)
        }
