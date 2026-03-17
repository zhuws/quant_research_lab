"""
Orderbook Recorder for Quant Research Lab.
Records and processes orderbook snapshots.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime
from collections import deque
import threading

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger


class OrderBookSnapshot:
    """
    Single orderbook snapshot container.
    """

    def __init__(
        self,
        symbol: str,
        exchange: str,
        timestamp: datetime,
        bids: List[List[float]],
        asks: List[List[float]]
    ):
        """
        Initialize orderbook snapshot.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timestamp: Snapshot timestamp
            bids: List of [price, quantity] for bids (sorted descending by price)
            asks: List of [price, quantity] for asks (sorted ascending by price)
        """
        self.symbol = symbol
        self.exchange = exchange
        self.timestamp = timestamp
        self.bids = sorted(bids, key=lambda x: -x[0])  # Descending
        self.asks = sorted(asks, key=lambda x: x[0])   # Ascending

    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price."""
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price."""
        return self.asks[0][0] if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid_price(self) -> Optional[float]:
        """Get mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        """Get spread in basis points."""
        if self.spread and self.mid_price:
            return (self.spread / self.mid_price) * 10000
        return None

    def get_imbalance(self, depth: int = 10) -> float:
        """
        Calculate orderbook imbalance.

        Args:
            depth: Number of levels to consider

        Returns:
            Imbalance value (-1 to 1)
        """
        bid_volume = sum(q for _, q in self.bids[:depth])
        ask_volume = sum(q for _, q in self.asks[:depth])

        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0

        return (bid_volume - ask_volume) / total_volume

    def get_micro_price(self) -> Optional[float]:
        """
        Calculate micro price.

        Returns:
            Micro price weighted by volumes at best bid/ask
        """
        if not self.bids or not self.asks:
            return None

        best_bid, bid_qty = self.bids[0]
        best_ask, ask_qty = self.asks[0]

        total_qty = bid_qty + ask_qty
        if total_qty == 0:
            return self.mid_price

        return (best_ask * bid_qty + best_bid * ask_qty) / total_qty

    def get_depth(self, levels: int = 5) -> Dict[str, float]:
        """
        Get orderbook depth statistics.

        Args:
            levels: Number of levels to analyze

        Returns:
            Dictionary with depth statistics
        """
        bid_volume = sum(q for _, q in self.bids[:levels])
        ask_volume = sum(q for _, q in self.asks[:levels])

        bid_value = sum(p * q for p, q in self.bids[:levels])
        ask_value = sum(p * q for p, q in self.asks[:levels])

        return {
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'total_volume': bid_volume + ask_volume,
            'bid_value': bid_value,
            'ask_value': ask_value,
            'imbalance': self.get_imbalance(levels)
        }

    def get_price_levels(self, side: str, count: int = 10) -> List[Dict]:
        """
        Get price levels for a side.

        Args:
            side: 'bid' or 'ask'
            count: Number of levels to return

        Returns:
            List of price level dictionaries
        """
        levels = self.bids[:count] if side == 'bid' else self.asks[:count]
        return [{'price': p, 'quantity': q} for p, q in levels]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timestamp': self.timestamp,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'spread': self.spread,
            'spread_bps': self.spread_bps,
            'mid_price': self.mid_price,
            'micro_price': self.get_micro_price(),
            'imbalance': self.get_imbalance(),
            'bid_count': len(self.bids),
            'ask_count': len(self.asks)
        }


class OrderBookRecorder:
    """
    Records and processes orderbook data.
    """

    def __init__(
        self,
        max_snapshots: int = 10000,
        storage: Optional[Any] = None
    ):
        """
        Initialize orderbook recorder.

        Args:
            max_snapshots: Maximum snapshots to keep in memory
            storage: Optional database storage
        """
        self.max_snapshots = max_snapshots
        self.storage = storage

        self.logger = get_logger('orderbook_recorder')
        self._snapshots: deque = deque(maxlen=max_snapshots)
        self._lock = threading.Lock()

    def record(
        self,
        symbol: str,
        exchange: str,
        timestamp: datetime,
        bids: List[List[float]],
        asks: List[List[float]]
    ) -> OrderBookSnapshot:
        """
        Record an orderbook snapshot.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timestamp: Snapshot timestamp
            bids: Bid levels
            asks: Ask levels

        Returns:
            OrderBookSnapshot object
        """
        snapshot = OrderBookSnapshot(
            symbol=symbol,
            exchange=exchange,
            timestamp=timestamp,
            bids=bids,
            asks=asks
        )

        with self._lock:
            self._snapshots.append(snapshot)

        return snapshot

    def get_latest(self) -> Optional[OrderBookSnapshot]:
        """Get latest snapshot."""
        with self._lock:
            return self._snapshots[-1] if self._snapshots else None

    def get_history(
        self,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        limit: int = 100
    ) -> List[OrderBookSnapshot]:
        """
        Get snapshot history.

        Args:
            symbol: Filter by symbol
            exchange: Filter by exchange
            limit: Maximum snapshots to return

        Returns:
            List of snapshots
        """
        with self._lock:
            snapshots = list(self._snapshots)

        if symbol:
            snapshots = [s for s in snapshots if s.symbol == symbol]
        if exchange:
            snapshots = [s for s in snapshots if s.exchange == exchange]

        return snapshots[-limit:]

    def calculate_features(
        self,
        symbol: str,
        exchange: str,
        window: int = 100
    ) -> Dict[str, float]:
        """
        Calculate orderbook features over a window.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            window: Number of snapshots for calculations

        Returns:
            Dictionary of features
        """
        history = self.get_history(symbol, exchange, window)

        if not history:
            return {}

        spreads = [s.spread_bps for s in history if s.spread_bps]
        imbalances = [s.get_imbalance() for s in history]
        mid_prices = [s.mid_price for s in history if s.mid_price]

        features = {
            'spread_bps_mean': np.mean(spreads) if spreads else 0,
            'spread_bps_std': np.std(spreads) if spreads else 0,
            'imbalance_mean': np.mean(imbalances) if imbalances else 0,
            'imbalance_std': np.std(imbalances) if imbalances else 0,
            'mid_price': mid_prices[-1] if mid_prices else 0,
            'mid_price_change': (
                (mid_prices[-1] - mid_prices[0]) / mid_prices[0] * 100
                if len(mid_prices) > 1 else 0
            )
        }

        return features

    def to_dataframe(self, limit: int = 1000) -> pd.DataFrame:
        """
        Convert snapshots to DataFrame.

        Args:
            limit: Maximum snapshots to include

        Returns:
            DataFrame with orderbook data
        """
        snapshots = list(self._snapshots)[-limit:]

        if not snapshots:
            return pd.DataFrame()

        records = [s.to_dict() for s in snapshots]
        df = pd.DataFrame(records)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    def clear(self) -> None:
        """Clear all recorded snapshots."""
        with self._lock:
            self._snapshots.clear()
        self.logger.info("Cleared orderbook recorder")


class OrderBookAnalyzer:
    """
    Analyzes orderbook patterns and generates signals.
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.3,
        spread_threshold_bps: float = 5.0
    ):
        """
        Initialize orderbook analyzer.

        Args:
            imbalance_threshold: Threshold for imbalance signals
            spread_threshold_bps: Threshold for spread alerts
        """
        self.imbalance_threshold = imbalance_threshold
        self.spread_threshold_bps = spread_threshold_bps
        self.logger = get_logger('orderbook_analyzer')

    def analyze_imbalance(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """
        Analyze orderbook imbalance.

        Args:
            snapshot: Orderbook snapshot

        Returns:
            Analysis results
        """
        imbalance = snapshot.get_imbalance()

        signal = 0
        if imbalance > self.imbalance_threshold:
            signal = 1  # Bullish
        elif imbalance < -self.imbalance_threshold:
            signal = -1  # Bearish

        return {
            'imbalance': imbalance,
            'signal': signal,
            'strength': abs(imbalance),
            'confidence': min(abs(imbalance) / self.imbalance_threshold, 1.0)
        }

    def analyze_spread(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """
        Analyze spread conditions.

        Args:
            snapshot: Orderbook snapshot

        Returns:
            Spread analysis
        """
        spread_bps = snapshot.spread_bps or 0

        is_wide = spread_bps > self.spread_threshold_bps
        liquidity = 'low' if is_wide else 'normal'

        return {
            'spread_bps': spread_bps,
            'is_wide': is_wide,
            'liquidity': liquidity
        }

    def detect_support_resistance(
        self,
        snapshot: OrderBookSnapshot,
        threshold_qty: float = 10.0
    ) -> Dict[str, List[float]]:
        """
        Detect support and resistance levels from orderbook.

        Args:
            snapshot: Orderbook snapshot
            threshold_qty: Minimum quantity to consider as level

        Returns:
            Dictionary with support and resistance prices
        """
        support = []
        resistance = []

        # Find large bid orders (support)
        for price, qty in snapshot.bids:
            if qty >= threshold_qty:
                support.append(price)

        # Find large ask orders (resistance)
        for price, qty in snapshot.asks:
            if qty >= threshold_qty:
                resistance.append(price)

        return {
            'support': support[:5],  # Top 5 levels
            'resistance': resistance[:5]
        }

    def generate_signal(
        self,
        snapshot: OrderBookSnapshot
    ) -> Dict[str, Any]:
        """
        Generate trading signal from orderbook.

        Args:
            snapshot: Orderbook snapshot

        Returns:
            Signal dictionary
        """
        imbalance_analysis = self.analyze_imbalance(snapshot)
        spread_analysis = self.analyze_spread(snapshot)
        levels = self.detect_support_resistance(snapshot)

        # Combine signals
        signal = imbalance_analysis['signal']
        confidence = imbalance_analysis['confidence']

        # Adjust for spread
        if spread_analysis['is_wide']:
            confidence *= 0.5  # Lower confidence in wide spread

        return {
            'signal': signal,
            'confidence': confidence,
            'imbalance': imbalance_analysis,
            'spread': spread_analysis,
            'levels': levels,
            'timestamp': snapshot.timestamp
        }
