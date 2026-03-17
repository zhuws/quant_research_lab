"""
Funding Rate Fetcher for Quant Research Lab.
Fetches and analyzes funding rates from exchanges.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from collections import deque

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger


class FundingRate:
    """
    Single funding rate record.
    """

    def __init__(
        self,
        symbol: str,
        exchange: str,
        timestamp: datetime,
        funding_rate: float,
        funding_time: datetime
    ):
        """
        Initialize funding rate.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timestamp: Data timestamp
            funding_rate: Funding rate value
            funding_time: Funding settlement time
        """
        self.symbol = symbol
        self.exchange = exchange
        self.timestamp = timestamp
        self.funding_rate = funding_rate
        self.funding_time = funding_time

    @property
    def annualized_rate(self) -> float:
        """Get annualized funding rate."""
        # 3 funding periods per day * 365 days
        return self.funding_rate * 3 * 365 * 100

    @property
    def is_positive(self) -> bool:
        """Check if funding is positive (longs pay shorts)."""
        return self.funding_rate > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timestamp': self.timestamp,
            'funding_rate': self.funding_rate,
            'funding_time': self.funding_time,
            'annualized_rate': self.annualized_rate
        }


class FundingRateFetcher:
    """
    Fetches and manages funding rate data.
    """

    def __init__(
        self,
        max_records: int = 10000,
        storage: Optional[Any] = None
    ):
        """
        Initialize funding rate fetcher.

        Args:
            max_records: Maximum records to keep in memory
            storage: Optional database storage
        """
        self.max_records = max_records
        self.storage = storage

        self.logger = get_logger('funding_rate_fetcher')
        self._rates: deque = deque(maxlen=max_records)

    def add_rate(
        self,
        symbol: str,
        exchange: str,
        timestamp: datetime,
        funding_rate: float,
        funding_time: datetime
    ) -> FundingRate:
        """
        Add a funding rate record.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timestamp: Data timestamp
            funding_rate: Funding rate value
            funding_time: Funding settlement time

        Returns:
            FundingRate object
        """
        rate = FundingRate(
            symbol=symbol,
            exchange=exchange,
            timestamp=timestamp,
            funding_rate=funding_rate,
            funding_time=funding_time
        )

        self._rates.append(rate)
        return rate

    def get_latest(
        self,
        symbol: str,
        exchange: str
    ) -> Optional[FundingRate]:
        """
        Get latest funding rate for symbol.

        Args:
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Latest FundingRate or None
        """
        for rate in reversed(self._rates):
            if rate.symbol == symbol and rate.exchange == exchange:
                return rate
        return None

    def get_history(
        self,
        symbol: str,
        exchange: str,
        limit: int = 100
    ) -> List[FundingRate]:
        """
        Get funding rate history.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            limit: Maximum records

        Returns:
            List of FundingRate objects
        """
        rates = [
            r for r in self._rates
            if r.symbol == symbol and r.exchange == exchange
        ]
        return rates[-limit:]

    def to_dataframe(
        self,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Convert to DataFrame.

        Args:
            symbol: Filter by symbol
            exchange: Filter by exchange

        Returns:
            DataFrame with funding rates
        """
        rates = list(self._rates)

        if symbol:
            rates = [r for r in rates if r.symbol == symbol]
        if exchange:
            rates = [r for r in rates if r.exchange == exchange]

        if not rates:
            return pd.DataFrame()

        records = [r.to_dict() for r in rates]
        df = pd.DataFrame(records)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df


class FundingRateAnalyzer:
    """
    Analyzes funding rates for trading signals.
    """

    def __init__(
        self,
        high_rate_threshold: float = 0.0005,  # 0.05%
        low_rate_threshold: float = -0.0005
    ):
        """
        Initialize funding rate analyzer.

        Args:
            high_rate_threshold: Threshold for high positive funding
            low_rate_threshold: Threshold for negative funding
        """
        self.high_rate_threshold = high_rate_threshold
        self.low_rate_threshold = low_rate_threshold
        self.logger = get_logger('funding_rate_analyzer')

    def analyze(
        self,
        rate: FundingRate
    ) -> Dict[str, Any]:
        """
        Analyze a funding rate.

        Args:
            rate: FundingRate object

        Returns:
            Analysis results
        """
        signal = 'neutral'
        sentiment = 'neutral'

        if rate.funding_rate > self.high_rate_threshold:
            signal = 'short_bias'  # High positive funding = shorts get paid
            sentiment = 'bullish'  # Market sentiment is bullish
        elif rate.funding_rate < self.low_rate_threshold:
            signal = 'long_bias'   # Negative funding = longs get paid
            sentiment = 'bearish'  # Market sentiment is bearish

        return {
            'funding_rate': rate.funding_rate,
            'annualized_rate': rate.annualized_rate,
            'signal': signal,
            'sentiment': sentiment,
            'is_extreme': abs(rate.funding_rate) > self.high_rate_threshold
        }

    def calculate_mean_reversion_signal(
        self,
        rates: List[FundingRate],
        lookback: int = 8  # ~1 day (8 * 8h = 64h)
    ) -> Dict[str, Any]:
        """
        Calculate mean reversion signal from funding rate history.

        Args:
            rates: List of funding rates
            lookback: Number of periods to analyze

        Returns:
            Mean reversion analysis
        """
        if len(rates) < lookback:
            return {'signal': 0, 'confidence': 0}

        recent_rates = [r.funding_rate for r in rates[-lookback:]]
        current_rate = recent_rates[-1]

        mean_rate = np.mean(recent_rates[:-1])
        std_rate = np.std(recent_rates[:-1])

        if std_rate == 0:
            return {'signal': 0, 'confidence': 0}

        z_score = (current_rate - mean_rate) / std_rate

        # Signal: if funding is high (positive z-score), expect mean reversion down
        signal = -np.clip(z_score / 2, -1, 1)
        confidence = min(abs(z_score) / 2, 1.0)

        return {
            'current_rate': current_rate,
            'mean_rate': mean_rate,
            'std_rate': std_rate,
            'z_score': z_score,
            'signal': signal,
            'confidence': confidence
        }

    def compare_exchanges(
        self,
        rates: Dict[str, FundingRate]
    ) -> Dict[str, Any]:
        """
        Compare funding rates across exchanges.

        Args:
            rates: Dictionary of exchange -> FundingRate

        Returns:
            Cross-exchange comparison
        """
        if len(rates) < 2:
            return {'arbitrage_opportunity': False}

        exchanges = list(rates.keys())
        rate_values = {ex: rates[ex].funding_rate for ex in exchanges}

        max_exchange = max(rate_values, key=rate_values.get)
        min_exchange = min(rate_values, key=rate_values.get)

        spread = rate_values[max_exchange] - rate_values[min_exchange]
        annualized_spread = spread * 3 * 365 * 100

        return {
            'rates': rate_values,
            'max_exchange': max_exchange,
            'min_exchange': min_exchange,
            'spread': spread,
            'annualized_spread_pct': annualized_spread,
            'arbitrage_opportunity': annualized_spread > 10  # >10% annualized
        }

    def predict_direction(
        self,
        rates: List[FundingRate],
        price_changes: List[float]
    ) -> Dict[str, Any]:
        """
        Predict price direction from funding rate.

        Args:
            rates: List of funding rates
            price_changes: Corresponding price changes

        Returns:
            Prediction analysis
        """
        if len(rates) < 10 or len(price_changes) < 10:
            return {'prediction': 0, 'accuracy': 0}

        # Calculate correlation between funding and subsequent price
        funding_values = [r.funding_rate for r in rates[-50:]]
        returns = price_changes[-50:]

        if len(funding_values) != len(returns):
            return {'prediction': 0, 'accuracy': 0}

        correlation = np.corrcoef(funding_values[:-1], returns[1:])[0, 1]

        # Use current funding to predict
        current_funding = funding_values[-1]

        if np.isnan(correlation):
            return {'prediction': 0, 'accuracy': 0}

        # If positive correlation, positive funding predicts positive returns
        prediction = np.sign(correlation) * np.sign(current_funding)

        return {
            'current_funding': current_funding,
            'correlation': correlation,
            'prediction': prediction,
            'confidence': abs(correlation)
        }
