"""
Volatility Feature Generator for Quant Research Lab.
Generates specialized volatility measures and related indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger


class VolatilityFeatureGenerator:
    """
    Generates volatility-based features from OHLCV data.

    Feature Categories:
        - Average True Range (ATR) variants
        - Bollinger Bands features
        - Keltner Channels
        - Historical volatility (realized, Parkinson, Garman-Klass)
        - Volatility regime indicators

    Total Features: ~35 volatility indicators
    """

    def __init__(
        self,
        atr_periods: List[int] = None,
        vol_periods: List[int] = None,
        bollinger_periods: List[int] = None,
        bollinger_std: float = 2.0
    ):
        """
        Initialize Volatility Feature Generator.

        Args:
            atr_periods: Periods for ATR calculations
            vol_periods: Periods for historical volatility
            bollinger_periods: Periods for Bollinger Bands
            bollinger_std: Standard deviation multiplier for Bollinger
        """
        self.logger = get_logger('volatility_features')

        self.atr_periods = atr_periods or [7, 14, 21]
        self.vol_periods = vol_periods or [5, 10, 20, 60]
        self.bollinger_periods = bollinger_periods or [20, 50]
        self.bollinger_std = bollinger_std

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all volatility features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with volatility features added
        """
        if df.empty:
            return df

        self.logger.info(f"Generating volatility features from {len(df)} bars")

        features = df.copy()

        # Calculate True Range first (needed for multiple features)
        features = self._calculate_true_range(features)

        # Generate feature categories
        features = self._generate_atr_features(features)
        features = self._generate_bollinger_features(features)
        features = self._generate_keltner_features(features)
        features = self._generate_historical_volatility(features)
        features = self._generate_volatility_regime(features)
        features = self._generate_range_features(features)

        # Replace inf values
        features = features.replace([np.inf, -np.inf], np.nan)

        self.logger.info(f"Generated {len(features.columns) - len(df.columns)} volatility features")

        return features

    def _calculate_true_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate True Range."""
        # True Range = max(high-low, abs(high-previous_close), abs(low-previous_close))
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )

        # Normalized True Range (as % of close)
        df['tr_normalized'] = df['true_range'] / df['close']

        # High-Low range
        df['hl_range'] = df['high'] - df['low']
        df['hl_range_pct'] = df['hl_range'] / df['close']

        return df

    def _generate_atr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate ATR-based features."""
        for period in self.atr_periods:
            # Average True Range
            df[f'atr_{period}'] = df['true_range'].rolling(period).mean()

            # ATR as percentage of price
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['close']

            # ATR normalized by typical price
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df[f'atr_normalized_{period}'] = df[f'atr_{period}'] / typical_price.rolling(period).mean()

            # ATR momentum
            df[f'atr_momentum_{period}'] = df[f'atr_{period}'].pct_change(period)

            # ATR ratio (current vs historical average)
            df[f'atr_ratio_{period}'] = df['true_range'] / df[f'atr_{period}']

            # ATR trend (rising/falling volatility)
            df[f'atr_trend_{period}'] = (
                df[f'atr_{period}'] > df[f'atr_{period}'].shift(period)
            ).astype(int)

        # Relative ATR (across multiple periods)
        if len(self.atr_periods) >= 2:
            short = self.atr_periods[0]
            long = self.atr_periods[-1]
            df['atr_relative'] = df[f'atr_{short}'] / df[f'atr_{long}']

        return df

    def _generate_bollinger_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Bollinger Bands features."""
        for period in self.bollinger_periods:
            # Bollinger Bands
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()

            upper = sma + (self.bollinger_std * std)
            lower = sma - (self.bollinger_std * std)

            df[f'bb_upper_{period}'] = upper
            df[f'bb_lower_{period}'] = lower
            df[f'bb_middle_{period}'] = sma
            df[f'bb_width_{period}'] = (upper - lower) / sma

            # Bollinger %B (position within bands)
            df[f'bb_pct_{period}'] = (df['close'] - lower) / (upper - lower)

            # Bollinger bandwidth expansion
            df[f'bb_width_change_{period}'] = df[f'bb_width_{period}'].pct_change(period)

            # Distance from bands
            df[f'dist_to_upper_bb_{period}'] = (upper - df['close']) / df['close']
            df[f'dist_to_lower_bb_{period}'] = (df['close'] - lower) / df['close']

            # Band squeeze detection
            width_ma = df[f'bb_width_{period}'].rolling(period).mean()
            df[f'bb_squeeze_{period}'] = (
                df[f'bb_width_{period}'] < width_ma * 0.5
            ).astype(int)

            # Breakout signals
            df[f'bb_breakout_upper_{period}'] = (df['close'] > upper).astype(int)
            df[f'bb_breakout_lower_{period}'] = (df['close'] < lower).astype(int)

            # Z-score within Bollinger
            df[f'bb_zscore_{period}'] = (df['close'] - sma) / std

        return df

    def _generate_keltner_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Keltner Channel features."""
        for period in [20]:
            # Typical Price
            tp = (df['high'] + df['low'] + df['close']) / 3

            # ATR for Keltner
            atr = df['true_range'].rolling(period).mean()

            # Keltner Channels (using 2x ATR)
            middle = tp.rolling(period).mean()
            upper = middle + (2 * atr)
            lower = middle - (2 * atr)

            df[f'keltner_upper_{period}'] = upper
            df[f'keltner_lower_{period}'] = lower
            df[f'keltner_middle_{period}'] = middle
            df[f'keltner_width_{period}'] = (upper - lower) / middle

            # Position within Keltner
            df[f'keltner_pct_{period}'] = (df['close'] - lower) / (upper - lower)

            # Keltner squeeze (when Bollinger inside Keltner)
            if f'bb_upper_{period}' in df.columns:
                df[f'keltner_squeeze_{period}'] = (
                    (df[f'bb_upper_{period}'] < upper) &
                    (df[f'bb_lower_{period}'] > lower)
                ).astype(int)

        return df

    def _generate_historical_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate historical volatility measures."""
        # Log returns
        log_returns = np.log(df['close'] / df['close'].shift(1))

        for period in self.vol_periods:
            # Realized Volatility (standard deviation of returns)
            df[f'realized_vol_{period}'] = log_returns.rolling(period).std() * np.sqrt(252)

            # Parkinson Volatility (high-low based)
            hl_ratio = np.log(df['high'] / df['low'])
            df[f'parkinson_vol_{period}'] = np.sqrt(
                (hl_ratio ** 2).rolling(period).mean() / (4 * np.log(2))
            ) * np.sqrt(252)

            # Garman-Klass Volatility
            # Uses OHLC data for more accurate estimation
            log_hl = np.log(df['high'] / df['low'])
            log_co = np.log(df['close'] / df['open'])

            gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
            df[f'garman_klass_vol_{period}'] = np.sqrt(
                gk.rolling(period).mean()
            ) * np.sqrt(252)

            # Rogers-Satchell Volatility
            log_ho = np.log(df['high'] / df['open'])
            log_lo = np.log(df['low'] / df['open'])
            log_co = np.log(df['close'] / df['open'])

            rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
            df[f'rogers_satchell_vol_{period}'] = np.sqrt(
                rs.rolling(period).mean()
            ) * np.sqrt(252)

            # Yang-Zhang Volatility (combines overnight and intraday)
            # Simplified version
            overnight_returns = np.log(df['open'] / df['close'].shift(1))
            intraday_returns = np.log(df['close'] / df['open'])

            overnight_var = overnight_returns.rolling(period).var()
            intraday_var = intraday_returns.rolling(period).var()
            rs_var = rs.rolling(period).mean()

            k = 0.34 / (1.34 + (period + 1) / (period - 1))
            df[f'yang_zhang_vol_{period}'] = np.sqrt(
                overnight_var + k * intraday_var + (1 - k) * rs_var
            ) * np.sqrt(252)

            # Volatility of volatility
            df[f'vol_of_vol_{period}'] = df[f'realized_vol_{period}'].rolling(period).std()

            # Volatility ratio (short vs long term)
            if period > self.vol_periods[0]:
                short_period = self.vol_periods[0]
                df[f'vol_ratio_{short_period}_{period}'] = (
                    df[f'realized_vol_{short_period}'] / df[f'realized_vol_{period}']
                )

        return df

    def _generate_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility regime indicators."""
        # Volatility percentile ranking
        for period in [20, 60]:
            if f'realized_vol_{period}' in df.columns:
                df[f'vol_percentile_{period}'] = (
                    df[f'realized_vol_{period}'].rolling(period * 2).rank(pct=True)
                )

                # High/low volatility regime
                df[f'high_vol_regime_{period}'] = (
                    df[f'vol_percentile_{period}'] > 0.8
                ).astype(int)
                df[f'low_vol_regime_{period}'] = (
                    df[f'vol_percentile_{period}'] < 0.2
                ).astype(int)

        # Volatility clustering
        df['vol_clustering'] = (
            (abs(df['close'].pct_change()) > abs(df['close'].pct_change()).rolling(20).quantile(0.8)) |
            (abs(df['close'].pct_change().shift(1)) > abs(df['close'].pct_change()).rolling(20).quantile(0.8))
        ).astype(int)

        # Volatility mean reversion signal
        df['vol_mean_reversion'] = (
            (df['realized_vol_20'] > df['realized_vol_20'].rolling(60).mean()) &
            (df['realized_vol_20'].shift(1) < df['realized_vol_20'].rolling(60).mean().shift(1))
        ).astype(int)

        # Volatility breakout
        for period in [20]:
            vol_std = df[f'realized_vol_{period}'].rolling(period).std()
            vol_mean = df[f'realized_vol_{period}'].rolling(period).mean()
            df[f'vol_breakout_{period}'] = (
                df[f'realized_vol_{period}'] > vol_mean + 2 * vol_std
            ).astype(int)

        # Implied volatility proxy (using GARCH-like approach)
        # Simplified: current vol vs trailing average
        if 'realized_vol_20' in df.columns:
            df['iv_proxy'] = df['realized_vol_20'].ewm(span=20).mean()
            df['iv_skew'] = df['realized_vol_20'] - df['iv_proxy']

        return df

    def _generate_range_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate range-based features."""
        for period in [5, 10, 20]:
            # Range expansion/contraction
            df[f'range_expansion_{period}'] = df['hl_range'].pct_change(period)

            # Range relative to ATR - use the closest available ATR period
            available_atr_periods = self.atr_periods
            closest_atr = min(available_atr_periods, key=lambda x: abs(x - period))
            df[f'range_to_atr_{period}'] = df['hl_range'] / df[f'atr_{closest_atr}']

            # Range efficiency (close-to-close vs high-low)
            close_to_close = abs(df['close'] - df['close'].shift(1))
            df[f'range_efficiency_{period}'] = close_to_close.rolling(period).mean() / df['hl_range'].rolling(period).mean()

            # Inside/outside range days
            df[f'inside_day_{period}'] = (
                (df['high'] < df['high'].shift(1)) &
                (df['low'] > df['low'].shift(1))
            ).rolling(period).sum()

            df[f'outside_day_{period}'] = (
                (df['high'] > df['high'].shift(1)) &
                (df['low'] < df['low'].shift(1))
            ).rolling(period).sum()

        # Range patterns
        df['narrow_range'] = (
            df['hl_range'] < df['hl_range'].rolling(7).mean() * 0.5
        ).astype(int)

        df['wide_range'] = (
            df['hl_range'] > df['hl_range'].rolling(7).mean() * 2.0
        ).astype(int)

        # NR4/NR7 patterns (Narrowest Range in 4/7 days)
        df['nr4'] = (df['hl_range'] == df['hl_range'].rolling(4).min()).astype(int)
        df['nr7'] = (df['hl_range'] == df['hl_range'].rolling(7).min()).astype(int)

        # Wide Range Bars
        df['wrb'] = (
            df['hl_range'] > df['hl_range'].rolling(10).mean() * 1.5
        ).astype(int)

        return df
