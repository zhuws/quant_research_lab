"""
Cross-Exchange Feature Generator for Quant Research Lab.
Generates cross-exchange arbitrage and correlation features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger


class CrossExchangeFeatureGenerator:
    """
    Generates cross-exchange features for arbitrage and correlation analysis.

    Feature Categories:
        - Price spread between exchanges
        - Premium/discount indicators
        - Correlation features
        - Arbitrage opportunity signals
        - Funding rate arbitrage

    Total Features: ~20 cross-exchange indicators
    """

    def __init__(
        self,
        spread_periods: List[int] = None,
        correlation_periods: List[int] = None
    ):
        """
        Initialize Cross-Exchange Feature Generator.

        Args:
            spread_periods: Periods for spread calculations
            correlation_periods: Periods for correlation calculations
        """
        self.logger = get_logger('cross_exchange_features')

        self.spread_periods = spread_periods or [1, 5, 10, 20]
        self.correlation_periods = correlation_periods or [20, 60, 120]

    def generate_features(
        self,
        df: pd.DataFrame,
        other_exchange_df: Optional[pd.DataFrame] = None,
        primary_exchange: str = 'binance',
        secondary_exchange: str = 'bybit'
    ) -> pd.DataFrame:
        """
        Generate cross-exchange features.

        Args:
            df: DataFrame with OHLCV data from primary exchange
            other_exchange_df: DataFrame with OHLCV data from secondary exchange
            primary_exchange: Primary exchange name
            secondary_exchange: Secondary exchange name

        Returns:
            DataFrame with cross-exchange features added
        """
        if df.empty:
            return df

        self.logger.info(f"Generating cross-exchange features from {len(df)} bars")

        features = df.copy()

        # If secondary exchange data available, generate full cross-exchange features
        if other_exchange_df is not None and not other_exchange_df.empty:
            features = self._merge_and_generate(features, other_exchange_df, primary_exchange, secondary_exchange)
        else:
            # Generate proxy features without second exchange data
            features = self._generate_proxy_features(features)

        # Replace inf values
        features = features.replace([np.inf, -np.inf], np.nan)

        self.logger.info(f"Generated {len(features.columns) - len(df.columns)} cross-exchange features")

        return features

    def _merge_and_generate(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        primary_exchange: str,
        secondary_exchange: str
    ) -> pd.DataFrame:
        """Merge exchange data and generate features."""
        # Ensure both have timestamp column
        if 'timestamp' not in primary_df.columns or 'timestamp' not in secondary_df.columns:
            self.logger.warning("Missing timestamp column for cross-exchange merge")
            return primary_df

        # Align timestamps
        primary_df = primary_df.copy()
        secondary_df = secondary_df.copy()

        primary_df['timestamp'] = pd.to_datetime(primary_df['timestamp'])
        secondary_df['timestamp'] = pd.to_datetime(secondary_df['timestamp'])

        # Merge on timestamp
        merged = pd.merge(
            primary_df,
            secondary_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']],
            on='timestamp',
            how='left',
            suffixes=('', f'_{secondary_exchange}')
        )

        # Generate spread features
        merged = self._generate_spread_features(
            merged,
            primary_exchange,
            secondary_exchange
        )

        # Generate correlation features
        merged = self._generate_correlation_features(
            merged,
            secondary_exchange
        )

        # Generate arbitrage features
        merged = self._generate_arbitrage_features(
            merged,
            primary_exchange,
            secondary_exchange
        )

        return merged

    def _generate_spread_features(
        self,
        df: pd.DataFrame,
        primary_exchange: str,
        secondary_exchange: str
    ) -> pd.DataFrame:
        """Generate price spread features."""
        close_col = f'close_{secondary_exchange}'

        if close_col not in df.columns:
            return df

        # Absolute spread
        df[f'spread_{primary_exchange}_{secondary_exchange}'] = (
            df['close'] - df[close_col]
        )

        # Percentage spread
        df[f'spread_pct_{primary_exchange}_{secondary_exchange}'] = (
            (df['close'] - df[close_col]) / df[close_col]
        )

        # Premium indicator (is primary exchange at premium?)
        df[f'premium_{primary_exchange}'] = (
            df[f'spread_pct_{primary_exchange}_{secondary_exchange}'] > 0
        ).astype(int)

        # Spread statistics
        for period in self.spread_periods:
            # Rolling spread mean
            df[f'spread_mean_{period}'] = (
                df[f'spread_pct_{primary_exchange}_{secondary_exchange}'].rolling(period).mean()
            )

            # Rolling spread std
            df[f'spread_std_{period}'] = (
                df[f'spread_pct_{primary_exchange}_{secondary_exchange}'].rolling(period).std()
            )

            # Spread z-score
            df[f'spread_zscore_{period}'] = (
                df[f'spread_pct_{primary_exchange}_{secondary_exchange}'] -
                df[f'spread_mean_{period}']
            ) / df[f'spread_std_{period}'].replace(0, np.nan)

            # Spread momentum
            df[f'spread_momentum_{period}'] = (
                df[f'spread_pct_{primary_exchange}_{secondary_exchange}'].diff(period)
            )

            # Spread mean reversion
            df[f'spread_reversion_{period}'] = (
                df[f'spread_pct_{primary_exchange}_{secondary_exchange}'] -
                df[f'spread_pct_{primary_exchange}_{secondary_exchange}'].rolling(period).mean()
            )

        # Spread extreme events
        spread_std_20 = df[f'spread_pct_{primary_exchange}_{secondary_exchange}'].rolling(20).std()
        spread_mean_20 = df[f'spread_pct_{primary_exchange}_{secondary_exchange}'].rolling(20).mean()
        df['spread_extreme_high'] = (
            df[f'spread_pct_{primary_exchange}_{secondary_exchange}'] >
            spread_mean_20 + 2 * spread_std_20
        ).astype(int)
        df['spread_extreme_low'] = (
            df[f'spread_pct_{primary_exchange}_{secondary_exchange}'] <
            spread_mean_20 - 2 * spread_std_20
        ).astype(int)

        # High/Low spread (using high and low prices)
        high_col = f'high_{secondary_exchange}'
        low_col = f'low_{secondary_exchange}'

        if high_col in df.columns and low_col in df.columns:
            # Best bid across exchanges
            df['cross_best_bid'] = df[['low', low_col]].max(axis=1)
            # Best ask across exchanges
            df['cross_best_ask'] = df[['high', high_col]].min(axis=1)
            # Cross-exchange spread
            df['cross_spread'] = df['cross_best_ask'] - df['cross_best_bid']

        return df

    def _generate_correlation_features(
        self,
        df: pd.DataFrame,
        secondary_exchange: str
    ) -> pd.DataFrame:
        """Generate correlation features between exchanges."""
        close_col = f'close_{secondary_exchange}'

        if close_col not in df.columns:
            return df

        # Price correlation
        for period in self.correlation_periods:
            df[f'price_corr_{period}'] = (
                df['close'].rolling(period).corr(df[close_col])
            )

            # Return correlation
            primary_ret = df['close'].pct_change()
            secondary_ret = df[close_col].pct_change()
            df[f'return_corr_{period}'] = primary_ret.rolling(period).corr(secondary_ret)

        # Correlation breakdown detection
        df['corr_breakdown'] = (
            (df['price_corr_20'] < 0.5) |
            (df['price_corr_20'].diff().abs() > 0.3)
        ).astype(int)

        # Lead-lag analysis
        for lag in [1, 2, 5]:
            df[f'lead_lag_{lag}'] = (
                df['close'].corr(df[close_col].shift(lag))
            )

        # Beta (price sensitivity)
        for period in [20, 60]:
            cov = df['close'].pct_change().rolling(period).cov(df[close_col].pct_change())
            var = df[close_col].pct_change().rolling(period).var()
            df[f'beta_{period}'] = cov / var.replace(0, np.nan)

        return df

    def _generate_arbitrage_features(
        self,
        df: pd.DataFrame,
        primary_exchange: str,
        secondary_exchange: str
    ) -> pd.DataFrame:
        """Generate arbitrage opportunity features."""
        close_col = f'close_{secondary_exchange}'

        if close_col not in df.columns:
            return df

        # Simple arbitrage signal
        # Buy on cheaper exchange, sell on more expensive
        spread_pct = df[f'spread_pct_{primary_exchange}_{secondary_exchange}']

        # Assume 0.1% transaction cost per side
        transaction_cost = 0.002  # 20 bps total

        # Arbitrage opportunity (spread > transaction costs)
        df['arb_opportunity_long'] = (
            spread_pct > transaction_cost
        ).astype(int)  # Primary is more expensive, short primary

        df['arb_opportunity_short'] = (
            spread_pct < -transaction_cost
        ).astype(int)  # Secondary is more expensive, long primary

        # Expected profit from arbitrage
        df['arb_profit_potential'] = spread_pct.abs() - transaction_cost
        df['arb_profit_potential'] = df['arb_profit_potential'].clip(lower=0)

        # Rolling arbitrage frequency
        df['arb_frequency_20'] = (
            (df['arb_opportunity_long'] | df['arb_opportunity_short']).rolling(20).mean()
        )

        # Arbitrage duration (how long spread persists)
        arb_active = (spread_pct.abs() > transaction_cost).astype(int)
        df['arb_duration'] = self._calculate_run_length(arb_active)

        # Spread efficiency
        # Ratio of realized spread to theoretical arbitrage profit
        for period in [20]:
            realized_profit = spread_pct.abs().rolling(period).sum()
            max_profit = spread_pct.abs().clip(lower=transaction_cost).rolling(period).sum()
            df['arb_efficiency'] = realized_profit / max_profit.replace(0, np.nan)

        return df

    def _calculate_run_length(self, series: pd.Series) -> pd.Series:
        """Calculate the length of consecutive runs of 1s."""
        groups = (series != series.shift()).cumsum()
        result = series.groupby(groups).transform('size') * series
        return result

    def _generate_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate proxy cross-exchange features when secondary data unavailable."""
        # Use price deviation from moving average as proxy for premium
        for period in [20, 60]:
            ma = df['close'].rolling(period).mean()
            df[f'premium_proxy_{period}'] = (df['close'] - ma) / ma

        # Use price volatility as proxy for cross-exchange spread
        for period in [20]:
            df[f'spread_proxy_{period}'] = df['close'].pct_change().rolling(period).std()

        # Synthetic spread using high-low range
        midpoint = (df['high'] + df['low']) / 2
        df['synthetic_spread'] = (df['close'] - midpoint).abs() / df['close']

        return df

    def generate_funding_arbitrage_features(
        self,
        df: pd.DataFrame,
        funding_rates: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate funding rate arbitrage features.

        Args:
            df: DataFrame with OHLCV data
            funding_rates: DataFrame with funding rate data (timestamp, funding_rate, exchange)

        Returns:
            DataFrame with funding arbitrage features
        """
        df = df.copy()

        if funding_rates is None or funding_rates.empty:
            # Generate proxy features
            df['funding_proxy'] = (
                df['close'].pct_change().rolling(480).mean() * 3  # ~8h funding period
            )
            return df

        # Process funding rate data
        funding_rates = funding_rates.copy()

        if 'timestamp' in funding_rates.columns and 'funding_rate' in funding_rates.columns:
            # Get funding rates by exchange
            if 'exchange' in funding_rates.columns:
                pivot_funding = funding_rates.pivot(
                    index='timestamp',
                    columns='exchange',
                    values='funding_rate'
                )
            else:
                pivot_funding = funding_rates.set_index('timestamp')[['funding_rate']]

            # Calculate funding spread
            if len(pivot_funding.columns) >= 2:
                df['funding_spread'] = (
                    pivot_funding.iloc[:, 0] - pivot_funding.iloc[:, 1]
                )

            # Cumulative funding
            df['cum_funding'] = funding_rates['funding_rate'].cumsum()

            # Funding rate momentum
            df['funding_momentum'] = funding_rates['funding_rate'].diff()

            # Funding rate extreme
            funding_mean = funding_rates['funding_rate'].rolling(100).mean()
            funding_std = funding_rates['funding_rate'].rolling(100).std()
            df['funding_extreme'] = (
                (funding_rates['funding_rate'] - funding_mean).abs() > 2 * funding_std
            ).astype(int)

        return df

    def generate_triangular_features(
        self,
        df: pd.DataFrame,
        quote_currency: str = 'USDT'
    ) -> pd.DataFrame:
        """
        Generate triangular arbitrage features.

        Args:
            df: DataFrame with multi-symbol OHLCV data
            quote_currency: Quote currency for triangular arbitrage

        Returns:
            DataFrame with triangular arbitrage features
        """
        df = df.copy()

        # This would require multiple symbol data
        # For now, generate placeholder features

        # Synthetic quote rate
        df['synthetic_quote'] = df['close'] / df['close'].shift(1)

        # Triangular inefficiency proxy
        df['tri_inefficiency'] = (
            df['synthetic_quote'].rolling(20).std() /
            df['synthetic_quote'].rolling(20).mean()
        )

        return df
