"""
Liquidity Feature Generator for Quant Research Lab.
Generates orderbook and liquidity-based features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger


class LiquidityFeatureGenerator:
    """
    Generates liquidity-based features from OHLCV and orderbook data.

    Feature Categories:
        - Orderbook imbalance
        - Micro price estimation
        - Spread analysis
        - Liquidity depth features
        - Price impact estimation

    Total Features: ~25 liquidity indicators
    """

    def __init__(
        self,
        depth_levels: int = 10,
        imbalance_periods: List[int] = None
    ):
        """
        Initialize Liquidity Feature Generator.

        Args:
            depth_levels: Number of orderbook depth levels
            imbalance_periods: Periods for imbalance calculations
        """
        self.logger = get_logger('liquidity_features')

        self.depth_levels = depth_levels
        self.imbalance_periods = imbalance_periods or [5, 10, 20]

    def generate_features(
        self,
        df: pd.DataFrame,
        orderbook_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate all liquidity features.

        Args:
            df: DataFrame with OHLCV data
            orderbook_df: Optional DataFrame with orderbook data

        Returns:
            DataFrame with liquidity features added
        """
        if df.empty:
            return df

        self.logger.info(f"Generating liquidity features from {len(df)} bars")

        features = df.copy()

        # Generate proxy features from OHLCV
        features = self._generate_spread_proxy_features(features)
        features = self._generate_liquidity_proxy_features(features)

        # If orderbook data available, generate additional features
        if orderbook_df is not None and not orderbook_df.empty:
            features = self._generate_orderbook_features(features, orderbook_df)

        # Replace inf values
        features = features.replace([np.inf, -np.inf], np.nan)

        self.logger.info(f"Generated {len(features.columns) - len(df.columns)} liquidity features")

        return features

    def _generate_spread_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate spread proxy features from OHLCV."""
        # Bid-Ask spread proxy (using high-low as approximation)
        df['spread_proxy'] = df['high'] - df['low']
        df['spread_proxy_pct'] = df['spread_proxy'] / df['close']

        for period in self.imbalance_periods:
            # Relative spread
            df[f'rel_spread_{period}'] = (
                df['spread_proxy'].rolling(period).mean() / df['close'].rolling(period).mean()
            )

            # Spread volatility
            df[f'spread_volatility_{period}'] = (
                df['spread_proxy'].rolling(period).std() / df['spread_proxy'].rolling(period).mean()
            )

        # Effective spread estimate
        # Using close-to-midpoint approximation
        midpoint = (df['high'] + df['low']) / 2
        df['effective_spread'] = 2 * abs(df['close'] - midpoint)
        df['effective_spread_pct'] = df['effective_spread'] / df['close']

        # Realized spread (price impact)
        for period in [5, 10]:
            future_mid = (df['high'].shift(-period) + df['low'].shift(-period)) / 2
            df[f'realized_spread_{period}'] = 2 * (df['close'] - future_mid).abs()
            df[f'realized_spread_pct_{period}'] = (
                df[f'realized_spread_{period}'] / df['close']
            )

        return df

    def _generate_liquidity_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate liquidity proxy features from OHLCV."""
        # Amihud illiquidity ratio
        for period in [20, 60]:
            daily_return = abs(df['close'].pct_change())
            dollar_volume = df['close'] * df['volume']

            df[f'amihud_illiq_{period}'] = (
                daily_return.rolling(period).mean() /
                dollar_volume.rolling(period).mean()
            )

        # Kyle's lambda proxy (price impact per unit volume)
        for period in [20]:
            returns = df['close'].pct_change().abs()
            df[f'kyle_lambda_{period}'] = (
                returns.rolling(period).mean() /
                df['volume'].rolling(period).mean().replace(0, np.nan)
            )

        # Volume-to-volatility ratio (inverse of illiquidity)
        for period in [20]:
            vol = df['close'].pct_change().rolling(period).std()
            volume = df['volume'].rolling(period).mean()
            df[f'vol_to_volume_{period}'] = volume / vol.replace(0, np.nan)

        # Liquidity ratio
        for period in [10, 20]:
            df[f'liquidity_ratio_{period}'] = (
                df['volume'].rolling(period).sum() /
                abs(df['close'].pct_change()).rolling(period).sum().replace(0, np.nan)
            )

        # Turnover ratio
        for period in [20]:
            df[f'turnover_ratio_{period}'] = (
                df['volume'].rolling(period).sum() /
                df['volume'].rolling(252).sum().replace(0, np.nan)
            ) if len(df) >= 252 else df['volume'] / df['volume'].rolling(period).mean()

        # Market depth proxy
        for period in [10]:
            avg_range = (df['high'] - df['low']).rolling(period).mean()
            avg_volume = df['volume'].rolling(period).mean()
            df[f'depth_proxy_{period}'] = avg_volume / avg_range.replace(0, np.nan)

        # Order flow toxicity (VPIN proxy)
        for period in [50]:
            # Simplified VPIN using volume imbalance
            close_position = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
            order_imbalance = abs(df['volume'] * (2 * close_position - 1))
            df[f'vpin_proxy_{period}'] = order_imbalance.rolling(period).sum() / df['volume'].rolling(period).sum()

        return df

    def _generate_orderbook_features(
        self,
        df: pd.DataFrame,
        orderbook_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate features from orderbook data."""
        if orderbook_df.empty:
            return df

        features = df.copy()
        ob = orderbook_df.copy()

        # Parse bids/asks if stored as JSON
        if 'bids' in ob.columns and 'asks' in ob.columns:
            for idx, row in ob.iterrows():
                try:
                    bids = row['bids'] if isinstance(row['bids'], list) else eval(row['bids'])
                    asks = row['asks'] if isinstance(row['asks'], list) else eval(row['asks'])

                    # Calculate orderbook features
                    ob_features = self._calculate_orderbook_metrics(bids, asks)

                    for key, value in ob_features.items():
                        if key not in features.columns:
                            features[key] = np.nan
                        features.loc[idx, key] = value
                except Exception as e:
                    self.logger.debug(f"Error processing orderbook row: {e}")
                    continue

        return features

    def _calculate_orderbook_metrics(
        self,
        bids: List[List[float]],
        asks: List[List[float]]
    ) -> Dict[str, float]:
        """Calculate orderbook metrics from bid/ask data."""
        metrics = {}

        if not bids or not asks:
            return metrics

        try:
            # Convert to numpy arrays
            bids = np.array(bids[:self.depth_levels])
            asks = np.array(asks[:self.depth_levels])

            # Best bid/ask
            best_bid = bids[0, 0]
            best_ask = asks[0, 0]
            metrics['best_bid'] = best_bid
            metrics['best_ask'] = best_ask

            # Spread
            metrics['spread'] = best_ask - best_bid
            metrics['spread_pct'] = (best_ask - best_bid) / best_bid
            metrics['midpoint'] = (best_bid + best_ask) / 2

            # Bid/Ask volumes
            bid_volume = bids[:, 1].sum()
            ask_volume = asks[:, 1].sum()

            # Orderbook imbalance
            metrics['ob_imbalance'] = (bid_volume - ask_volume) / (bid_volume + ask_volume)

            # Micro price
            metrics['micro_price'] = (
                best_ask * bid_volume + best_bid * ask_volume
            ) / (bid_volume + ask_volume)

            # Weighted spread
            metrics['weighted_spread'] = metrics['micro_price'] - metrics['midpoint']

            # Depth at different levels
            for level in [1, 5, 10]:
                if len(bids) >= level and len(asks) >= level:
                    metrics[f'bid_depth_{level}'] = bids[:level, 1].sum()
                    metrics[f'ask_depth_{level}'] = asks[:level, 1].sum()
                    metrics[f'total_depth_{level}'] = metrics[f'bid_depth_{level}'] + metrics[f'ask_depth_{level}']

            # Depth imbalance at different levels
            for level in [5, 10]:
                if f'bid_depth_{level}' in metrics and f'ask_depth_{level}' in metrics:
                    metrics[f'depth_imbalance_{level}'] = (
                        metrics[f'bid_depth_{level}'] - metrics[f'ask_depth_{level}']
                    ) / (metrics[f'bid_depth_{level}'] + metrics[f'ask_depth_{level}'])

            # Price levels (range of prices in top N levels)
            metrics['bid_price_range'] = bids[0, 0] - bids[-1, 0] if len(bids) > 1 else 0
            metrics['ask_price_range'] = asks[-1, 0] - asks[0, 0] if len(asks) > 1 else 0

            # Volume-weighted average price of bids/asks
            metrics['vwap_bids'] = (bids[:, 0] * bids[:, 1]).sum() / bid_volume
            metrics['vwap_asks'] = (asks[:, 0] * asks[:, 1]).sum() / ask_volume

            # Slope of orderbook (liquidity gradient)
            if len(bids) >= 2 and len(asks) >= 2:
                bid_prices = bids[:, 0]
                bid_volumes = bids[:, 1]
                ask_prices = asks[:, 0]
                ask_volumes = asks[:, 1]

                # Linear regression slope
                from numpy.polynomial import polynomial as P
                try:
                    bid_coef = np.polyfit(bid_prices, bid_volumes, 1)
                    ask_coef = np.polyfit(ask_prices, ask_volumes, 1)
                    metrics['bid_slope'] = bid_coef[0]
                    metrics['ask_slope'] = ask_coef[0]
                except:
                    metrics['bid_slope'] = 0
                    metrics['ask_slope'] = 0

            # Order flow pressure
            metrics['bid_pressure'] = bid_volume * best_bid
            metrics['ask_pressure'] = ask_volume * best_ask
            metrics['net_pressure'] = metrics['bid_pressure'] - metrics['ask_pressure']

        except Exception as e:
            self.logger.debug(f"Error calculating orderbook metrics: {e}")

        return metrics

    def generate_imbalance_features(
        self,
        df: pd.DataFrame,
        bid_col: str = 'bid_volume',
        ask_col: str = 'ask_volume'
    ) -> pd.DataFrame:
        """
        Generate orderbook imbalance features from aggregated data.

        Args:
            df: DataFrame with bid/ask volume columns
            bid_col: Column name for bid volume
            ask_col: Column name for ask volume

        Returns:
            DataFrame with imbalance features
        """
        df = df.copy()

        if bid_col in df.columns and ask_col in df.columns:
            # Orderbook imbalance
            df['ob_imbalance'] = (
                df[bid_col] - df[ask_col]
            ) / (df[bid_col] + df[ask_col]).replace(0, np.nan)

            # Imbalance momentum
            for period in self.imbalance_periods:
                df[f'ob_imbalance_ma_{period}'] = df['ob_imbalance'].rolling(period).mean()
                df[f'ob_imbalance_std_{period}'] = df['ob_imbalance'].rolling(period).std()

                # Imbalance trend
                df[f'ob_imbalance_trend_{period}'] = (
                    df['ob_imbalance'].rolling(period).apply(
                        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0
                    )
                )

            # Imbalance extreme
            imb_mean = df['ob_imbalance'].rolling(100).mean()
            imb_std = df['ob_imbalance'].rolling(100).std()
            df['ob_imbalance_zscore'] = (df['ob_imbalance'] - imb_mean) / imb_std.replace(0, np.nan)

            # Imbalance regime
            df['bid_dominant'] = (df['ob_imbalance'] > 0.3).astype(int)
            df['ask_dominant'] = (df['ob_imbalance'] < -0.3).astype(int)
            df['balanced'] = (df['ob_imbalance'].abs() < 0.1).astype(int)

        return df

    def generate_auction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate auction-based features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with auction features
        """
        df = df.copy()

        # Opening auction proxy (first bar of session)
        if 'timestamp' in df.columns:
            df['is_open_auction'] = (
                df['timestamp'].dt.hour == 0) & (df['timestamp'].dt.minute == 0
            ).astype(int)

            # Closing auction proxy (last bar of session)
            df['is_close_auction'] = (
                (df['timestamp'].dt.hour == 23) & (df['timestamp'].dt.minute == 59)
            ).astype(int)

        # Auction imbalance proxy
        # Using the ratio of close position in range
        close_position = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
        df['auction_imbalance_proxy'] = close_position * df['volume']

        return df
