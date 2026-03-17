"""
Order Flow Feature Generator for Quant Research Lab.
Generates trade flow and order flow analysis features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger


class OrderFlowFeatureGenerator:
    """
    Generates order flow-based features from OHLCV and trade data.

    Feature Categories:
        - Volume profile features
        - Trade imbalance features
        - VWAP analysis
        - Volume at Price analysis
        - Aggressive order detection

    Total Features: ~30 order flow indicators
    """

    def __init__(
        self,
        flow_periods: List[int] = None,
        vwap_periods: List[int] = None
    ):
        """
        Initialize Order Flow Feature Generator.

        Args:
            flow_periods: Periods for flow calculations
            vwap_periods: Periods for VWAP calculations
        """
        self.logger = get_logger('orderflow_features')

        self.flow_periods = flow_periods or [5, 10, 20, 60]
        self.vwap_periods = vwap_periods or [20, 50, 100]

    def generate_features(
        self,
        df: pd.DataFrame,
        trades_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate all order flow features.

        Args:
            df: DataFrame with OHLCV data
            trades_df: Optional DataFrame with trade data (timestamp, price, quantity, side)

        Returns:
            DataFrame with order flow features added
        """
        if df.empty:
            return df

        self.logger.info(f"Generating order flow features from {len(df)} bars")

        features = df.copy()

        # Generate features from OHLCV
        features = self._generate_volume_profile_features(features)
        features = self._generate_vwap_features(features)
        features = self._generate_trade_imbalance_features(features)
        features = self._generate_pressure_features(features)
        features = self._generate_flow_momentum_features(features)

        # If trade data available, generate additional features
        if trades_df is not None and not trades_df.empty:
            features = self._generate_trade_level_features(features, trades_df)

        # Replace inf values
        features = features.replace([np.inf, -np.inf], np.nan)

        self.logger.info(f"Generated {len(features.columns) - len(df.columns)} order flow features")

        return features

    def _generate_volume_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volume profile features."""
        # Volume at different price levels relative to current
        for period in self.flow_periods:
            # Volume-weighted price levels
            vwp = (df['close'] * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
            df[f'vwp_{period}'] = vwp
            df[f'price_to_vwp_{period}'] = df['close'] / vwp - 1

            # Volume concentration
            vol_mean = df['volume'].rolling(period).mean()
            vol_std = df['volume'].rolling(period).std()
            df[f'vol_concentration_{period}'] = (df['volume'] - vol_mean) / vol_std.replace(0, np.nan)

            # High volume nodes (where volume is above average)
            df[f'high_vol_node_{period}'] = (
                df['volume'] > vol_mean + vol_std
            ).astype(int)

            # Low volume nodes
            df[f'low_vol_node_{period}'] = (
                df['volume'] < vol_mean - vol_std
            ).astype(int)

        # Volume delta (buying vs selling pressure approximation)
        # Using close position in range as proxy
        close_position = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
        df['volume_delta'] = df['volume'] * (2 * close_position - 1)  # -volume to +volume

        for period in self.flow_periods:
            df[f'cum_vol_delta_{period}'] = df['volume_delta'].rolling(period).sum()
            df[f'vol_delta_ratio_{period}'] = (
                df['volume_delta'].rolling(period).sum() /
                df['volume'].rolling(period).sum()
            )

        # Volume at Close (late session volume importance)
        df['vol_at_close'] = df['volume'] * (df['close'] == df['close']).astype(int)
        df['late_vol_ratio'] = df['volume'] / df['volume'].shift(1).replace(0, np.nan)

        return df

    def _generate_vwap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate VWAP-based features."""
        # Standard VWAP (cumulative from session start)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_tp_v = (typical_price * df['volume']).cumsum()
        cumulative_v = df['volume'].cumsum()
        df['vwap_session'] = cumulative_tp_v / cumulative_v

        # Rolling VWAP
        for period in self.vwap_periods:
            df[f'vwap_{period}'] = (
                (typical_price * df['volume']).rolling(period).sum() /
                df['volume'].rolling(period).sum()
            )
            df[f'price_to_vwap_{period}'] = df['close'] / df[f'vwap_{period}'] - 1

            # VWAP bands
            vol_std = df['volume'].rolling(period).std()
            df[f'vwap_upper_{period}'] = df[f'vwap_{period}'] + vol_std
            df[f'vwap_lower_{period}'] = df[f'vwap_{period}'] - vol_std

            # VWAP slope (trend direction)
            df[f'vwap_slope_{period}'] = (
                df[f'vwap_{period}'] - df[f'vwap_{period}'].shift(5)
            ) / df[f'vwap_{period}'].shift(5)

        # VWAP cross signals
        df['vwap_cross_up'] = (
            (df['close'] > df['vwap_session']) &
            (df['close'].shift(1) <= df['vwap_session'].shift(1))
        ).astype(int)
        df['vwap_cross_down'] = (
            (df['close'] < df['vwap_session']) &
            (df['close'].shift(1) >= df['vwap_session'].shift(1))
        ).astype(int)

        # VWAP distance
        df['vwap_distance'] = (df['close'] - df['vwap_session']) / df['vwap_session']

        return df

    def _generate_trade_imbalance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trade imbalance features from OHLCV proxy."""
        # Buy/Sell pressure proxy based on candle analysis
        # Positive when close > open, negative otherwise
        candle_direction = np.sign(df['close'] - df['open'])
        candle_magnitude = abs(df['close'] - df['open'])

        # Imbalance weighted by volume
        df['trade_imbalance'] = candle_direction * df['volume'] * candle_magnitude / df['close']

        for period in self.flow_periods:
            # Cumulative imbalance
            df[f'imbalance_sum_{period}'] = df['trade_imbalance'].rolling(period).sum()
            df[f'imbalance_mean_{period}'] = df['trade_imbalance'].rolling(period).mean()

            # Imbalance ratio (net vs total volume)
            df[f'imbalance_ratio_{period}'] = (
                df[f'imbalance_sum_{period}'] /
                df['volume'].rolling(period).sum().replace(0, np.nan)
            )

        # Trade intensity (volume per unit time, assuming regular intervals)
        for period in [5, 10, 20]:
            df[f'trade_intensity_{period}'] = df['volume'].rolling(period).sum() / period

        # Large trade detection (volume spikes)
        vol_mean = df['volume'].rolling(20).mean()
        vol_std = df['volume'].rolling(20).std()
        df['large_trade'] = (df['volume'] > vol_mean + 2 * vol_std).astype(int)

        # Cluster detection (multiple large trades)
        df['large_trade_cluster'] = df['large_trade'].rolling(5).sum()

        return df

    def _generate_pressure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate buying/selling pressure features."""
        # Buying pressure: high - max(open, close)
        # Selling pressure: min(open, close) - low
        buying_pressure = df['high'] - df[['open', 'close']].max(axis=1)
        selling_pressure = df[['open', 'close']].min(axis=1) - df['low']

        df['buying_pressure'] = buying_pressure
        df['selling_pressure'] = selling_pressure

        # Pressure ratio
        df['pressure_ratio'] = buying_pressure / selling_pressure.replace(0, np.nan)

        # Volume-weighted pressure
        df['vw_buying_pressure'] = buying_pressure * df['volume']
        df['vw_selling_pressure'] = selling_pressure * df['volume']

        for period in self.flow_periods:
            df[f'net_pressure_{period}'] = (
                df['vw_buying_pressure'].rolling(period).sum() -
                df['vw_selling_pressure'].rolling(period).sum()
            )

            # Pressure trend
            df[f'pressure_trend_{period}'] = (
                df['pressure_ratio'].rolling(period).mean() > 1
            ).astype(int)

        # Aggressive buying/selling detection
        # Large move up with high volume
        df['aggressive_buy'] = (
            (df['close'] > df['open']) &
            (df['close'] - df['open'] > df['high'] - df['low']) &
            (df['volume'] > df['volume'].rolling(10).mean())
        ).astype(int)

        # Large move down with high volume
        df['aggressive_sell'] = (
            (df['close'] < df['open']) &
            (df['open'] - df['close'] > df['high'] - df['low']) &
            (df['volume'] > df['volume'].rolling(10).mean())
        ).astype(int)

        return df

    def _generate_flow_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate order flow momentum features."""
        # Volume momentum
        for period in [5, 10]:
            df[f'vol_momentum_{period}'] = (
                df['volume'].rolling(period).sum() /
                df['volume'].shift(period).rolling(period).sum().replace(0, np.nan)
            )

        # Price-volume trend
        pv_trend = (df['close'].pct_change() * df['volume']).cumsum()
        df['pv_trend'] = pv_trend
        df['pv_trend_ma'] = pv_trend.rolling(20).mean()

        # Ease of Movement
        distance = (df['high'] + df['low']) / 2 - (df['high'].shift(1) + df['low'].shift(1)) / 2
        box_ratio = df['volume'] / (df['high'] - df['low']).replace(0, np.nan)
        df['ease_of_movement'] = distance / box_ratio
        df['ease_of_movement_ma'] = df['ease_of_movement'].rolling(14).mean()

        # Volume Flow Indicator (VFI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        interim = np.where(
            typical_price > typical_price.shift(1),
            df['volume'],
            np.where(
                typical_price < typical_price.shift(1),
                -df['volume'],
                0
            )
        )
        df['vfi'] = pd.Series(interim).ewm(span=20).mean()
        df['vfi_signal'] = (df['vfi'] > df['vfi'].shift(1)).astype(int)

        # Williams Accumulation/Distribution
        trh = np.maximum(df['high'], df['close'].shift(1))
        trl = np.minimum(df['low'], df['close'].shift(1))
        ad = np.where(
            df['close'] > df['close'].shift(1),
            df['close'] - trl,
            np.where(
                df['close'] < df['close'].shift(1),
                df['close'] - trh,
                0
            )
        )
        df['williams_ad'] = pd.Series(ad).cumsum()

        return df

    def _generate_trade_level_features(
        self,
        df: pd.DataFrame,
        trades_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate features from individual trade data."""
        if trades_df.empty:
            return df

        # Aggregate trade data to match OHLCV timestamps
        trades_df = trades_df.copy()

        # Buy volume vs Sell volume
        if 'side' in trades_df.columns:
            buy_vol = trades_df[trades_df['side'] == 'buy']['quantity'].sum()
            sell_vol = trades_df[trades_df['side'] == 'sell']['quantity'].sum()
            df['trade_buy_vol_ratio'] = buy_vol / (buy_vol + sell_vol).replace(0, np.nan)

        # Trade count
        if 'trade_id' in trades_df.columns:
            df['trade_count'] = len(trades_df)

        # Average trade size
        if 'quantity' in trades_df.columns:
            df['avg_trade_size'] = trades_df['quantity'].mean()
            df['trade_size_std'] = trades_df['quantity'].std()

        return df


    def generate_intraday_features(
        self,
        df: pd.DataFrame,
        session_start: int = 0,
        session_end: int = 24
    ) -> pd.DataFrame:
        """
        Generate intraday order flow features.

        Args:
            df: DataFrame with timestamp and OHLCV
            session_start: Session start hour
            session_end: Session end hour

        Returns:
            DataFrame with intraday features
        """
        if df.empty or 'timestamp' not in df.columns:
            return df

        df = df.copy()

        # Extract time features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute

        # Session volume profile
        session_mask = (df['hour'] >= session_start) & (df['hour'] < session_end)
        df['is_session'] = session_mask.astype(int)

        # Volume by hour
        hourly_vol = df.groupby('hour')['volume'].transform('mean')
        df['vol_vs_hourly_avg'] = df['volume'] / hourly_vol

        # Opening range volume (first 30 minutes)
        opening_mask = (df['hour'] == session_start) & (df['minute'] < 30)
        df['is_opening'] = opening_mask.astype(int)

        # Closing range volume (last 30 minutes)
        closing_mask = (df['hour'] == session_end - 1) & (df['minute'] >= 30)
        df['is_closing'] = closing_mask.astype(int)

        return df
