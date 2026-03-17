"""
Technical Feature Generator for Quant Research Lab.
Generates momentum, volume, and price pattern features from OHLCV data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger


class TechnicalFeatureGenerator:
    """
    Generates technical analysis features from OHLCV data.

    Feature Categories:
        - Momentum features (price changes, ROC, RSI, MACD)
        - Volume features (volume changes, OBV, VWAP)
        - Price pattern features (candlestick patterns, support/resistance)
        - Trend features (moving averages, ADX)

    Total Features: ~70 technical indicators
    """

    def __init__(
        self,
        momentum_periods: List[int] = None,
        ma_periods: List[int] = None,
        rsi_periods: List[int] = None,
        macd_params: Tuple[int, int, int] = (12, 26, 9)
    ):
        """
        Initialize Technical Feature Generator.

        Args:
            momentum_periods: Periods for momentum calculations
            ma_periods: Periods for moving averages
            rsi_periods: Periods for RSI calculations
            macd_params: (fast, slow, signal) for MACD
        """
        self.logger = get_logger('technical_features')

        self.momentum_periods = momentum_periods or [1, 3, 5, 10, 20, 60, 120]
        self.ma_periods = ma_periods or [5, 10, 20, 60, 120, 200]
        self.rsi_periods = rsi_periods or [7, 14, 21]
        self.macd_params = macd_params

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all technical features.

        Args:
            df: DataFrame with OHLCV data (timestamp, open, high, low, close, volume)

        Returns:
            DataFrame with technical features added
        """
        if df.empty:
            return df

        self.logger.info(f"Generating technical features from {len(df)} bars")

        features = df.copy()

        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in features.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Generate feature categories
        features = self._generate_price_features(features)
        features = self._generate_momentum_features(features)
        features = self._generate_ma_features(features)
        features = self._generate_rsi_features(features)
        features = self._generate_macd_features(features)
        features = self._generate_volume_features(features)
        features = self._generate_oscillator_features(features)
        features = self._generate_pattern_features(features)
        features = self._generate_trend_features(features)

        # Drop NaN from initial calculations
        features = features.replace([np.inf, -np.inf], np.nan)

        self.logger.info(f"Generated {len(features.columns) - len(df.columns)} technical features")

        return features

    def _generate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate basic price-based features."""
        # Returns at different periods
        for period in self.momentum_periods:
            # Simple returns
            df[f'return_{period}'] = df['close'].pct_change(period)

            # Log returns
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))

            # Price position (close relative to high-low range)
            df[f'price_position_{period}'] = (
                (df['close'] - df['low'].rolling(period).min()) /
                (df['high'].rolling(period).max() - df['low'].rolling(period).min())
            )

        # Candlestick features
        df['body_size'] = (df['close'] - df['open']).abs()
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['candle_range'] = df['high'] - df['low']

        # Relative candle features
        df['body_ratio'] = df['body_size'] / df['candle_range'].replace(0, np.nan)
        df['upper_shadow_ratio'] = df['upper_shadow'] / df['candle_range'].replace(0, np.nan)
        df['lower_shadow_ratio'] = df['lower_shadow'] / df['candle_range'].replace(0, np.nan)

        # Gap features
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_ratio'] = df['gap'] / df['close'].shift(1)

        # Price relative to OHLC
        df['close_to_open'] = df['close'] / df['open'] - 1
        df['high_to_low'] = df['high'] / df['low'] - 1
        df['close_to_high'] = df['close'] / df['high'] - 1
        df['close_to_low'] = df['close'] / df['low'] - 1

        return df

    def _generate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum indicators."""
        for period in self.momentum_periods:
            # Rate of Change
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100

            # Momentum
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)

            # Price acceleration
            df[f'acceleration_{period}'] = (
                df[f'momentum_{period}'] - df[f'momentum_{period}'].shift(period)
            )

            # Momentum divergence (price vs momentum direction)
            price_up = df['close'] > df['close'].shift(period)
            mom_up = df[f'momentum_{period}'] > df[f'momentum_{period}'].shift(period)
            df[f'momentum_divergence_{period}'] = (price_up != mom_up).astype(int)

        # Williams %R
        for period in [14, 21]:
            highest = df['high'].rolling(period).max()
            lowest = df['low'].rolling(period).min()
            df[f'williams_r_{period}'] = (highest - df['close']) / (highest - lowest) * -100

        # Stochastic Oscillator
        for period in [14, 21]:
            lowest_low = df['low'].rolling(period).min()
            highest_high = df['high'].rolling(period).max()
            df[f'stoch_k_{period}'] = (
                (df['close'] - lowest_low) / (highest_high - lowest_low) * 100
            )
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(3).mean()

        # Commodity Channel Index
        for period in [14, 21]:
            tp = (df['high'] + df['low'] + df['close']) / 3
            ma = tp.rolling(period).mean()
            md = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
            df[f'cci_{period}'] = (tp - ma) / (0.015 * md)

        return df

    def _generate_ma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate moving average features."""
        # Simple Moving Averages
        for period in self.ma_periods:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'sma_ratio_{period}'] = df['close'] / df[f'sma_{period}'] - 1

        # Exponential Moving Averages
        for period in self.ma_periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'ema_ratio_{period}'] = df['close'] / df[f'ema_{period}'] - 1

        # MA crossovers
        for i, fast in enumerate(self.ma_periods[:-1]):
            for slow in self.ma_periods[i+1:]:
                df[f'sma_cross_{fast}_{slow}'] = (
                    df[f'sma_{fast}'] - df[f'sma_{slow}']
                ) / df[f'sma_{slow}']
                df[f'ema_cross_{fast}_{slow}'] = (
                    df[f'ema_{fast}'] - df[f'ema_{slow}']
                ) / df[f'ema_{slow}']

        # Displacement from MA
        for period in [20, 60]:
            df[f'dist_from_sma_{period}'] = (
                (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
            )
            # Z-score relative to MA
            df[f'zscore_sma_{period}'] = (
                (df['close'] - df[f'sma_{period}']) /
                df['close'].rolling(period).std()
            )

        return df

    def _generate_rsi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI indicators."""
        for period in self.rsi_periods:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

            rs = gain / loss.replace(0, np.nan)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

            # RSI overbought/oversold signals
            df[f'rsi_overbought_{period}'] = (df[f'rsi_{period}'] > 70).astype(int)
            df[f'rsi_oversold_{period}'] = (df[f'rsi_{period}'] < 30).astype(int)

            # RSI divergence
            df[f'rsi_divergence_{period}'] = (
                df[f'rsi_{period}'].diff().rolling(5).corr(df['close'].diff())
            )

        return df

    def _generate_macd_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate MACD indicators."""
        fast, slow, signal = self.macd_params

        # MACD Line
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        df['macd_line'] = ema_fast - ema_slow

        # Signal Line
        df['macd_signal'] = df['macd_line'].ewm(span=signal, adjust=False).mean()

        # Histogram
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']

        # MACD features
        df['macd_cross'] = (df['macd_line'] > df['macd_signal']).astype(int)
        df['macd_cross_signal'] = (
            (df['macd_line'] > df['macd_signal']) &
            (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))
        ).astype(int)

        # MACD momentum
        df['macd_momentum'] = df['macd_histogram'].diff()

        return df

    def _generate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-based features."""
        # Volume moving averages
        for period in [5, 10, 20, 60]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']

        # Volume momentum
        for period in [1, 5, 10]:
            df[f'volume_change_{period}'] = df['volume'].pct_change(period)
            df[f'volume_momentum_{period}'] = df['volume'] - df['volume'].shift(period)

        # On-Balance Volume
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv'] = obv
        df['obv_sma_20'] = obv.rolling(20).mean()
        df['obv_trend'] = obv - obv.rolling(20).mean()

        # Volume Price Trend
        vpt = (df['close'].pct_change() * df['volume']).fillna(0).cumsum()
        df['vpt'] = vpt
        df['vpt_sma_20'] = vpt.rolling(20).mean()

        # Accumulation/Distribution Line
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
              (df['high'] - df['low']).replace(0, np.nan)
        ad = (clv * df['volume']).fillna(0).cumsum()
        df['ad_line'] = ad

        # Chaikin Money Flow
        for period in [20]:
            mf_multiplier = (
                (df['close'] - df['low']) - (df['high'] - df['close'])
            ) / (df['high'] - df['low']).replace(0, np.nan)
            mf_volume = mf_multiplier * df['volume']
            df[f'cmf_{period}'] = mf_volume.rolling(period).sum() / df['volume'].rolling(period).sum()

        # VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_tp_v = (typical_price * df['volume']).cumsum()
        cumulative_v = df['volume'].cumsum()
        df['vwap'] = cumulative_tp_v / cumulative_v
        df['vwap_ratio'] = df['close'] / df['vwap'] - 1

        # Money Flow Index
        for period in [14]:
            tp = (df['high'] + df['low'] + df['close']) / 3
            mf = tp * df['volume']

            positive_mf = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
            negative_mf = mf.where(tp < tp.shift(1), 0).rolling(period).sum()

            mfi_ratio = positive_mf / negative_mf.replace(0, np.nan)
            df[f'mfi_{period}'] = 100 - (100 / (1 + mfi_ratio))

        return df

    def _generate_oscillator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate oscillator indicators."""
        # Awesome Oscillator
        median_price = (df['high'] + df['low']) / 2
        ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()
        df['awesome_oscillator'] = ao
        df['ao_saucer'] = ((ao > ao.shift(1)) & (ao.shift(1) > ao.shift(2)) & (ao < 0)).astype(int)

        # Accelerator Oscillator
        df['accelerator_osc'] = ao - ao.rolling(5).mean()

        # Ultimate Oscillator
        bp = df['close'] - np.minimum(df['low'], df['close'].shift(1))
        tr = np.maximum(df['high'], df['close'].shift(1)) - np.minimum(df['low'], df['close'].shift(1))

        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()

        df['ultimate_oscillator'] = 100 * ((4 * avg7 + 2 * avg14 + avg28) / 7)

        # Detrended Price Oscillator
        for period in [20]:
            df[f'dpo_{period}'] = df['close'] - df['close'].rolling(period).mean().shift(int(period/2) + 1)

        return df

    def _generate_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate candlestick pattern features."""
        # Doji patterns
        doji_threshold = 0.1
        df['is_doji'] = (
            df['body_size'] / df['candle_range'].replace(0, np.nan) < doji_threshold
        ).astype(int)

        # Engulfing patterns
        df['is_bullish_engulfing'] = (
            (df['close'] > df['open']) &
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1))
        ).astype(int)

        df['is_bearish_engulfing'] = (
            (df['close'] < df['open']) &
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < df['open'].shift(1))
        ).astype(int)

        # Hammer / Hanging Man
        df['is_hammer'] = (
            (df['lower_shadow'] > df['body_size'] * 2) &
            (df['upper_shadow'] < df['body_size'] * 0.5) &
            (df['candle_range'] > 0)
        ).astype(int)

        # Morning / Evening Star
        df['is_morning_star'] = (
            (df['close'].shift(2) < df['open'].shift(2)) &  # First bearish
            (df['body_size'].shift(1) < df['body_size'].shift(2) * 0.3) &  # Small body
            (df['close'] > df['open']) &  # Third bullish
            (df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2)  # Closes above midpoint
        ).astype(int)

        # Three white soldiers / Three black crows
        df['is_three_white'] = (
            (df['close'] > df['open']) &
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'].shift(2) > df['open'].shift(2)) &
            (df['close'] > df['close'].shift(1)) &
            (df['close'].shift(1) > df['close'].shift(2))
        ).astype(int)

        df['is_three_black'] = (
            (df['close'] < df['open']) &
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'].shift(2) < df['open'].shift(2)) &
            (df['close'] < df['close'].shift(1)) &
            (df['close'].shift(1) < df['close'].shift(2))
        ).astype(int)

        # Inside bar
        df['is_inside_bar'] = (
            (df['high'] < df['high'].shift(1)) &
            (df['low'] > df['low'].shift(1))
        ).astype(int)

        # Outside bar
        df['is_outside_bar'] = (
            (df['high'] > df['high'].shift(1)) &
            (df['low'] < df['low'].shift(1))
        ).astype(int)

        # Higher highs / Higher lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['lower_high'] = (df['high'] < df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)

        return df

    def _generate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend indicators."""
        # Average Directional Index (ADX)
        for period in [14]:
            # True Range
            tr = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )

            # Directional Movement
            plus_dm = np.where(
                (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                np.maximum(df['high'] - df['high'].shift(1), 0),
                0
            )
            minus_dm = np.where(
                (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                np.maximum(df['low'].shift(1) - df['low'], 0),
                0
            )

            # Smoothed values
            atr = pd.Series(tr).rolling(period).mean()
            plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

            # ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df[f'adx_{period}'] = dx.rolling(period).mean()
            df[f'plus_di_{period}'] = plus_di
            df[f'minus_di_{period}'] = minus_di

            # Trend direction
            df[f'trend_direction_{period}'] = (plus_di > minus_di).astype(int)

        # Parabolic SAR (simplified)
        af = 0.02
        max_af = 0.2

        # Calculate SAR using rolling high/low as approximation
        df['sar'] = df['close'].rolling(4).mean()  # Simplified SAR

        # SuperTrend (simplified)
        for period in [10]:
            hl2 = (df['high'] + df['low']) / 2
            atr = tr if 'tr' in dir() else pd.Series(tr)
            atr = atr.rolling(period).mean()

            upper_band = hl2 + (3 * atr)
            lower_band = hl2 - (3 * atr)

            df['supertrend'] = np.where(
                df['close'] > upper_band.shift(1),
                lower_band,
                np.where(df['close'] < lower_band.shift(1), upper_band, hl2)
            )
            df['supertrend_signal'] = (df['close'] > df['supertrend']).astype(int)

        # Trend strength based on consecutive direction
        for period in [5, 10]:
            df[f'consecutive_up_{period}'] = (
                (df['close'] > df['open']).rolling(period).sum()
            )
            df[f'consecutive_down_{period}'] = (
                (df['close'] < df['open']).rolling(period).sum()
            )

        return df
