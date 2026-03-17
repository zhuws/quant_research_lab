"""
Regime Detection for Quant Research Lab.
Identifies market regimes for strategy adaptation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger


class MarketRegime(Enum):
    """Market regime types."""
    BULL = 'bull'
    BEAR = 'bear'
    SIDEWAYS = 'sideways'
    HIGH_VOLATILITY = 'high_volatility'
    LOW_VOLATILITY = 'low_volatility'
    TRENDING = 'trending'
    MEAN_REVERTING = 'mean_reverting'
    CRISIS = 'crisis'


@dataclass
class RegimeResult:
    """
    Result of regime detection.

    Attributes:
        regime: Detected regime
        probability: Probability of regime
        confidence: Confidence level
        features: Regime features
    """
    regime: MarketRegime
    probability: float = 0.0
    confidence: float = 0.0
    features: Optional[Dict[str, float]] = None


class RegimeDetector:
    """
    Detect market regimes using various methods.

    Methods:
        - Volatility regime (high/low volatility)
        - Trend regime (bull/bear/sideways)
        - Statistical regime (clustering-based)
        - Hidden Markov Model regime
    """

    def __init__(
        self,
        lookback_window: int = 60,
        volatility_threshold_high: float = 0.03,
        volatility_threshold_low: float = 0.01,
        trend_threshold: float = 0.02
    ):
        """
        Initialize Regime Detector.

        Args:
            lookback_window: Window for regime calculations
            volatility_threshold_high: Threshold for high volatility
            volatility_threshold_low: Threshold for low volatility
            trend_threshold: Threshold for trend detection
        """
        self.logger = get_logger('regime_detector')
        self.lookback_window = lookback_window
        self.volatility_threshold_high = volatility_threshold_high
        self.volatility_threshold_low = volatility_threshold_low
        self.trend_threshold = trend_threshold

        # Model storage
        self._cluster_model = None
        self._scaler = None
        self._pca = None

    def detect_regime(
        self,
        data: pd.DataFrame,
        method: str = 'combined'
    ) -> RegimeResult:
        """
        Detect current market regime.

        Args:
            data: DataFrame with OHLCV data
            method: Detection method ('volatility', 'trend', 'cluster', 'combined')

        Returns:
            RegimeResult with detected regime
        """
        if len(data) < self.lookback_window:
            self.logger.warning("Insufficient data for regime detection")
            return RegimeResult(regime=MarketRegime.SIDEWAYS)

        if method == 'volatility':
            return self._detect_volatility_regime(data)
        elif method == 'trend':
            return self._detect_trend_regime(data)
        elif method == 'cluster':
            return self._detect_cluster_regime(data)
        elif method == 'combined':
            return self._detect_combined_regime(data)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _detect_volatility_regime(self, data: pd.DataFrame) -> RegimeResult:
        """Detect volatility-based regime."""
        returns = data['close'].pct_change()
        volatility = returns.rolling(self.lookback_window).std().iloc[-1]

        # Annualize volatility (assuming 1-minute data)
        annualized_vol = volatility * np.sqrt(252 * 24 * 60)

        features = {
            'realized_volatility': volatility,
            'annualized_volatility': annualized_vol
        }

        if volatility > self.volatility_threshold_high:
            return RegimeResult(
                regime=MarketRegime.HIGH_VOLATILITY,
                probability=min(volatility / self.volatility_threshold_high, 1.0),
                confidence=0.7,
                features=features
            )
        elif volatility < self.volatility_threshold_low:
            return RegimeResult(
                regime=MarketRegime.LOW_VOLATILITY,
                probability=1.0 - volatility / self.volatility_threshold_low,
                confidence=0.7,
                features=features
            )
        else:
            return RegimeResult(
                regime=MarketRegime.SIDEWAYS,
                probability=0.5,
                confidence=0.5,
                features=features
            )

    def _detect_trend_regime(self, data: pd.DataFrame) -> RegimeResult:
        """Detect trend-based regime."""
        close = data['close']

        # Calculate returns over different periods
        returns_short = close.pct_change(self.lookback_window // 4).iloc[-1]
        returns_medium = close.pct_change(self.lookback_window // 2).iloc[-1]
        returns_long = close.pct_change(self.lookback_window).iloc[-1]

        # Calculate ADX for trend strength
        adx = self._calculate_adx(data)
        adx_value = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0

        # Moving average slope
        ma_short = close.rolling(self.lookback_window // 4).mean()
        ma_long = close.rolling(self.lookback_window).mean()
        ma_slope = (ma_short.iloc[-1] - ma_short.iloc[-5]) / ma_short.iloc[-5] if len(ma_short) > 5 else 0

        features = {
            'return_short': returns_short,
            'return_medium': returns_medium,
            'return_long': returns_long,
            'adx': adx_value,
            'ma_slope': ma_slope
        }

        # Determine regime
        avg_return = (returns_short + returns_medium + returns_long) / 3
        trend_strength = adx_value / 100 if adx_value > 25 else 0.3

        if avg_return > self.trend_threshold:
            return RegimeResult(
                regime=MarketRegime.BULL,
                probability=min(avg_return / self.trend_threshold, 1.0),
                confidence=trend_strength,
                features=features
            )
        elif avg_return < -self.trend_threshold:
            return RegimeResult(
                regime=MarketRegime.BEAR,
                probability=min(-avg_return / self.trend_threshold, 1.0),
                confidence=trend_strength,
                features=features
            )
        else:
            return RegimeResult(
                regime=MarketRegime.SIDEWAYS,
                probability=1.0 - abs(avg_return) / self.trend_threshold,
                confidence=0.5,
                features=features
            )

    def _detect_cluster_regime(self, data: pd.DataFrame) -> RegimeResult:
        """Detect regime using clustering."""
        # Prepare features for clustering
        features = self._prepare_regime_features(data)

        if len(features) < 10:
            return RegimeResult(regime=MarketRegime.SIDEWAYS)

        # Fit or use existing model
        if self._cluster_model is None:
            self._scaler = StandardScaler()
            scaled_features = self._scaler.fit_transform(features)

            # Use PCA for dimensionality reduction
            self._pca = PCA(n_components=min(3, scaled_features.shape[1]))
            pca_features = self._pca.fit_transform(scaled_features)

            # Fit clustering model
            self._cluster_model = GaussianMixture(
                n_components=4,
                covariance_type='full',
                random_state=42
            )
            self._cluster_model.fit(pca_features)

        # Predict current regime
        current_features = features.iloc[-1:].values
        scaled_current = self._scaler.transform(current_features)
        pca_current = self._pca.transform(scaled_current)

        regime_probs = self._cluster_model.predict_proba(pca_current)[0]
        regime_idx = np.argmax(regime_probs)

        # Map cluster to regime
        regime_map = {
            0: MarketRegime.LOW_VOLATILITY,
            1: MarketRegime.HIGH_VOLATILITY,
            2: MarketRegime.BULL,
            3: MarketRegime.BEAR
        }

        regime = regime_map.get(regime_idx, MarketRegime.SIDEWAYS)

        return RegimeResult(
            regime=regime,
            probability=regime_probs[regime_idx],
            confidence=regime_probs[regime_idx],
            features=features.iloc[-1].to_dict()
        )

    def _detect_combined_regime(self, data: pd.DataFrame) -> RegimeResult:
        """Detect regime using combined methods."""
        vol_result = self._detect_volatility_regime(data)
        trend_result = self._detect_trend_regime(data)

        # Combine results
        vol_regime = vol_result.regime
        trend_regime = trend_result.regime

        # Priority: crisis > high_volatility > bull/bear > low_volatility > sideways
        combined_features = {**vol_result.features, **trend_result.features}

        # Crisis detection
        returns = data['close'].pct_change()
        recent_drawdown = self._calculate_drawdown(data['close'])

        if recent_drawdown < -0.20 or vol_result.features.get('annualized_volatility', 0) > 1.0:
            return RegimeResult(
                regime=MarketRegime.CRISIS,
                probability=0.8,
                confidence=0.9,
                features=combined_features
            )

        # High volatility with trend
        if vol_regime == MarketRegime.HIGH_VOLATILITY:
            return RegimeResult(
                regime=MarketRegime.HIGH_VOLATILITY,
                probability=vol_result.probability,
                confidence=vol_result.confidence,
                features=combined_features
            )

        # Trend regime
        if trend_regime in [MarketRegime.BULL, MarketRegime.BEAR]:
            return RegimeResult(
                regime=trend_regime,
                probability=trend_result.probability,
                confidence=trend_result.confidence,
                features=combined_features
            )

        # Low volatility or sideways
        if vol_regime == MarketRegime.LOW_VOLATILITY:
            return RegimeResult(
                regime=MarketRegime.LOW_VOLATILITY,
                probability=vol_result.probability,
                confidence=vol_result.confidence,
                features=combined_features
            )

        return RegimeResult(
            regime=MarketRegime.SIDEWAYS,
            probability=0.5,
            confidence=0.5,
            features=combined_features
        )

    def _prepare_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for regime clustering."""
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']

        features = pd.DataFrame(index=data.index)

        # Return features
        features['return_1'] = close.pct_change(1)
        features['return_5'] = close.pct_change(5)
        features['return_20'] = close.pct_change(20)

        # Volatility features
        features['volatility_5'] = close.pct_change().rolling(5).std()
        features['volatility_20'] = close.pct_change().rolling(20).std()

        # Volume features
        features['volume_ratio'] = volume / volume.rolling(20).mean()

        # Range features
        features['range_ratio'] = (high - low) / close

        # Momentum features
        features['momentum_5'] = close / close.shift(5) - 1
        features['momentum_20'] = close / close.shift(20) - 1

        # Trend features
        features['price_position'] = (
            (close - low.rolling(20).min()) /
            (high.rolling(20).max() - low.rolling(20).min() + 0.0001)
        )

        # Drop NaN rows
        features = features.dropna()

        return features

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        high = data['high']
        low = data['low']
        close = data['close']

        tr = np.maximum(
            high - low,
            np.maximum(
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            )
        )

        plus_dm = np.where(
            (high - high.shift(1)) > (low.shift(1) - low),
            np.maximum(high - high.shift(1), 0),
            0
        )
        minus_dm = np.where(
            (low.shift(1) - low) > (high - high.shift(1)),
            np.maximum(low.shift(1) - low, 0),
            0
        )

        atr = pd.Series(tr).rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.rolling(period).mean()

        return adx

    def _calculate_drawdown(self, close: pd.Series) -> float:
        """Calculate current drawdown."""
        rolling_max = close.rolling(len(close), min_periods=1).max()
        drawdown = (close - rolling_max) / rolling_max
        return drawdown.iloc[-1]

    def get_regime_history(
        self,
        data: pd.DataFrame,
        method: str = 'combined'
    ) -> pd.DataFrame:
        """
        Get regime history for entire dataset.

        Args:
            data: DataFrame with OHLCV data
            method: Detection method

        Returns:
            DataFrame with regime labels and probabilities
        """
        results = []

        for i in range(self.lookback_window, len(data)):
            window_data = data.iloc[:i+1]
            result = self.detect_regime(window_data, method)

            results.append({
                'timestamp': data.index[i] if isinstance(data.index, pd.DatetimeIndex) else i,
                'regime': result.regime.value,
                'probability': result.probability,
                'confidence': result.confidence
            })

        return pd.DataFrame(results)

    def get_regime_statistics(self, regime_history: pd.DataFrame) -> Dict[str, Dict]:
        """
        Get statistics about regime occurrences.

        Args:
            regime_history: DataFrame from get_regime_history

        Returns:
            Dictionary with regime statistics
        """
        stats = {}

        for regime in regime_history['regime'].unique():
            regime_data = regime_history[regime_history['regime'] == regime]
            stats[regime] = {
                'count': len(regime_data),
                'pct_of_time': len(regime_data) / len(regime_history) * 100,
                'avg_probability': regime_data['probability'].mean(),
                'avg_confidence': regime_data['confidence'].mean()
            }

        return stats


class AdaptiveRegimeStrategy:
    """
    Strategy adaptation based on market regime.
    """

    def __init__(
        self,
        regime_detector: Optional[RegimeDetector] = None,
        strategy_weights: Optional[Dict[MarketRegime, Dict[str, float]]] = None
    ):
        """
        Initialize Adaptive Regime Strategy.

        Args:
            regime_detector: RegimeDetector instance
            strategy_weights: Strategy weights for each regime
        """
        self.logger = get_logger('adaptive_regime_strategy')
        self.regime_detector = regime_detector or RegimeDetector()

        # Default strategy weights
        self.strategy_weights = strategy_weights or {
            MarketRegime.BULL: {
                'momentum': 0.8,
                'mean_reversion': 0.1,
                'volatility_breakout': 0.1
            },
            MarketRegime.BEAR: {
                'momentum': 0.3,
                'mean_reversion': 0.5,
                'volatility_breakout': 0.2
            },
            MarketRegime.SIDEWAYS: {
                'momentum': 0.2,
                'mean_reversion': 0.7,
                'volatility_breakout': 0.1
            },
            MarketRegime.HIGH_VOLATILITY: {
                'momentum': 0.3,
                'mean_reversion': 0.3,
                'volatility_breakout': 0.4
            },
            MarketRegime.LOW_VOLATILITY: {
                'momentum': 0.5,
                'mean_reversion': 0.4,
                'volatility_breakout': 0.1
            },
            MarketRegime.CRISIS: {
                'momentum': 0.1,
                'mean_reversion': 0.3,
                'volatility_breakout': 0.1
            }
        }

    def get_strategy_weights(
        self,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Get strategy weights based on current regime.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary of strategy weights
        """
        result = self.regime_detector.detect_regime(data)

        weights = self.strategy_weights.get(
            result.regime,
            {'momentum': 0.33, 'mean_reversion': 0.33, 'volatility_breakout': 0.34}
        )

        # Adjust weights based on confidence
        confidence = result.confidence
        adjusted_weights = {}

        for strategy, weight in weights.items():
            # Blend with equal weights if confidence is low
            adjusted_weights[strategy] = (
                confidence * weight +
                (1 - confidence) * 0.33
            )

        # Normalize
        total = sum(adjusted_weights.values())
        adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}

        return adjusted_weights

    def get_position_multiplier(
        self,
        data: pd.DataFrame
    ) -> float:
        """
        Get position size multiplier based on regime.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Position size multiplier (0.0 - 1.0)
        """
        result = self.regime_detector.detect_regime(data)

        # Position multipliers for each regime
        multipliers = {
            MarketRegime.BULL: 1.0,
            MarketRegime.BEAR: 0.5,
            MarketRegime.SIDEWAYS: 0.7,
            MarketRegime.HIGH_VOLATILITY: 0.4,
            MarketRegime.LOW_VOLATILITY: 1.0,
            MarketRegime.CRISIS: 0.1,
            MarketRegime.TRENDING: 0.9,
            MarketRegime.MEAN_REVERTING: 0.8
        }

        base_multiplier = multipliers.get(result.regime, 0.5)

        # Adjust by confidence
        adjusted_multiplier = base_multiplier * (0.5 + 0.5 * result.confidence)

        return adjusted_multiplier


def detect_regime(
    data: pd.DataFrame,
    method: str = 'combined'
) -> RegimeResult:
    """
    Convenience function for regime detection.

    Args:
        data: DataFrame with OHLCV data
        method: Detection method

    Returns:
        RegimeResult with detected regime
    """
    detector = RegimeDetector()
    return detector.detect_regime(data, method)
