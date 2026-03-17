"""
Volatility Filter for Quant Research Lab.

Provides volatility-based trading filters and risk adjustment:
    - Real-time volatility monitoring
    - Volatility regime detection
    - Position size adjustment based on volatility
    - Trading halts during extreme volatility
    - Historical volatility analysis

Volatility filtering is essential for:
    - Adapting strategy to market conditions
    - Reducing risk during turbulent periods
    - Improving risk-adjusted returns
    - Preventing catastrophic losses

Example usage:
    ```python
    from risk.volatility_filter import VolatilityFilter, VolatilityConfig

    config = VolatilityConfig(
        lookback_period=20,
        high_vol_threshold=0.03,
        position_scaling=True
    )

    filter = VolatilityFilter(config)
    filter.update(prices)

    if filter.is_trading_allowed():
        position_size = filter.get_adjusted_size(base_size)
    ```
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.math_utils import safe_divide


class VolatilityRegime(Enum):
    """Volatility regime classification."""
    LOW = 'low'
    NORMAL = 'normal'
    ELEVATED = 'elevated'
    HIGH = 'high'
    EXTREME = 'extreme'


class VolatilityAction(Enum):
    """Actions based on volatility level."""
    NORMAL_TRADING = 'normal_trading'
    REDUCE_SIZE = 'reduce_size'
    TIGHTEN_STOPS = 'tighten_stops'
    LIMIT_NEW_TRADES = 'limit_new_trades'
    HALT_TRADING = 'halt_trading'


@dataclass
class VolatilityConfig:
    """
    Volatility Filter Configuration.

    Attributes:
        lookback_period: Period for volatility calculation
        high_vol_threshold: Threshold for high volatility (daily)
        extreme_vol_threshold: Threshold for extreme volatility
        low_vol_threshold: Threshold for low volatility
        scaling_method: Method for position size scaling ('linear', 'inverse', 'sqrt')
        min_scale_factor: Minimum scale factor (don't go below this)
        max_scale_factor: Maximum scale factor
        regime_lookback: Lookback for regime classification
        vol_of_vol_lookback: Lookback for volatility of volatility
        halt_on_extreme: Whether to halt trading on extreme volatility
        ewma_span: Span for EWMA volatility calculation
    """
    lookback_period: int = 20
    high_vol_threshold: float = 0.03
    extreme_vol_threshold: float = 0.05
    low_vol_threshold: float = 0.01
    scaling_method: str = 'inverse'  # 'linear', 'inverse', 'sqrt'
    min_scale_factor: float = 0.25
    max_scale_factor: float = 2.0
    regime_lookback: int = 252
    vol_of_vol_lookback: int = 20
    halt_on_extreme: bool = True
    ewma_span: int = 20

    def validate(self) -> None:
        """Validate configuration."""
        if self.lookback_period < 2:
            raise ValueError("lookback_period must be >= 2")
        if self.high_vol_threshold <= 0:
            raise ValueError("high_vol_threshold must be positive")
        if self.extreme_vol_threshold <= self.high_vol_threshold:
            raise ValueError("extreme_vol_threshold must be > high_vol_threshold")


@dataclass
class VolatilityState:
    """
    Current volatility state.

    Attributes:
        current_vol: Current volatility estimate
        historical_vol: Historical average volatility
        vol_percentile: Current vol percentile vs history
        vol_of_vol: Volatility of volatility
        regime: Current volatility regime
        action: Recommended action
        scale_factor: Position size scaling factor
        is_trading_allowed: Whether trading is allowed
        timestamp: Last update time
    """
    current_vol: float = 0.0
    historical_vol: float = 0.0
    vol_percentile: float = 0.5
    vol_of_vol: float = 0.0
    regime: VolatilityRegime = VolatilityRegime.NORMAL
    action: VolatilityAction = VolatilityAction.NORMAL_TRADING
    scale_factor: float = 1.0
    is_trading_allowed: bool = True
    timestamp: Optional[datetime] = None


class VolatilityFilter:
    """
    Volatility-Based Trading Filter.

    Provides:
        - Real-time volatility monitoring
        - Regime detection
        - Position size scaling
        - Trading halts during extreme volatility

    Attributes:
        config: Volatility configuration
        state: Current volatility state
    """

    def __init__(self, config: Optional[VolatilityConfig] = None):
        """
        Initialize Volatility Filter.

        Args:
            config: Volatility configuration
        """
        self.config = config or VolatilityConfig()
        self.config.validate()
        self.logger = get_logger('volatility_filter')

        # State
        self.state = VolatilityState()
        self._returns_history: List[float] = []
        self._vol_history: List[float] = []
        self._last_price: Optional[float] = None

    def update(
        self,
        price: float,
        timestamp: Optional[datetime] = None
    ) -> VolatilityState:
        """
        Update volatility state with new price.

        Args:
            price: Current price
            timestamp: Current timestamp

        Returns:
            Updated VolatilityState
        """
        timestamp = timestamp or datetime.utcnow()

        # Calculate return
        if self._last_price is not None and self._last_price > 0:
            ret = (price - self._last_price) / self._last_price
            self._returns_history.append(ret)

            # Keep only lookback period
            max_history = max(self.config.lookback_period, self.config.regime_lookback) + 1
            if len(self._returns_history) > max_history:
                self._returns_history = self._returns_history[-max_history:]

        self._last_price = price

        # Calculate volatility if we have enough data
        if len(self._returns_history) >= self.config.lookback_period:
            self._calculate_volatility()
            self._classify_regime()
            self._determine_action()

        self.state.timestamp = timestamp

        return self.state

    def update_with_returns(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]],
        timestamp: Optional[datetime] = None
    ) -> VolatilityState:
        """
        Update volatility with return series.

        Args:
            returns: Return series
            timestamp: Current timestamp

        Returns:
            Updated VolatilityState
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        elif isinstance(returns, list):
            returns = np.array(returns)

        self._returns_history = list(returns)

        if len(self._returns_history) >= self.config.lookback_period:
            self._calculate_volatility()
            self._classify_regime()
            self._determine_action()

        self.state.timestamp = timestamp or datetime.utcnow()

        return self.state

    def get_adjusted_size(self, base_size: float) -> float:
        """
        Get volatility-adjusted position size.

        Args:
            base_size: Base position size

        Returns:
            Adjusted position size
        """
        return base_size * self.state.scale_factor

    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed under current volatility."""
        return self.state.is_trading_allowed

    def get_volatility_regime(self) -> VolatilityRegime:
        """Get current volatility regime."""
        return self.state.regime

    def get_scale_factor(self) -> float:
        """Get current position scale factor."""
        return self.state.scale_factor

    def get_state(self) -> VolatilityState:
        """Get current volatility state."""
        return self.state

    def calculate_historical_vol(self, returns: np.ndarray) -> float:
        """
        Calculate historical volatility.

        Args:
            returns: Array of returns

        Returns:
            Annualized volatility
        """
        if len(returns) < 2:
            return 0.0
        return np.std(returns) * np.sqrt(252)  # Annualized

    def calculate_ewma_vol(self, returns: np.ndarray) -> float:
        """
        Calculate EWMA volatility.

        Args:
            returns: Array of returns

        Returns:
            EWMA volatility estimate
        """
        if len(returns) < 2:
            return 0.0

        span = self.config.ewma_span
        weights = np.exp(np.linspace(-1, 0, len(returns)))
        weights = weights / weights.sum()

        weighted_var = np.sum(weights * returns**2)
        return np.sqrt(weighted_var * 252)  # Annualized

    def calculate_parkinson_vol(
        self,
        high: np.ndarray,
        low: np.ndarray
    ) -> float:
        """
        Calculate Parkinson volatility (using high-low prices).

        Args:
            high: Array of high prices
            low: Array of low prices

        Returns:
            Parkinson volatility estimate
        """
        if len(high) < 2 or len(low) < 2:
            return 0.0

        hl_ratio = np.log(high / low)
        parkinson = np.sqrt(np.mean(hl_ratio**2) / (4 * np.log(2)))

        return parkinson * np.sqrt(252)  # Annualized

    def calculate_garman_klass_vol(
        self,
        open_price: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> float:
        """
        Calculate Garman-Klass volatility.

        Args:
            open_price: Array of open prices
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices

        Returns:
            Garman-Klass volatility estimate
        """
        if len(close) < 2:
            return 0.0

        hl = np.log(high / low)
        co = np.log(close / open_price)

        gk = np.sqrt(
            np.mean(0.5 * hl**2 - (2 * np.log(2) - 1) * co**2)
        )

        return gk * np.sqrt(252)  # Annualized

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get volatility statistics.

        Returns:
            Dictionary with volatility statistics
        """
        return {
            'current_vol': self.state.current_vol,
            'historical_vol': self.state.historical_vol,
            'vol_percentile': self.state.vol_percentile,
            'vol_of_vol': self.state.vol_of_vol,
            'regime': self.state.regime.value,
            'action': self.state.action.value,
            'scale_factor': self.state.scale_factor,
            'is_trading_allowed': self.state.is_trading_allowed
        }

    def reset(self) -> None:
        """Reset volatility filter state."""
        self.state = VolatilityState()
        self._returns_history = []
        self._vol_history = []
        self._last_price = None
        self.logger.info("Volatility filter reset")

    def _calculate_volatility(self) -> None:
        """Calculate current and historical volatility."""
        returns = np.array(self._returns_history)

        # Current volatility (using lookback period)
        if len(returns) >= self.config.lookback_period:
            recent_returns = returns[-self.config.lookback_period:]
            self.state.current_vol = self.calculate_historical_vol(recent_returns)

        # Historical volatility (using longer lookback)
        if len(returns) >= self.config.regime_lookback:
            self.state.historical_vol = self.calculate_historical_vol(
                returns[-self.config.regime_lookback:]
            )

            # Calculate percentile
            rolling_vols = []
            for i in range(self.config.lookback_period, len(returns)):
                window_returns = returns[i-self.config.lookback_period:i]
                rolling_vols.append(np.std(window_returns) * np.sqrt(252))

            if rolling_vols:
                self.state.vol_percentile = np.searchsorted(
                    np.sort(rolling_vols), self.state.current_vol
                ) / len(rolling_vols)

        # Volatility of volatility
        if len(self._vol_history) >= self.config.vol_of_vol_lookback:
            recent_vols = self._vol_history[-self.config.vol_of_vol_lookback:]
            self.state.vol_of_vol = np.std(recent_vols)

        # Store volatility history
        self._vol_history.append(self.state.current_vol)
        if len(self._vol_history) > self.config.regime_lookback:
            self._vol_history = self._vol_history[-self.config.regime_lookback:]

    def _classify_regime(self) -> None:
        """Classify current volatility regime."""
        vol = self.state.current_vol

        if vol < self.config.low_vol_threshold:
            self.state.regime = VolatilityRegime.LOW

        elif vol < self.config.high_vol_threshold:
            self.state.regime = VolatilityRegime.NORMAL

        elif vol < self.config.extreme_vol_threshold:
            self.state.regime = VolatilityRegime.HIGH

        else:
            self.state.regime = VolatilityRegime.EXTREME

        # Log regime changes
        self.logger.debug(f"Volatility regime: {self.state.regime.value} ({vol*100:.2f}%)")

    def _determine_action(self) -> None:
        """Determine action and scale factor based on volatility."""
        regime = self.state.regime
        vol = self.state.current_vol
        cfg = self.config

        if regime == VolatilityRegime.LOW:
            self.state.action = VolatilityAction.NORMAL_TRADING
            self.state.scale_factor = self._calculate_scale_factor()
            self.state.is_trading_allowed = True

        elif regime == VolatilityRegime.NORMAL:
            self.state.action = VolatilityAction.NORMAL_TRADING
            self.state.scale_factor = 1.0
            self.state.is_trading_allowed = True

        elif regime == VolatilityRegime.ELEVATED:
            self.state.action = VolatilityAction.REDUCE_SIZE
            self.state.scale_factor = self._calculate_scale_factor()
            self.state.is_trading_allowed = True
            self.logger.warning(f"Reducing size due to elevated volatility: {vol*100:.2f}%")

        elif regime == VolatilityRegime.HIGH:
            self.state.action = VolatilityAction.LIMIT_NEW_TRADES
            self.state.scale_factor = self._calculate_scale_factor()
            self.state.is_trading_allowed = True
            self.logger.warning(f"Limiting new trades due to high volatility: {vol*100:.2f}%")

        else:  # EXTREME
            self.state.action = VolatilityAction.HALT_TRADING
            self.state.scale_factor = cfg.min_scale_factor
            self.state.is_trading_allowed = not cfg.halt_on_extreme
            self.logger.error(f"Extreme volatility detected: {vol*100:.2f}%")

    def _calculate_scale_factor(self) -> float:
        """Calculate position size scale factor."""
        vol = self.state.current_vol
        cfg = self.config

        if vol == 0:
            return cfg.max_scale_factor

        # Reference volatility (normal market conditions)
        ref_vol = self.config.high_vol_threshold

        if cfg.scaling_method == 'linear':
            # Linear scaling: reduce linearly as vol increases
            factor = ref_vol / vol

        elif cfg.scaling_method == 'inverse':
            # Inverse scaling: more aggressive reduction at high vol
            factor = (ref_vol / vol) ** 1.5

        elif cfg.scaling_method == 'sqrt':
            # Square root scaling: moderate reduction
            factor = np.sqrt(ref_vol / vol)

        else:
            factor = ref_vol / vol

        # Bound the factor
        factor = max(cfg.min_scale_factor, min(cfg.max_scale_factor, factor))

        return factor

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Volatility: {self.state.current_vol*100:.2f}% "
            f"Regime: {self.state.regime.value} "
            f"Scale: {self.state.scale_factor:.2f}x "
            f"Trading: {'Allowed' if self.state.is_trading_allowed else 'Halted'}"
        )


__all__ = [
    'VolatilityRegime',
    'VolatilityAction',
    'VolatilityConfig',
    'VolatilityState',
    'VolatilityFilter'
]
