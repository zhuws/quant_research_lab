"""
Exposure Limits for Quant Research Lab.

Provides comprehensive exposure and position limit management:
    - Position size limits per symbol
    - Sector/exchange exposure limits
    - Correlation-based exposure control
    - Gross/net exposure tracking
    - Dynamic exposure adjustment

Exposure limits are essential for:
    - Diversification enforcement
    - Concentration risk prevention
    - Exchange risk management
    - Portfolio-level risk control

Example usage:
    ```python
    from risk.exposure_limits import ExposureLimits, ExposureConfig

    config = ExposureConfig(
        max_single_position=0.1,
        max_sector_exposure=0.3,
        max_gross_exposure=2.0
    )

    limits = ExposureLimits(config)
    limits.set_equity(100000)
    limits.update_position('ETHUSDT', 0.5, 2000)

    if limits.can_add_position('BTCUSDT', 0.1, 40000):
        # Execute trade
        pass
    ```
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.math_utils import safe_divide


class ExposureType(Enum):
    """Types of exposure limits."""
    NET = 'net'  # Net exposure (longs - shorts)
    GROSS = 'gross'  # Gross exposure (|longs| + |shorts|)
    LONG = 'long'  # Long exposure only
    SHORT = 'short'  # Short exposure only


class LimitScope(Enum):
    """Scope of exposure limits."""
    SYMBOL = 'symbol'
    SECTOR = 'sector'
    EXCHANGE = 'exchange'
    STRATEGY = 'strategy'
    PORTFOLIO = 'portfolio'


@dataclass
class ExposureConfig:
    """
    Exposure Limits Configuration.

    Attributes:
        max_single_position: Maximum single position as fraction of equity
        max_sector_exposure: Maximum sector exposure as fraction of equity
        max_exchange_exposure: Maximum per-exchange exposure
        max_strategy_exposure: Maximum per-strategy exposure
        max_gross_exposure: Maximum gross exposure (leverage)
        max_net_exposure: Maximum net exposure
        max_long_exposure: Maximum long exposure
        max_short_exposure: Maximum short exposure
        max_correlated_assets: Maximum correlated position count
        correlation_threshold: Threshold for considering assets correlated
        rebalance_threshold: Fraction deviation to trigger rebalance
        hard_limit: Whether to enforce hard limits (vs soft warnings)
    """
    max_single_position: float = 0.1
    max_sector_exposure: float = 0.3
    max_exchange_exposure: float = 0.5
    max_strategy_exposure: float = 0.4
    max_gross_exposure: float = 2.0
    max_net_exposure: float = 1.5
    max_long_exposure: float = 1.5
    max_short_exposure: float = 0.5
    max_correlated_assets: int = 3
    correlation_threshold: float = 0.7
    rebalance_threshold: float = 0.05
    hard_limit: bool = True

    def validate(self) -> None:
        """Validate configuration."""
        if not 0 < self.max_single_position <= 1:
            raise ValueError("max_single_position must be between 0 and 1")
        if self.max_gross_exposure < 1:
            raise ValueError("max_gross_exposure must be >= 1")


@dataclass
class PositionExposure:
    """
    Position exposure information.

    Attributes:
        symbol: Trading symbol
        quantity: Position quantity
        market_value: Current market value
        weight: Weight in portfolio
        sector: Asset sector
        exchange: Trading exchange
        strategy: Assigned strategy
        correlation_group: Correlation group identifier
    """
    symbol: str
    quantity: float = 0.0
    market_value: float = 0.0
    weight: float = 0.0
    sector: str = 'default'
    exchange: str = 'default'
    strategy: str = 'default'
    correlation_group: str = 'default'


@dataclass
class ExposureState:
    """
    Current exposure state.

    Attributes:
        gross_exposure: Total gross exposure
        net_exposure: Net exposure (longs - shorts)
        long_exposure: Total long exposure
        short_exposure: Total short exposure
        position_count: Number of positions
        sector_exposures: Exposure by sector
        exchange_exposures: Exposure by exchange
        strategy_exposures: Exposure by strategy
        largest_position: Largest single position weight
        violations: Current limit violations
        timestamp: Last update time
    """
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    position_count: int = 0
    sector_exposures: Dict[str, float] = field(default_factory=dict)
    exchange_exposures: Dict[str, float] = field(default_factory=dict)
    strategy_exposures: Dict[str, float] = field(default_factory=dict)
    largest_position: float = 0.0
    violations: List[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None


class ExposureLimits:
    """
    Exposure and Position Limits Manager.

    Provides:
        - Position size limits
        - Sector/exchange exposure control
        - Correlation-based limits
        - Dynamic exposure adjustment

    Attributes:
        config: Exposure configuration
        state: Current exposure state
    """

    def __init__(self, config: Optional[ExposureConfig] = None):
        """
        Initialize Exposure Limits.

        Args:
            config: Exposure configuration
        """
        self.config = config or ExposureConfig()
        self.config.validate()
        self.logger = get_logger('exposure_limits')

        # State
        self.state = ExposureState()
        self._equity: float = 0.0
        self._positions: Dict[str, PositionExposure] = {}

        # Asset metadata
        self._sector_map: Dict[str, str] = {}
        self._exchange_map: Dict[str, str] = {}
        self._correlation_matrix: Optional[pd.DataFrame] = None

    def set_equity(self, equity: float) -> None:
        """
        Set current equity for weight calculations.

        Args:
            equity: Current portfolio equity
        """
        self._equity = equity
        self._update_state()

    def set_asset_metadata(
        self,
        sector_map: Optional[Dict[str, str]] = None,
        exchange_map: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Set asset metadata for grouping.

        Args:
            sector_map: Dict of {symbol: sector}
            exchange_map: Dict of {symbol: exchange}
        """
        if sector_map:
            self._sector_map.update(sector_map)
        if exchange_map:
            self._exchange_map.update(exchange_map)

    def set_correlation_matrix(self, corr_matrix: pd.DataFrame) -> None:
        """
        Set correlation matrix for correlation-based limits.

        Args:
            corr_matrix: Correlation matrix DataFrame
        """
        self._correlation_matrix = corr_matrix

    def update_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        strategy: str = 'default'
    ) -> None:
        """
        Update position exposure.

        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Current price
            strategy: Strategy identifier
        """
        market_value = abs(quantity * price)

        if symbol in self._positions:
            pos = self._positions[symbol]
            pos.quantity = quantity
            pos.market_value = market_value
            pos.strategy = strategy
        else:
            pos = PositionExposure(
                symbol=symbol,
                quantity=quantity,
                market_value=market_value,
                sector=self._sector_map.get(symbol, 'default'),
                exchange=self._exchange_map.get(symbol, 'default'),
                strategy=strategy
            )
            self._positions[symbol] = pos

        self._update_state()
        self._check_limits()

    def remove_position(self, symbol: str) -> None:
        """
        Remove a position from tracking.

        Args:
            symbol: Symbol to remove
        """
        if symbol in self._positions:
            del self._positions[symbol]
            self._update_state()

    def can_add_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        strategy: str = 'default'
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a new position would violate limits.

        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Entry price
            strategy: Strategy identifier

        Returns:
            Tuple of (is_allowed, reason_if_not)
        """
        if self._equity <= 0:
            return False, "Equity not set"

        market_value = abs(quantity * price)
        weight = market_value / self._equity

        # Check single position limit
        if weight > self.config.max_single_position:
            if self.config.hard_limit:
                return False, f"Position weight {weight*100:.1f}% exceeds limit {self.config.max_single_position*100:.1f}%"
            else:
                self.logger.warning(f"Position weight {weight*100:.1f}% exceeds limit")

        # Calculate hypothetical new exposure
        new_gross = self.state.gross_exposure + weight
        if new_gross > self.config.max_gross_exposure:
            if self.config.hard_limit:
                return False, f"Gross exposure {new_gross:.2f}x would exceed limit {self.config.max_gross_exposure:.2f}x"

        # Check sector exposure
        sector = self._sector_map.get(symbol, 'default')
        sector_exp = self.state.sector_exposures.get(sector, 0) + weight
        if sector_exp > self.config.max_sector_exposure:
            if self.config.hard_limit:
                return False, f"Sector '{sector}' exposure {sector_exp*100:.1f}% would exceed limit {self.config.max_sector_exposure*100:.1f}%"

        # Check exchange exposure
        exchange = self._exchange_map.get(symbol, 'default')
        exchange_exp = self.state.exchange_exposures.get(exchange, 0) + weight
        if exchange_exp > self.config.max_exchange_exposure:
            if self.config.hard_limit:
                return False, f"Exchange '{exchange}' exposure {exchange_exp*100:.1f}% would exceed limit {self.config.max_exchange_exposure*100:.1f}%"

        # Check correlation-based limits
        if self._correlation_matrix is not None:
            correlated_count = self._count_correlated_positions(symbol)
            if correlated_count >= self.config.max_correlated_assets:
                if self.config.hard_limit:
                    return False, f"Would have {correlated_count + 1} correlated positions (max: {self.config.max_correlated_assets})"

        return True, None

    def get_position_multiplier(self, symbol: str) -> float:
        """
        Get position size multiplier for a symbol.

        Accounts for current exposure and limits.

        Args:
            symbol: Trading symbol

        Returns:
            Position size multiplier (0-1)
        """
        if self._equity <= 0:
            return 0.0

        # Check if already at gross exposure limit
        if self.state.gross_exposure >= self.config.max_gross_exposure:
            return 0.0

        # Calculate available exposure headroom
        available = self.config.max_gross_exposure - self.state.gross_exposure

        # Check sector headroom
        sector = self._sector_map.get(symbol, 'default')
        sector_exp = self.state.sector_exposures.get(sector, 0)
        sector_headroom = max(0, self.config.max_sector_exposure - sector_exp)

        # Check exchange headroom
        exchange = self._exchange_map.get(symbol, 'default')
        exchange_exp = self.state.exchange_exposures.get(exchange, 0)
        exchange_headroom = max(0, self.config.max_exchange_exposure - exchange_exp)

        # Return the most restrictive
        multiplier = min(available, sector_headroom, exchange_headroom, self.config.max_single_position)

        return max(0.0, multiplier)

    def get_rebalance_orders(self) -> List[Dict[str, Any]]:
        """
        Get suggested rebalance orders to meet limits.

        Returns:
            List of suggested order dictionaries
        """
        orders = []
        violations = []

        # Check for over-exposed positions
        for symbol, pos in self._positions.items():
            if pos.weight > self.config.max_single_position * (1 + self.config.rebalance_threshold):
                target_weight = self.config.max_single_position
                target_value = target_weight * self._equity
                reduce_amount = pos.market_value - target_value

                orders.append({
                    'symbol': symbol,
                    'action': 'reduce',
                    'current_weight': pos.weight,
                    'target_weight': target_weight,
                    'reduce_value': reduce_amount,
                    'reason': 'single_position_limit'
                })
                violations.append(f"Position {symbol} overweight at {pos.weight*100:.1f}%")

        # Check sector over-exposures
        for sector, exp in self.state.sector_exposures.items():
            if exp > self.config.max_sector_exposure:
                violations.append(f"Sector {sector} over-exposed at {exp*100:.1f}%")
                # Would generate sector rebalance orders

        self.state.violations = violations

        return orders

    def get_state(self) -> ExposureState:
        """Get current exposure state."""
        return self.state

    def get_exposure_summary(self) -> Dict[str, Any]:
        """Get exposure summary."""
        return {
            'gross_exposure': self.state.gross_exposure,
            'net_exposure': self.state.net_exposure,
            'long_exposure': self.state.long_exposure,
            'short_exposure': self.state.short_exposure,
            'position_count': self.state.position_count,
            'largest_position': self.state.largest_position,
            'violations': len(self.state.violations),
            'sectors': dict(self.state.sector_exposures),
            'exchanges': dict(self.state.exchange_exposures)
        }

    def _update_state(self) -> None:
        """Update exposure state from positions."""
        if self._equity <= 0:
            return

        # Calculate exposures
        long_exp = 0.0
        short_exp = 0.0
        sector_exp: Dict[str, float] = {}
        exchange_exp: Dict[str, float] = {}
        strategy_exp: Dict[str, float] = {}
        largest = 0.0

        for pos in self._positions.values():
            weight = pos.market_value / self._equity

            if pos.quantity > 0:
                long_exp += weight
            else:
                short_exp += weight

            # Sector exposure
            sector = pos.sector
            sector_exp[sector] = sector_exp.get(sector, 0) + weight

            # Exchange exposure
            exchange = pos.exchange
            exchange_exp[exchange] = exchange_exp.get(exchange, 0) + weight

            # Strategy exposure
            strategy = pos.strategy
            strategy_exp[strategy] = strategy_exp.get(strategy, 0) + weight

            # Largest position
            largest = max(largest, weight)

            # Update position weight
            pos.weight = weight

        # Update state
        self.state.gross_exposure = long_exp + short_exp
        self.state.net_exposure = long_exp - short_exp
        self.state.long_exposure = long_exp
        self.state.short_exposure = short_exp
        self.state.position_count = len(self._positions)
        self.state.sector_exposures = sector_exp
        self.state.exchange_exposures = exchange_exp
        self.state.strategy_exposures = strategy_exp
        self.state.largest_position = largest
        self.state.timestamp = datetime.utcnow()

    def _check_limits(self) -> None:
        """Check all exposure limits and log violations."""
        violations = []

        # Check gross exposure
        if self.state.gross_exposure > self.config.max_gross_exposure:
            msg = f"Gross exposure {self.state.gross_exposure:.2f}x exceeds limit {self.config.max_gross_exposure:.2f}x"
            violations.append(msg)
            self.logger.warning(msg)

        # Check net exposure
        if abs(self.state.net_exposure) > self.config.max_net_exposure:
            msg = f"Net exposure {self.state.net_exposure:.2f}x exceeds limit {self.config.max_net_exposure:.2f}x"
            violations.append(msg)
            self.logger.warning(msg)

        # Check largest position
        if self.state.largest_position > self.config.max_single_position:
            msg = f"Largest position {self.state.largest_position*100:.1f}% exceeds limit {self.config.max_single_position*100:.1f}%"
            violations.append(msg)
            self.logger.warning(msg)

        self.state.violations = violations

    def _count_correlated_positions(self, symbol: str) -> int:
        """Count positions correlated with given symbol."""
        if self._correlation_matrix is None or symbol not in self._correlation_matrix.index:
            return 0

        count = 0
        correlations = self._correlation_matrix.loc[symbol]

        for pos_symbol in self._positions:
            if pos_symbol != symbol and pos_symbol in correlations:
                if abs(correlations[pos_symbol]) > self.config.correlation_threshold:
                    count += 1

        return count

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Exposure: Gross={self.state.gross_exposure:.2f}x, "
            f"Net={self.state.net_exposure:.2f}x, "
            f"Positions={self.state.position_count}, "
            f"Violations={len(self.state.violations)}"
        )


__all__ = [
    'ExposureType',
    'LimitScope',
    'ExposureConfig',
    'PositionExposure',
    'ExposureState',
    'ExposureLimits'
]
