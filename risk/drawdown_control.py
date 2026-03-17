"""
Drawdown Control for Quant Research Lab.

Provides comprehensive drawdown monitoring and protection:
    - Real-time drawdown tracking
    - Multiple drawdown measures (peak-to-trough, rolling, underwater)
    - Automatic position reduction on drawdown breach
    - Recovery strategies after drawdown events
    - Historical drawdown analysis

Drawdown control is essential for:
    - Capital preservation
    - Risk-adjusted performance optimization
    - Psychological risk management
    - Strategy health monitoring

Example usage:
    ```python
    from risk.drawdown_control import DrawdownControl, DrawdownConfig

    config = DrawdownConfig(
        max_drawdown=0.2,
        warning_threshold=0.1,
        reduce_at=0.15
    )

    control = DrawdownControl(config)
    control.update(equity=100000)

    if control.is_warning():
        # Reduce position sizes
        pass
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


class DrawdownLevel(Enum):
    """Drawdown severity levels."""
    NORMAL = 'normal'
    WARNING = 'warning'
    CRITICAL = 'critical'
    BREACH = 'breach'


class RecoveryAction(Enum):
    """Actions to take during drawdown recovery."""
    NONE = 'none'
    REDUCE_SIZE = 'reduce_size'
    HALT_TRADING = 'halt_trading'
    CLOSE_POSITIONS = 'close_positions'


@dataclass
class DrawdownConfig:
    """
    Drawdown Control Configuration.

    Attributes:
        max_drawdown: Maximum allowed drawdown (fraction)
        warning_threshold: Drawdown level for warnings
        reduce_at: Drawdown level to start reducing positions
        halt_at: Drawdown level to halt new trades
        close_at: Drawdown level to close all positions
        recovery_threshold: Drawdown recovery to resume normal trading
        lookback_period: Period for rolling drawdown calculation
        underwater_threshold: Time underwater before action
        position_reduction_rate: Rate to reduce positions at each step
    """
    max_drawdown: float = 0.2
    warning_threshold: float = 0.1
    reduce_at: float = 0.15
    halt_at: float = 0.18
    close_at: float = 0.2
    recovery_threshold: float = 0.05
    lookback_period: int = 252
    underwater_threshold: int = 30
    position_reduction_rate: float = 0.25

    def validate(self) -> None:
        """Validate configuration."""
        if not 0 < self.max_drawdown < 1:
            raise ValueError("max_drawdown must be between 0 and 1")
        if self.warning_threshold >= self.max_drawdown:
            raise ValueError("warning_threshold must be less than max_drawdown")
        if self.reduce_at >= self.max_drawdown:
            raise ValueError("reduce_at must be less than max_drawdown")


@dataclass
class DrawdownState:
    """
    Current drawdown state.

    Attributes:
        current_drawdown: Current drawdown from peak
        peak_equity: Highest equity achieved
        trough_equity: Lowest equity since peak
        rolling_drawdown: Drawdown over lookback period
        max_drawdown: Maximum historical drawdown
        underwater_days: Days since last new high
        level: Current severity level
        action: Recommended action
        timestamp: Last update time
    """
    current_drawdown: float = 0.0
    peak_equity: float = 0.0
    trough_equity: float = 0.0
    rolling_drawdown: float = 0.0
    max_drawdown: float = 0.0
    underwater_days: int = 0
    level: DrawdownLevel = DrawdownLevel.NORMAL
    action: RecoveryAction = RecoveryAction.NONE
    timestamp: Optional[datetime] = None


@dataclass
class DrawdownEvent:
    """
    Record of a drawdown event.

    Attributes:
        start_date: When drawdown started
        end_date: When drawdown recovered (None if ongoing)
        peak_equity: Equity at peak
        trough_equity: Lowest equity during drawdown
        max_drawdown: Maximum drawdown during event
        duration_days: Duration in days
        recovery_days: Days to recover (None if not recovered)
    """
    start_date: datetime
    end_date: Optional[datetime] = None
    peak_equity: float = 0.0
    trough_equity: float = 0.0
    max_drawdown: float = 0.0
    duration_days: int = 0
    recovery_days: Optional[int] = None


class DrawdownControl:
    """
    Drawdown Monitoring and Control System.

    Provides:
        - Real-time drawdown tracking
        - Multiple drawdown measures
        - Automatic position reduction
        - Recovery strategies
        - Historical analysis

    Attributes:
        config: Drawdown configuration
        state: Current drawdown state
    """

    def __init__(self, config: Optional[DrawdownConfig] = None):
        """
        Initialize Drawdown Control.

        Args:
            config: Drawdown configuration
        """
        self.config = config or DrawdownConfig()
        self.config.validate()
        self.logger = get_logger('drawdown_control')

        # State tracking
        self.state = DrawdownState()
        self._equity_history: List[Tuple[datetime, float]] = []
        self._drawdown_events: List[DrawdownEvent] = []
        self._current_event: Optional[DrawdownEvent] = None
        self._in_drawdown: bool = False
        self._last_peak_date: Optional[datetime] = None

    def update(
        self,
        equity: float,
        timestamp: Optional[datetime] = None
    ) -> DrawdownState:
        """
        Update drawdown state with new equity value.

        Args:
            equity: Current equity
            timestamp: Current timestamp

        Returns:
            Updated DrawdownState
        """
        timestamp = timestamp or datetime.utcnow()

        # Update peak equity
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
            self.state.trough_equity = equity
            self._last_peak_date = timestamp

            # If we were in drawdown, mark recovery
            if self._in_drawdown and self._current_event:
                self._current_event.end_date = timestamp
                self._current_event.recovery_days = (
                    timestamp - self._current_event.start_date
                ).days
                self._drawdown_events.append(self._current_event)
                self._current_event = None
                self._in_drawdown = False
                self.logger.info("Drawdown recovered")

        # Update trough
        if equity < self.state.trough_equity:
            self.state.trough_equity = equity

        # Calculate current drawdown
        if self.state.peak_equity > 0:
            self.state.current_drawdown = 1 - equity / self.state.peak_equity
        else:
            self.state.current_drawdown = 0.0

        # Track equity history
        self._equity_history.append((timestamp, equity))

        # Update rolling drawdown
        self._update_rolling_drawdown()

        # Update max drawdown
        if self.state.current_drawdown > self.state.max_drawdown:
            self.state.max_drawdown = self.state.current_drawdown

        # Track drawdown events
        self._track_drawdown_event(timestamp, equity)

        # Calculate underwater days
        if self._last_peak_date:
            self.state.underwater_days = (timestamp - self._last_peak_date).days
        else:
            self.state.underwater_days = 0

        # Update level and action
        self._update_level_and_action()

        self.state.timestamp = timestamp

        return self.state

    def get_position_multiplier(self) -> float:
        """
        Get position size multiplier based on drawdown level.

        Returns:
            Multiplier (1.0 = full size, 0.0 = no trading)
        """
        dd = self.state.current_drawdown
        cfg = self.config

        if dd < cfg.warning_threshold:
            return 1.0
        elif dd < cfg.reduce_at:
            # Gradual reduction
            reduction_range = cfg.reduce_at - cfg.warning_threshold
            reduction_amount = dd - cfg.warning_threshold
            reduction_pct = reduction_amount / reduction_range
            return 1.0 - reduction_pct * cfg.position_reduction_rate
        elif dd < cfg.halt_at:
            # Significant reduction
            return 1.0 - cfg.position_reduction_rate
        elif dd < cfg.close_at:
            # Minimal trading
            return 1.0 - cfg.position_reduction_rate * 2
        else:
            return 0.0

    def is_warning(self) -> bool:
        """Check if in warning state."""
        return self.state.level in [DrawdownLevel.WARNING, DrawdownLevel.CRITICAL, DrawdownLevel.BREACH]

    def is_critical(self) -> bool:
        """Check if in critical state."""
        return self.state.level in [DrawdownLevel.CRITICAL, DrawdownLevel.BREACH]

    def should_halt_trading(self) -> bool:
        """Check if trading should be halted."""
        return self.state.current_drawdown >= self.config.halt_at

    def should_close_positions(self) -> bool:
        """Check if positions should be closed."""
        return self.state.current_drawdown >= self.config.close_at

    def get_state(self) -> DrawdownState:
        """Get current drawdown state."""
        return self.state

    def get_drawdown_history(self) -> List[DrawdownEvent]:
        """Get historical drawdown events."""
        return self._drawdown_events

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get drawdown statistics.

        Returns:
            Dictionary with drawdown statistics
        """
        if not self._drawdown_events:
            return {
                'total_events': 0,
                'avg_drawdown': 0.0,
                'max_drawdown': self.state.max_drawdown,
                'avg_duration': 0,
                'avg_recovery': 0
            }

        dd_values = [e.max_drawdown for e in self._drawdown_events]
        durations = [e.duration_days for e in self._drawdown_events]
        recoveries = [e.recovery_days for e in self._drawdown_events if e.recovery_days]

        return {
            'total_events': len(self._drawdown_events),
            'avg_drawdown': np.mean(dd_values) if dd_values else 0,
            'max_drawdown': max(dd_values) if dd_values else 0,
            'current_drawdown': self.state.current_drawdown,
            'avg_duration': np.mean(durations) if durations else 0,
            'avg_recovery': np.mean(recoveries) if recoveries else 0,
            'underwater_days': self.state.underwater_days,
            'rolling_dd': self.state.rolling_drawdown
        }

    def calculate_calmar_ratio(self, annual_return: float) -> float:
        """
        Calculate Calmar ratio.

        Args:
            annual_return: Annualized return

        Returns:
            Calmar ratio
        """
        if self.state.max_drawdown == 0:
            return float('inf') if annual_return > 0 else 0.0
        return annual_return / self.state.max_drawdown

    def calculate_pain_index(self) -> float:
        """
        Calculate pain index (average drawdown).

        Returns:
            Pain index
        """
        if not self._equity_history:
            return 0.0

        # Calculate average drawdown over history
        equity_values = [e[1] for e in self._equity_history]
        peak = equity_values[0]
        drawdowns = []

        for equity in equity_values:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            drawdowns.append(dd)

        return np.mean(drawdowns)

    def calculate_pain_ratio(self, annual_return: float) -> float:
        """
        Calculate pain ratio (return / average drawdown).

        Args:
            annual_return: Annualized return

        Returns:
            Pain ratio
        """
        pain_index = self.calculate_pain_index()
        if pain_index == 0:
            return float('inf') if annual_return > 0 else 0.0
        return annual_return / pain_index

    def reset(self, initial_equity: float) -> None:
        """
        Reset drawdown control with new initial equity.

        Args:
            initial_equity: Starting equity
        """
        self.state = DrawdownState()
        self.state.peak_equity = initial_equity
        self.state.trough_equity = initial_equity
        self._equity_history = [(datetime.utcnow(), initial_equity)]
        self._drawdown_events = []
        self._current_event = None
        self._in_drawdown = False
        self._last_peak_date = datetime.utcnow()

        self.logger.info(f"Drawdown control reset with equity ${initial_equity:.2f}")

    def _update_rolling_drawdown(self) -> None:
        """Update rolling drawdown calculation."""
        if len(self._equity_history) < 2:
            self.state.rolling_drawdown = 0.0
            return

        # Get equity values within lookback period
        lookback = min(self.config.lookback_period, len(self._equity_history))
        recent_equity = [e[1] for e in self._equity_history[-lookback:]]

        # Calculate rolling max drawdown
        peak = recent_equity[0]
        max_dd = 0.0

        for eq in recent_equity:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        self.state.rolling_drawdown = max_dd

    def _track_drawdown_event(self, timestamp: datetime, equity: float) -> None:
        """Track drawdown events."""
        if self.state.current_drawdown > 0 and not self._in_drawdown:
            # Start of new drawdown
            self._in_drawdown = True
            self._current_event = DrawdownEvent(
                start_date=timestamp,
                peak_equity=self.state.peak_equity,
                trough_equity=equity,
                max_drawdown=self.state.current_drawdown
            )
            self.logger.info(
                f"Drawdown started: {self.state.current_drawdown*100:.1f}%"
            )

        elif self._in_drawdown and self._current_event:
            # Update existing drawdown
            self._current_event.trough_equity = min(
                self._current_event.trough_equity, equity
            )
            self._current_event.max_drawdown = max(
                self._current_event.max_drawdown,
                self.state.current_drawdown
            )
            self._current_event.duration_days = (
                timestamp - self._current_event.start_date
            ).days

    def _update_level_and_action(self) -> None:
        """Update drawdown level and recommended action."""
        dd = self.state.current_drawdown
        cfg = self.config

        if dd < cfg.warning_threshold:
            self.state.level = DrawdownLevel.NORMAL
            self.state.action = RecoveryAction.NONE

        elif dd < cfg.reduce_at:
            self.state.level = DrawdownLevel.WARNING
            self.state.action = RecoveryAction.NONE
            self.logger.warning(f"Drawdown warning: {dd*100:.1f}%")

        elif dd < cfg.halt_at:
            self.state.level = DrawdownLevel.WARNING
            self.state.action = RecoveryAction.REDUCE_SIZE
            self.logger.warning(
                f"Reducing position size due to drawdown: {dd*100:.1f}%"
            )

        elif dd < cfg.close_at:
            self.state.level = DrawdownLevel.CRITICAL
            self.state.action = RecoveryAction.HALT_TRADING
            self.logger.error(
                f"Halting trading due to critical drawdown: {dd*100:.1f}%"
            )

        else:
            self.state.level = DrawdownLevel.BREACH
            self.state.action = RecoveryAction.CLOSE_POSITIONS
            self.logger.critical(
                f"Drawdown breach! Closing positions: {dd*100:.1f}%"
            )

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Drawdown: {self.state.current_drawdown*100:.1f}% "
            f"(Max: {self.state.max_drawdown*100:.1f}%) "
            f"Level: {self.state.level.value} "
            f"Action: {self.state.action.value}"
        )


__all__ = [
    'DrawdownLevel',
    'RecoveryAction',
    'DrawdownConfig',
    'DrawdownState',
    'DrawdownEvent',
    'DrawdownControl'
]
