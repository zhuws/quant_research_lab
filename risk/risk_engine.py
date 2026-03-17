"""
Risk Engine for Quant Research Lab.

Central risk management engine that coordinates:
    - Position sizing and limits
    - Drawdown monitoring and protection
    - Exposure limits across strategies and assets
    - Volatility-based trading filters
    - Real-time risk monitoring

The RiskEngine acts as the central authority for all risk-related decisions
in the trading system, ensuring that trades comply with risk parameters
before execution.

Example usage:
    ```python
    from risk.risk_engine import RiskEngine, RiskConfig

    config = RiskConfig(
        max_position=1.0,
        max_drawdown=0.2,
        daily_loss_limit=0.05
    )

    engine = RiskEngine(config)
    engine.update_position('ETHUSDT', 0.5, 2000)

    # Check if order is allowed
    if engine.check_order('ETHUSDT', {'side': 'buy', 'quantity': 0.1}):
        # Execute order
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


class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'


class RiskAction(Enum):
    """Actions to take when risk limits are hit."""
    WARN = 'warn'  # Log warning only
    REDUCE = 'reduce'  # Reduce position size
    BLOCK = 'block'  # Block the trade
    CLOSE = 'close'  # Close all positions


@dataclass
class RiskConfig:
    """
    Risk Engine Configuration.

    Attributes:
        max_position: Maximum position size per symbol (in base currency)
        max_drawdown: Maximum allowed drawdown before action
        daily_loss_limit: Maximum daily loss as fraction of equity
        max_leverage: Maximum leverage allowed
        max_correlation: Maximum correlation between positions
        max_sector_exposure: Maximum exposure to single sector
        max_single_trade: Maximum single trade size as fraction of equity
        volatility_threshold: Volatility threshold for risk reduction
        risk_free_rate: Risk-free rate for calculations
        commission_rate: Trading commission rate
        action_on_limit: Action when limits are hit
        cooldown_minutes: Cooldown period after limit breach
        alert_threshold: Fraction of limit to trigger alert
    """
    max_position: float = 1.0
    max_drawdown: float = 0.2
    daily_loss_limit: float = 0.05
    max_leverage: float = 3.0
    max_correlation: float = 0.7
    max_sector_exposure: float = 0.3
    max_single_trade: float = 0.05
    volatility_threshold: float = 0.03
    risk_free_rate: float = 0.02
    commission_rate: float = 0.001
    action_on_limit: RiskAction = RiskAction.WARN
    cooldown_minutes: int = 30
    alert_threshold: float = 0.8

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_position <= 0:
            raise ValueError("max_position must be positive")
        if not 0 < self.max_drawdown < 1:
            raise ValueError("max_drawdown must be between 0 and 1")
        if not 0 < self.daily_loss_limit < 1:
            raise ValueError("daily_loss_limit must be between 0 and 1")
        if self.max_leverage < 1:
            raise ValueError("max_leverage must be >= 1")


@dataclass
class PositionInfo:
    """
    Position information for risk tracking.

    Attributes:
        symbol: Trading symbol
        quantity: Position quantity (positive=long, negative=short)
        entry_price: Average entry price
        current_price: Current market price
        unrealized_pnl: Unrealized profit/loss
        market_value: Current market value of position
        timestamp: Last update timestamp
    """
    symbol: str
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    market_value: float = 0.0
    timestamp: Optional[datetime] = None

    def update(self, price: float, timestamp: Optional[datetime] = None) -> None:
        """Update position with current price."""
        self.current_price = price
        self.market_value = abs(self.quantity * price)
        if self.quantity != 0:
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        self.timestamp = timestamp or datetime.utcnow()


@dataclass
class RiskState:
    """
    Current risk state snapshot.

    Attributes:
        equity: Current equity
        peak_equity: Peak equity achieved
        current_drawdown: Current drawdown
        daily_pnl: Today's profit/loss
        daily_pnl_pct: Today's P&L as percentage
        total_exposure: Total market exposure
        leverage: Current leverage
        risk_level: Current risk severity level
        warnings: Active warning messages
        blocked_until: Cooldown end time if blocked
        last_updated: Last state update time
    """
    equity: float = 0.0
    peak_equity: float = 0.0
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    total_exposure: float = 0.0
    leverage: float = 1.0
    risk_level: RiskLevel = RiskLevel.LOW
    warnings: List[str] = field(default_factory=list)
    blocked_until: Optional[datetime] = None
    last_updated: Optional[datetime] = None


class RiskEngine:
    """
    Central Risk Management Engine.

    Coordinates all risk management activities:
        - Position sizing and limits
        - Drawdown monitoring and protection
        - Exposure limits across strategies
        - Volatility filtering
        - Real-time risk state tracking

    The RiskEngine must approve all trades before execution to ensure
    they comply with risk parameters.

    Attributes:
        config: Risk configuration
        state: Current risk state
        positions: Dictionary of position info by symbol
    """

    def __init__(self, config: Optional[RiskConfig] = None, **kwargs):
        """
        Initialize Risk Engine.

        Args:
            config: Risk configuration
            **kwargs: Override config parameters
        """
        if config is None:
            config = RiskConfig(**kwargs)
        self.config = config
        self.config.validate()
        self.logger = get_logger('risk_engine')

        # State tracking
        self.state = RiskState()
        self.positions: Dict[str, PositionInfo] = {}
        self._initial_equity: float = 0.0
        self._start_of_day_equity: float = 0.0
        self._trade_history: List[Dict] = []
        self._last_cooldown: Optional[datetime] = None

        # Sub-components (will be initialized if needed)
        self._drawdown_control = None
        self._exposure_limits = None
        self._volatility_filter = None

    def initialize(
        self,
        initial_equity: float,
        positions: Optional[Dict[str, Dict]] = None
    ) -> None:
        """
        Initialize risk engine with starting equity and positions.

        Args:
            initial_equity: Starting equity
            positions: Optional dict of {symbol: {'quantity': float, 'price': float}}
        """
        self._initial_equity = initial_equity
        self.state.equity = initial_equity
        self.state.peak_equity = initial_equity
        self._start_of_day_equity = initial_equity

        # Initialize positions
        if positions:
            for symbol, pos_data in positions.items():
                self.positions[symbol] = PositionInfo(
                    symbol=symbol,
                    quantity=pos_data.get('quantity', 0),
                    entry_price=pos_data.get('price', 0),
                    current_price=pos_data.get('price', 0)
                )

        self.logger.info(f"Risk engine initialized with equity: ${initial_equity:.2f}")

    def check_order(
        self,
        symbol: str,
        order: Dict[str, Any],
        current_price: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if an order is allowed under current risk parameters.

        Args:
            symbol: Trading symbol
            order: Order dictionary with 'side', 'quantity', 'price' etc.
            current_price: Current market price (optional)

        Returns:
            Tuple of (is_allowed, reason_if_not_allowed)
        """
        # Check if in cooldown
        if self.state.blocked_until and datetime.utcnow() < self.state.blocked_until:
            remaining = (self.state.blocked_until - datetime.utcnow()).total_seconds() / 60
            return False, f"In risk cooldown for {remaining:.1f} more minutes"

        # Check drawdown limit
        if self.state.current_drawdown >= self.config.max_drawdown:
            return self._handle_limit_breach(
                'drawdown',
                f"Drawdown {self.state.current_drawdown*100:.1f}% exceeds limit {self.config.max_drawdown*100:.1f}%"
            )

        # Check daily loss limit
        if self.state.daily_pnl_pct <= -self.config.daily_loss_limit:
            return self._handle_limit_breach(
                'daily_loss',
                f"Daily loss {self.state.daily_pnl_pct*100:.1f}% exceeds limit {self.config.daily_loss_limit*100:.1f}%"
            )

        # Calculate trade value
        quantity = order.get('quantity', 0)
        price = order.get('price', current_price or 0)
        trade_value = abs(quantity * price)

        # Check single trade limit
        if self.state.equity > 0:
            trade_pct = trade_value / self.state.equity
            if trade_pct > self.config.max_single_trade:
                return False, f"Trade size {trade_pct*100:.1f}% exceeds limit {self.config.max_single_trade*100:.1f}%"

        # Check position limit
        current_position = self.positions.get(symbol, PositionInfo(symbol=symbol))
        new_quantity = current_position.quantity + quantity if order.get('side') == 'buy' else current_position.quantity - quantity
        new_position_value = abs(new_quantity * price)

        if new_position_value > self.config.max_position:
            return False, f"Position size ${new_position_value:.2f} exceeds limit ${self.config.max_position:.2f}"

        # Check leverage
        total_exposure = self._calculate_total_exposure() + trade_value
        leverage = total_exposure / self.state.equity if self.state.equity > 0 else 0

        if leverage > self.config.max_leverage:
            return False, f"Leverage {leverage:.1f}x exceeds limit {self.config.max_leverage:.1f}x"

        # All checks passed
        self.logger.debug(f"Order approved for {symbol}: {order}")
        return True, None

    def update_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Update position after trade execution.

        Args:
            symbol: Trading symbol
            quantity: Position change (positive=buy, negative=sell)
            price: Execution price
            timestamp: Trade timestamp
        """
        timestamp = timestamp or datetime.utcnow()

        if symbol not in self.positions:
            self.positions[symbol] = PositionInfo(symbol=symbol)

        pos = self.positions[symbol]

        # Update position
        if pos.quantity == 0:
            # New position
            pos.quantity = quantity
            pos.entry_price = price
        else:
            # Update existing position
            if (pos.quantity > 0 and quantity > 0) or (pos.quantity < 0 and quantity < 0):
                # Adding to position
                total_cost = pos.entry_price * abs(pos.quantity) + price * abs(quantity)
                total_qty = abs(pos.quantity) + abs(quantity)
                pos.entry_price = total_cost / total_qty if total_qty > 0 else price
                pos.quantity += quantity
            else:
                # Reducing or reversing position
                pos.quantity += quantity
                if abs(pos.quantity) < 1e-8:
                    pos.quantity = 0
                    pos.entry_price = 0

        pos.update(price, timestamp)

        # Record trade
        self._trade_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'position_after': pos.quantity
        })

        self.logger.debug(f"Updated position {symbol}: qty={pos.quantity:.4f}, price={price:.2f}")

    def update_market_prices(self, prices: Dict[str, float]) -> None:
        """
        Update market prices for all positions.

        Args:
            prices: Dictionary of {symbol: price}
        """
        total_pnl = 0.0
        total_exposure = 0.0

        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.update(prices[symbol])
                total_pnl += pos.unrealized_pnl
                total_exposure += pos.market_value

        # Update state
        if self._initial_equity > 0:
            self.state.equity = self._initial_equity + total_pnl + self._calculate_realized_pnl()
            self.state.total_exposure = total_exposure

            # Update peak equity and drawdown
            if self.state.equity > self.state.peak_equity:
                self.state.peak_equity = self.state.equity

            self.state.current_drawdown = 1 - self.state.equity / self.state.peak_equity

            # Calculate leverage
            self.state.leverage = total_exposure / self.state.equity if self.state.equity > 0 else 1.0

        # Update risk level
        self._update_risk_level()

        self.state.last_updated = datetime.utcnow()

    def update_daily_pnl(self, daily_pnl: float) -> None:
        """
        Update daily P&L tracking.

        Args:
            daily_pnl: Today's profit/loss
        """
        self.state.daily_pnl = daily_pnl
        if self._start_of_day_equity > 0:
            self.state.daily_pnl_pct = daily_pnl / self._start_of_day_equity

    def reset_daily(self) -> None:
        """Reset daily counters (call at start of trading day)."""
        self._start_of_day_equity = self.state.equity
        self.state.daily_pnl = 0.0
        self.state.daily_pnl_pct = 0.0
        self.logger.info(f"Daily reset: start equity ${self._start_of_day_equity:.2f}")

    def get_state(self) -> RiskState:
        """Get current risk state."""
        return self.state

    def get_positions(self) -> Dict[str, PositionInfo]:
        """Get all current positions."""
        return self.positions

    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get position for a specific symbol."""
        return self.positions.get(symbol)

    def calculate_var(
        self,
        confidence: float = 0.95,
        time_horizon: int = 1
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            confidence: Confidence level (0-1)
            time_horizon: Time horizon in days

        Returns:
            VaR estimate as fraction of equity
        """
        if not self._trade_history or self.state.equity == 0:
            return 0.0

        # Simple historical VaR based on daily P&L
        # In production, this would use more sophisticated methods
        returns = []
        for i, trade in enumerate(self._trade_history[1:], 1):
            if 'pnl' in trade:
                returns.append(trade['pnl'] / self.state.equity)

        if not returns:
            return 0.0

        returns = np.array(returns)
        var = np.percentile(returns, (1 - confidence) * 100)

        # Scale for time horizon
        var = var * np.sqrt(time_horizon)

        return abs(var)

    def force_block(self, minutes: int = 30, reason: str = "Manual block") -> None:
        """
        Force a trading block for specified duration.

        Args:
            minutes: Block duration in minutes
            reason: Reason for block
        """
        self.state.blocked_until = datetime.utcnow() + timedelta(minutes=minutes)
        self.state.warnings.append(f"BLOCKED: {reason}")
        self.logger.warning(f"Trading blocked for {minutes} minutes: {reason}")

    def clear_block(self) -> None:
        """Clear any active trading block."""
        self.state.blocked_until = None
        self.logger.info("Trading block cleared")

    def _calculate_total_exposure(self) -> float:
        """Calculate total market exposure."""
        return sum(pos.market_value for pos in self.positions.values())

    def _calculate_realized_pnl(self) -> float:
        """Calculate total realized P&L from trade history."""
        # Simplified calculation
        realized = 0.0
        for trade in self._trade_history:
            if 'realized_pnl' in trade:
                realized += trade['realized_pnl']
        return realized

    def _handle_limit_breach(
        self,
        limit_type: str,
        message: str
    ) -> Tuple[bool, Optional[str]]:
        """Handle a risk limit breach."""
        self.state.warnings.append(message)
        self.logger.warning(f"Risk limit breach [{limit_type}]: {message}")

        if self.config.action_on_limit == RiskAction.WARN:
            return True, None  # Allow trade but warn
        elif self.config.action_on_limit == RiskAction.BLOCK:
            self.force_block(self.config.cooldown_minutes, message)
            return False, message
        elif self.config.action_on_limit == RiskAction.REDUCE:
            # Would reduce position size in real implementation
            return False, f"Position size reduced: {message}"
        else:
            return False, message

    def _update_risk_level(self) -> None:
        """Update current risk level based on state."""
        risk_score = 0.0

        # Drawdown component
        dd_ratio = self.state.current_drawdown / self.config.max_drawdown
        risk_score += dd_ratio * 0.4

        # Leverage component
        lev_ratio = self.state.leverage / self.config.max_leverage
        risk_score += lev_ratio * 0.3

        # Daily loss component
        if self.state.daily_pnl_pct < 0:
            loss_ratio = abs(self.state.daily_pnl_pct) / self.config.daily_loss_limit
            risk_score += loss_ratio * 0.3

        # Determine risk level
        if risk_score < 0.3:
            self.state.risk_level = RiskLevel.LOW
        elif risk_score < 0.6:
            self.state.risk_level = RiskLevel.MEDIUM
        elif risk_score < 0.8:
            self.state.risk_level = RiskLevel.HIGH
        else:
            self.state.risk_level = RiskLevel.CRITICAL

    def summary(self) -> Dict[str, Any]:
        """Generate risk summary."""
        return {
            'equity': self.state.equity,
            'peak_equity': self.state.peak_equity,
            'drawdown': self.state.current_drawdown,
            'daily_pnl': self.state.daily_pnl,
            'daily_pnl_pct': self.state.daily_pnl_pct,
            'total_exposure': self.state.total_exposure,
            'leverage': self.state.leverage,
            'risk_level': self.state.risk_level.value,
            'active_warnings': len(self.state.warnings),
            'positions': len(self.positions),
            'is_blocked': self.state.blocked_until is not None
        }


def create_risk_engine(
    max_position: float = 1.0,
    max_drawdown: float = 0.2,
    **kwargs
) -> RiskEngine:
    """
    Convenience function to create a risk engine.

    Args:
        max_position: Maximum position size
        max_drawdown: Maximum drawdown
        **kwargs: Additional config parameters

    Returns:
        Configured RiskEngine
    """
    config = RiskConfig(
        max_position=max_position,
        max_drawdown=max_drawdown,
        **kwargs
    )
    return RiskEngine(config)


__all__ = [
    'RiskLevel',
    'RiskAction',
    'RiskConfig',
    'PositionInfo',
    'RiskState',
    'RiskEngine',
    'create_risk_engine'
]
