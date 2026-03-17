"""
Capital Allocator for Quant Research Lab.

Handles position sizing and capital management:
    - Risk-based position sizing
    - Kelly criterion sizing
    - Volatility-adjusted sizing
    - Drawdown-aware allocation
    - Leverage management
    - Multi-asset capital allocation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.math_utils import safe_divide


class SizingMethod(Enum):
    """Position sizing methods."""
    FIXED = 'fixed'
    PERCENT_RISK = 'percent_risk'
    VOLATILITY_ADJUSTED = 'volatility_adjusted'
    KELLY = 'kelly'
    RISK_PARITY = 'risk_parity'
    ATR_BASED = 'atr_based'
    OPTIMAL_F = 'optimal_f'


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def update(self, current_price: float) -> None:
        """Update position with current price."""
        self.current_price = current_price
        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
            self.unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
            self.unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price


@dataclass
class CapitalState:
    """Tracks current capital state."""
    total_capital: float
    available_capital: float
    deployed_capital: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    max_capital: float = 0.0
    drawdown: float = 0.0
    leverage: float = 1.0

    def update_high_watermark(self) -> None:
        """Update max capital and drawdown."""
        current_total = self.total_capital + self.unrealized_pnl
        self.max_capital = max(self.max_capital, current_total)
        if self.max_capital > 0:
            self.drawdown = (self.max_capital - current_total) / self.max_capital


class CapitalAllocator:
    """
    Capital and Position Management System.

    Manages capital allocation across positions with various sizing methods:
        - Fixed fractional sizing
        - Risk parity sizing
        - Kelly criterion
        - Volatility-adjusted sizing
        - ATR-based sizing

    Features:
        - Drawdown protection
        - Leverage management
        - Position limits
        - Stop-loss management

    Attributes:
        total_capital: Total portfolio capital
        risk_per_trade: Maximum risk per trade (fraction)
        max_position_size: Maximum position size as fraction of capital
        max_positions: Maximum number of concurrent positions
        sizing_method: Default position sizing method
    """

    def __init__(
        self,
        total_capital: float = 100000,
        risk_per_trade: float = 0.02,
        max_position_size: float = 0.10,
        max_positions: int = 10,
        max_total_exposure: float = 1.0,
        max_drawdown_limit: float = 0.20,
        sizing_method: str = 'percent_risk',
        leverage_limit: float = 1.0,
        reserve_ratio: float = 0.1
    ):
        """
        Initialize Capital Allocator.

        Args:
            total_capital: Initial capital
            risk_per_trade: Maximum risk per trade (default: 2%)
            max_position_size: Maximum position size as fraction of capital
            max_positions: Maximum number of concurrent positions
            max_total_exposure: Maximum total exposure (fraction)
            max_drawdown_limit: Maximum allowed drawdown before scaling down
            sizing_method: Default position sizing method
            leverage_limit: Maximum leverage allowed
            reserve_ratio: Fraction of capital to keep in reserve
        """
        self.logger = get_logger('capital_allocator')

        try:
            self.sizing_method = SizingMethod(sizing_method)
        except ValueError:
            self.logger.warning(f"Unknown sizing method {sizing_method}, using percent_risk")
            self.sizing_method = SizingMethod.PERCENT_RISK

        # Risk parameters
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        self.max_total_exposure = max_total_exposure
        self.max_drawdown_limit = max_drawdown_limit
        self.leverage_limit = leverage_limit
        self.reserve_ratio = reserve_ratio

        # Capital state
        self._capital = CapitalState(
            total_capital=total_capital,
            available_capital=total_capital * (1 - reserve_ratio),
            max_capital=total_capital
        )

        # Position registry
        self._positions: Dict[str, Position] = {}

        # Trade history
        self._trade_history: List[Dict] = []

        # Performance tracking
        self._daily_pnl: List[Tuple[datetime, float]] = []

    @property
    def total_capital(self) -> float:
        """Get current total capital."""
        return self._capital.total_capital + self._capital.unrealized_pnl

    @property
    def available_capital(self) -> float:
        """Get available capital for new positions."""
        return self._capital.available_capital

    @property
    def current_drawdown(self) -> float:
        """Get current drawdown."""
        self._capital.update_high_watermark()
        return self._capital.drawdown

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: Optional[float] = None,
        volatility: Optional[float] = None,
        atr: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None,
        method: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Calculate optimal position size.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price (optional)
            volatility: Asset volatility (annualized)
            atr: Average True Range
            win_rate: Historical win rate
            avg_win_loss_ratio: Average win/loss ratio
            method: Override default sizing method
            **kwargs: Additional method-specific arguments

        Returns:
            Dictionary with position sizing details
        """
        method = method or self.sizing_method.value

        try:
            sizing_method = SizingMethod(method)
        except ValueError:
            sizing_method = self.sizing_method

        # Check if we can open more positions
        if len(self._positions) >= self.max_positions:
            self.logger.warning(f"Maximum positions ({self.max_positions}) reached")
            return {'size': 0, 'reason': 'max_positions_reached'}

        # Check drawdown limit
        if self.current_drawdown >= self.max_drawdown_limit:
            self.logger.warning(f"Drawdown limit ({self.max_drawdown_limit}) reached")
            return {'size': 0, 'reason': 'drawdown_limit_reached'}

        # Scale capital based on drawdown
        scale_factor = self._get_drawdown_scale_factor()
        effective_capital = self._capital.total_capital * scale_factor

        # Calculate size based on method
        method_map = {
            SizingMethod.FIXED: self._fixed_sizing,
            SizingMethod.PERCENT_RISK: self._percent_risk_sizing,
            SizingMethod.VOLATILITY_ADJUSTED: self._volatility_adjusted_sizing,
            SizingMethod.KELLY: self._kelly_sizing,
            SizingMethod.RISK_PARITY: self._risk_parity_sizing,
            SizingMethod.ATR_BASED: self._atr_sizing,
            SizingMethod.OPTIMAL_F: self._optimal_f_sizing
        }

        result = method_map[sizing_method](
            effective_capital=effective_capital,
            entry_price=entry_price,
            stop_loss=stop_loss,
            volatility=volatility,
            atr=atr,
            win_rate=win_rate,
            avg_win_loss_ratio=avg_win_loss_ratio,
            **kwargs
        )

        # Apply constraints
        result['size'] = self._apply_size_constraints(result['size'], entry_price, effective_capital)
        result['symbol'] = symbol
        result['method'] = sizing_method.value
        result['entry_price'] = entry_price
        result['capital_allocated'] = result['size'] * entry_price
        result['risk_amount'] = self._calculate_risk_amount(result['size'], entry_price, stop_loss)

        return result

    def _fixed_sizing(
        self,
        effective_capital: float,
        entry_price: float,
        **kwargs
    ) -> Dict:
        """Fixed fractional position sizing."""
        size = (effective_capital * self.max_position_size) / entry_price
        return {'size': size}

    def _percent_risk_sizing(
        self,
        effective_capital: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        **kwargs
    ) -> Dict:
        """Risk a fixed percentage of capital per trade."""
        risk_amount = effective_capital * self.risk_per_trade

        if stop_loss and entry_price != stop_loss:
            risk_per_share = abs(entry_price - stop_loss)
            size = risk_amount / risk_per_share
        else:
            # Default: assume 2% stop if not provided
            risk_per_share = entry_price * 0.02
            size = risk_amount / risk_per_share

        return {'size': size, 'risk_amount': risk_amount}

    def _volatility_adjusted_sizing(
        self,
        effective_capital: float,
        entry_price: float,
        volatility: Optional[float] = None,
        target_volatility: float = 0.15,
        **kwargs
    ) -> Dict:
        """Size based on target volatility contribution."""
        if volatility is None or volatility == 0:
            volatility = 0.20  # Default 20% annualized

        # Position size for target volatility
        # Position volatility = position_value * asset_volatility
        target_position_value = (effective_capital * target_volatility) / volatility
        size = target_position_value / entry_price

        return {'size': size, 'implied_volatility': volatility}

    def _kelly_sizing(
        self,
        effective_capital: float,
        entry_price: float,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None,
        kelly_fraction: float = 0.5,
        **kwargs
    ) -> Dict:
        """
        Kelly criterion position sizing.

        Kelly % = W - (1-W)/R
        where W = win rate, R = avg win / avg loss
        """
        if win_rate is None:
            win_rate = 0.5
        if avg_win_loss_ratio is None:
            avg_win_loss_ratio = 1.5

        # Kelly formula
        kelly_pct = win_rate - (1 - win_rate) / avg_win_loss_ratio

        # Apply Kelly fraction (half-Kelly by default for safety)
        kelly_pct = max(0, kelly_pct) * kelly_fraction

        # Cap at max position size
        kelly_pct = min(kelly_pct, self.max_position_size)

        size = (effective_capital * kelly_pct) / entry_price

        return {
            'size': size,
            'kelly_percentage': kelly_pct,
            'raw_kelly': win_rate - (1 - win_rate) / avg_win_loss_ratio
        }

    def _risk_parity_sizing(
        self,
        effective_capital: float,
        entry_price: float,
        volatility: Optional[float] = None,
        **kwargs
    ) -> Dict:
        """Size inversely proportional to volatility."""
        if volatility is None or volatility == 0:
            volatility = 0.20

        # Inverse volatility weight
        inv_vol = 1.0 / volatility

        # Normalize to max position size
        position_value = effective_capital * min(inv_vol / 10, self.max_position_size)
        size = position_value / entry_price

        return {'size': size, 'inv_volatility': inv_vol}

    def _atr_sizing(
        self,
        effective_capital: float,
        entry_price: float,
        atr: Optional[float] = None,
        atr_multiplier: float = 2.0,
        **kwargs
    ) -> Dict:
        """Size based on Average True Range."""
        if atr is None:
            atr = entry_price * 0.02  # Default 2% ATR

        # Risk is ATR * multiplier
        risk_per_share = atr * atr_multiplier
        risk_amount = effective_capital * self.risk_per_trade
        size = risk_amount / risk_per_share

        return {
            'size': size,
            'atr': atr,
            'stop_distance': risk_per_share
        }

    def _optimal_f_sizing(
        self,
        effective_capital: float,
        entry_price: float,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None,
        **kwargs
    ) -> Dict:
        """
        Optimal F position sizing.

        Based on Ralph Vince's Optimal F formula.
        """
        if win_rate is None:
            win_rate = 0.5
        if avg_win_loss_ratio is None:
            avg_win_loss_ratio = 1.5

        # Simplified Optimal F calculation
        # f* = ((W * (R+1) - 1) / R)
        # where W = win rate, R = win/loss ratio
        optimal_f = ((win_rate * (avg_win_loss_ratio + 1) - 1) / avg_win_loss_ratio)

        # Apply safety margin
        optimal_f = max(0, optimal_f) * 0.8
        optimal_f = min(optimal_f, self.max_position_size)

        size = (effective_capital * optimal_f) / entry_price

        return {'size': size, 'optimal_f': optimal_f}

    def _apply_size_constraints(
        self,
        size: float,
        entry_price: float,
        effective_capital: float
    ) -> float:
        """Apply position size constraints."""
        # Maximum size based on max position size
        max_size = (effective_capital * self.max_position_size) / entry_price

        # Maximum size based on available capital
        max_available = self._capital.available_capital / entry_price

        # Apply all constraints
        size = min(size, max_size, max_available)
        size = max(0, size)

        return size

    def _calculate_risk_amount(
        self,
        size: float,
        entry_price: float,
        stop_loss: Optional[float]
    ) -> float:
        """Calculate risk amount for the position."""
        if stop_loss:
            return abs(size * (entry_price - stop_loss))
        return size * entry_price * 0.02  # Assume 2% risk

    def _get_drawdown_scale_factor(self) -> float:
        """Get capital scaling factor based on drawdown."""
        dd = self.current_drawdown
        if dd < self.max_drawdown_limit * 0.5:
            return 1.0
        elif dd < self.max_drawdown_limit * 0.75:
            return 0.75
        elif dd < self.max_drawdown_limit:
            return 0.5
        else:
            return 0.25

    def open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[Position]:
        """
        Open a new position.

        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            size: Position size
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Position object if successful
        """
        # Check constraints
        if len(self._positions) >= self.max_positions:
            self.logger.warning(f"Cannot open position: max positions reached")
            return None

        if symbol in self._positions:
            self.logger.warning(f"Position already exists for {symbol}")
            return None

        # Calculate required capital
        required_capital = size * entry_price
        if required_capital > self._capital.available_capital:
            self.logger.warning(f"Insufficient capital: need {required_capital}, have {self._capital.available_capital}")
            return None

        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        # Update capital state
        self._positions[symbol] = position
        self._capital.deployed_capital += required_capital
        self._capital.available_capital -= required_capital

        self.logger.info(f"Opened {side} position: {symbol} @ {entry_price}, size={size}")

        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = 'manual'
    ) -> Optional[Dict]:
        """
        Close an existing position.

        Args:
            symbol: Trading symbol
            exit_price: Exit price
            reason: Reason for closing

        Returns:
            Trade result dictionary
        """
        if symbol not in self._positions:
            self.logger.warning(f"No position found for {symbol}")
            return None

        position = self._positions[symbol]
        position.update(exit_price)

        # Calculate realized P&L
        realized_pnl = position.unrealized_pnl

        # Update capital state
        capital_returned = position.size * exit_price
        self._capital.deployed_capital -= position.size * position.entry_price
        self._capital.available_capital += capital_returned
        self._capital.total_capital += realized_pnl
        self._capital.realized_pnl += realized_pnl

        # Record trade
        trade_record = {
            'symbol': symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'size': position.size,
            'pnl': realized_pnl,
            'pnl_pct': position.unrealized_pnl_pct,
            'entry_time': position.timestamp,
            'exit_time': datetime.now(),
            'reason': reason
        }

        self._trade_history.append(trade_record)
        self._daily_pnl.append((datetime.now(), realized_pnl))

        # Remove position
        del self._positions[symbol]

        self.logger.info(f"Closed {position.side} position: {symbol} @ {exit_price}, P&L={realized_pnl:.2f}")

        return trade_record

    def update_positions(self, prices: Dict[str, float]) -> None:
        """
        Update all positions with current prices.

        Args:
            prices: Dictionary of symbol -> current price
        """
        total_unrealized = 0

        for symbol, position in self._positions.items():
            if symbol in prices:
                position.update(prices[symbol])
                total_unrealized += position.unrealized_pnl

        self._capital.unrealized_pnl = total_unrealized
        self._capital.update_high_watermark()

    def check_stops(self, prices: Dict[str, float]) -> List[Dict]:
        """
        Check and execute stop losses and take profits.

        Args:
            prices: Current prices

        Returns:
            List of closed positions
        """
        closed = []

        for symbol, position in list(self._positions.items()):
            if symbol not in prices:
                continue

            current_price = prices[symbol]

            # Check stop loss
            if position.stop_loss:
                if position.side == 'long' and current_price <= position.stop_loss:
                    closed.append(self.close_position(symbol, current_price, 'stop_loss'))
                elif position.side == 'short' and current_price >= position.stop_loss:
                    closed.append(self.close_position(symbol, current_price, 'stop_loss'))

            # Check take profit
            if position.take_profit and symbol in self._positions:
                if position.side == 'long' and current_price >= position.take_profit:
                    closed.append(self.close_position(symbol, current_price, 'take_profit'))
                elif position.side == 'short' and current_price <= position.take_profit:
                    closed.append(self.close_position(symbol, current_price, 'take_profit'))

        return closed

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary."""
        self._capital.update_high_watermark()

        return {
            'total_capital': self.total_capital,
            'available_capital': self._capital.available_capital,
            'deployed_capital': self._capital.deployed_capital,
            'unrealized_pnl': self._capital.unrealized_pnl,
            'realized_pnl': self._capital.realized_pnl,
            'current_drawdown': self._capital.drawdown,
            'max_drawdown': self.max_drawdown_limit,
            'n_positions': len(self._positions),
            'max_positions': self.max_positions,
            'exposure_ratio': self._capital.deployed_capital / self._capital.total_capital,
            'positions': {s: {'side': p.side, 'size': p.size, 'pnl': p.unrealized_pnl}
                         for s, p in self._positions.items()}
        }

    def get_trade_statistics(self) -> Dict:
        """Calculate trade statistics."""
        if not self._trade_history:
            return {'total_trades': 0}

        pnls = [t['pnl'] for t in self._trade_history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        return {
            'total_trades': len(self._trade_history),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(pnls) if pnls else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'total_pnl': sum(pnls),
            'profit_factor': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0,
            'max_win': max(pnls) if pnls else 0,
            'max_loss': min(pnls) if pnls else 0,
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'std_pnl': np.std(pnls) if len(pnls) > 1 else 0
        }

    def calculate_position_risk(self, symbol: str) -> Dict:
        """Calculate risk metrics for a position."""
        if symbol not in self._positions:
            return {}

        position = self._positions[symbol]
        entry_value = position.size * position.entry_price
        current_value = position.size * position.current_price

        # Calculate stop loss risk
        stop_loss_risk = 0
        if position.stop_loss:
            stop_loss_risk = abs(position.size * (position.entry_price - position.stop_loss))

        return {
            'symbol': symbol,
            'entry_value': entry_value,
            'current_value': current_value,
            'unrealized_pnl': position.unrealized_pnl,
            'stop_loss_risk': stop_loss_risk,
            'stop_loss_risk_pct': stop_loss_risk / entry_value if entry_value > 0 else 0
        }

    def get_portfolio_risk(self) -> Dict:
        """Calculate total portfolio risk."""
        total_stop_risk = 0
        position_risks = []

        for symbol in self._positions:
            risk = self.calculate_position_risk(symbol)
            position_risks.append(risk)
            total_stop_risk += risk.get('stop_loss_risk', 0)

        total_capital = self._capital.total_capital

        return {
            'total_stop_risk': total_stop_risk,
            'stop_risk_pct': total_stop_risk / total_capital if total_capital > 0 else 0,
            'exposure_pct': self._capital.deployed_capital / total_capital if total_capital > 0 else 0,
            'position_risks': position_risks,
            'max_risk_pct': self.risk_per_trade * self.max_positions
        }

    def resize_positions_for_risk(self, target_risk: float) -> Dict[str, float]:
        """
        Calculate position size adjustments to meet target risk.

        Args:
            target_risk: Target portfolio risk (fraction of capital)

        Returns:
            Dictionary of symbol -> new size
        """
        current_risk = self.get_portfolio_risk()
        risk_ratio = current_risk['stop_risk_pct'] / target_risk if target_risk > 0 else 1

        new_sizes = {}
        for symbol, position in self._positions.items():
            if risk_ratio > 1:
                # Reduce positions
                new_sizes[symbol] = position.size / risk_ratio
            else:
                new_sizes[symbol] = position.size

        return new_sizes

    def reset_capital(self, new_capital: Optional[float] = None) -> None:
        """
        Reset capital state.

        Args:
            new_capital: New capital amount (optional, keeps current if not provided)
        """
        if new_capital is not None:
            self._capital.total_capital = new_capital

        self._capital.available_capital = self._capital.total_capital * (1 - self.reserve_ratio)
        self._capital.deployed_capital = 0
        self._capital.unrealized_pnl = 0
        self._capital.realized_pnl = 0
        self._capital.max_capital = self._capital.total_capital
        self._capital.drawdown = 0

        self._positions.clear()

        self.logger.info(f"Capital reset to {self._capital.total_capital}")


def create_capital_allocator(
    initial_capital: float = 100000,
    risk_per_trade: float = 0.02,
    max_position_size: float = 0.10,
    **kwargs
) -> CapitalAllocator:
    """
    Convenience function to create a capital allocator.

    Args:
        initial_capital: Starting capital
        risk_per_trade: Risk per trade as fraction
        max_position_size: Maximum position size as fraction
        **kwargs: Additional arguments

    Returns:
        Configured CapitalAllocator
    """
    return CapitalAllocator(
        total_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        max_position_size=max_position_size,
        **kwargs
    )
