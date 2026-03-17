"""
Base Strategy Module for Quant Research Lab.

Provides abstract base class for all trading strategies with:
    - Signal generation interface
    - Position management
    - Performance tracking
    - Risk management integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.math_utils import safe_divide


class SignalType(Enum):
    """Trading signal types."""
    BUY = 'buy'
    SELL = 'sell'
    CLOSE_LONG = 'close_long'
    CLOSE_SHORT = 'close_short'
    HOLD = 'hold'
    NO_SIGNAL = 'no_signal'


class StrategyState(Enum):
    """Strategy execution states."""
    IDLE = 'idle'
    RUNNING = 'running'
    PAUSED = 'paused'
    STOPPED = 'stopped'
    ERROR = 'error'


@dataclass
class Signal:
    """Trading signal data structure."""
    symbol: str
    signal_type: SignalType
    price: Optional[float] = None
    size: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    strategy_id: str = ''
    exchange: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    confidence: float = 1.0

    @property
    def is_entry(self) -> bool:
        """Check if this is an entry signal."""
        return self.signal_type in [SignalType.BUY, SignalType.SELL]

    @property
    def is_exit(self) -> bool:
        """Check if this is an exit signal."""
        return self.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'price': self.price,
            'size': self.size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'timestamp': self.timestamp.isoformat(),
            'strategy_id': self.strategy_id,
            'exchange': self.exchange,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


@dataclass
class Position:
    """Strategy position."""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def update_pnl(self, current_price: float) -> float:
        """Update unrealized P&L with current price."""
        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
        return self.unrealized_pnl


@dataclass
class StrategyMetrics:
    """Strategy performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0


class BaseStrategy(ABC):
    """
    Abstract Base Strategy Class.

    All trading strategies must inherit from this class and implement
    the generate_signals method.

    Features:
        - Signal generation interface
        - Position management
        - Performance tracking
        - Risk management hooks
        - Event-driven architecture support

    Attributes:
        strategy_id: Unique strategy identifier
        name: Human-readable strategy name
        state: Current strategy state
        positions: Dictionary of active positions
        metrics: Performance metrics
    """

    def __init__(
        self,
        name: str = 'BaseStrategy',
        capital: float = 100000,
        max_positions: int = 5,
        risk_per_trade: float = 0.02,
        transaction_cost: float = 0.001
    ):
        """
        Initialize Base Strategy.

        Args:
            name: Strategy name
            capital: Initial capital
            max_positions: Maximum concurrent positions
            risk_per_trade: Maximum risk per trade (fraction)
            transaction_cost: Transaction cost per trade (fraction)
        """
        self.strategy_id = str(uuid.uuid4())[:8]
        self.name = name
        self.logger = get_logger(f'strategy_{self.strategy_id}')

        # Capital and risk
        self.initial_capital = capital
        self.capital = capital
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.transaction_cost = transaction_cost

        # State management
        self.state = StrategyState.IDLE
        self.positions: Dict[str, Position] = {}
        self.pending_orders: List[Signal] = []

        # Performance tracking
        self.metrics = StrategyMetrics()
        self.trade_history: List[Dict] = []
        self.pnl_history: List[Tuple[datetime, float]] = []
        self._daily_pnl: List[float] = []

        # Configuration
        self.config: Dict[str, Any] = {}

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        **kwargs
    ) -> List[Signal]:
        """
        Generate trading signals from market data.

        Must be implemented by subclasses.

        Args:
            data: Market data DataFrame
            **kwargs: Additional parameters

        Returns:
            List of Signal objects
        """
        pass

    @abstractmethod
    def should_close_position(
        self,
        position: Position,
        current_data: pd.Series
    ) -> Optional[Signal]:
        """
        Determine if position should be closed.

        Must be implemented by subclasses.

        Args:
            position: Position to evaluate
            current_data: Current market data

        Returns:
            Close signal if should close, None otherwise
        """
        pass

    def initialize(self, **kwargs) -> None:
        """
        Initialize strategy with configuration.

        Override in subclasses for custom initialization.

        Args:
            **kwargs: Configuration parameters
        """
        self.config.update(kwargs)
        self.state = StrategyState.IDLE
        self.logger.info(f"Strategy {self.name} initialized with config: {self.config}")

    def start(self) -> None:
        """Start strategy execution."""
        if self.state == StrategyState.RUNNING:
            self.logger.warning(f"Strategy {self.name} already running")
            return

        self.state = StrategyState.RUNNING
        self.logger.info(f"Strategy {self.name} started")

    def stop(self) -> None:
        """Stop strategy execution."""
        self.state = StrategyState.STOPPED
        self.logger.info(f"Strategy {self.name} stopped")

    def pause(self) -> None:
        """Pause strategy execution."""
        if self.state == StrategyState.RUNNING:
            self.state = StrategyState.PAUSED
            self.logger.info(f"Strategy {self.name} paused")

    def resume(self) -> None:
        """Resume strategy execution."""
        if self.state == StrategyState.PAUSED:
            self.state = StrategyState.RUNNING
            self.logger.info(f"Strategy {self.name} resumed")

    def open_position(
        self,
        signal: Signal
    ) -> Optional[Position]:
        """
        Open a new position from signal.

        Args:
            signal: Entry signal

        Returns:
            Position if opened successfully
        """
        if self.state != StrategyState.RUNNING:
            self.logger.warning(f"Cannot open position: strategy not running")
            return None

        if len(self.positions) >= self.max_positions:
            self.logger.warning(f"Cannot open position: max positions reached")
            return None

        if signal.symbol in self.positions:
            self.logger.warning(f"Position already exists for {signal.symbol}")
            return None

        # Determine position side
        side = 'long' if signal.signal_type == SignalType.BUY else 'short'

        # Calculate position size if not provided
        size = signal.size
        if size is None:
            size = self._calculate_position_size(signal)

        if size <= 0:
            self.logger.warning(f"Invalid position size for {signal.symbol}")
            return None

        # Create position
        position = Position(
            symbol=signal.symbol,
            side=side,
            size=size,
            entry_price=signal.price or 0,
            entry_time=signal.timestamp,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            metadata=signal.metadata
        )

        self.positions[signal.symbol] = position
        self.logger.info(
            f"Opened {side} position: {signal.symbol} @ {signal.price}, size={size}"
        )

        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = 'signal'
    ) -> Optional[Dict]:
        """
        Close an existing position.

        Args:
            symbol: Position symbol
            exit_price: Exit price
            reason: Reason for closing

        Returns:
            Trade record if closed successfully
        """
        if symbol not in self.positions:
            self.logger.warning(f"No position found for {symbol}")
            return None

        position = self.positions[symbol]

        # Calculate P&L
        position.update_pnl(exit_price)

        # Apply transaction costs
        transaction_cost = position.size * exit_price * self.transaction_cost
        realized_pnl = position.unrealized_pnl - transaction_cost

        # Update capital
        self.capital += realized_pnl

        # Record trade
        trade_record = {
            'symbol': symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'size': position.size,
            'pnl': realized_pnl,
            'pnl_pct': realized_pnl / (position.entry_price * position.size),
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'duration': (datetime.now() - position.entry_time).total_seconds() / 3600,
            'reason': reason,
            'strategy_id': self.strategy_id
        }

        self.trade_history.append(trade_record)
        self.pnl_history.append((datetime.now(), realized_pnl))
        self._daily_pnl.append(realized_pnl)

        # Remove position
        del self.positions[symbol]

        self.logger.info(
            f"Closed {position.side} position: {symbol} @ {exit_price}, P&L={realized_pnl:.2f}"
        )

        return trade_record

    def update_positions(
        self,
        prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Update all positions with current prices.

        Args:
            prices: Current prices by symbol

        Returns:
            Dictionary of unrealized P&L by symbol
        """
        unrealized_pnl = {}

        for symbol, position in self.positions.items():
            if symbol in prices:
                pnl = position.update_pnl(prices[symbol])
                unrealized_pnl[symbol] = pnl

        return unrealized_pnl

    def check_position_stops(
        self,
        prices: Dict[str, float]
    ) -> List[Signal]:
        """
        Check stop loss and take profit for all positions.

        Args:
            prices: Current prices by symbol

        Returns:
            List of close signals triggered
        """
        close_signals = []

        for symbol, position in self.positions.items():
            if symbol not in prices:
                continue

            price = prices[symbol]

            # Check stop loss
            if position.stop_loss:
                if position.side == 'long' and price <= position.stop_loss:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.CLOSE_LONG,
                        price=price,
                        timestamp=datetime.now(),
                        strategy_id=self.strategy_id,
                        metadata={'reason': 'stop_loss'}
                    )
                    close_signals.append(signal)
                    continue
                elif position.side == 'short' and price >= position.stop_loss:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.CLOSE_SHORT,
                        price=price,
                        timestamp=datetime.now(),
                        strategy_id=self.strategy_id,
                        metadata={'reason': 'stop_loss'}
                    )
                    close_signals.append(signal)
                    continue

            # Check take profit
            if position.take_profit:
                if position.side == 'long' and price >= position.take_profit:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.CLOSE_LONG,
                        price=price,
                        timestamp=datetime.now(),
                        strategy_id=self.strategy_id,
                        metadata={'reason': 'take_profit'}
                    )
                    close_signals.append(signal)
                elif position.side == 'short' and price <= position.take_profit:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.CLOSE_SHORT,
                        price=price,
                        timestamp=datetime.now(),
                        strategy_id=self.strategy_id,
                        metadata={'reason': 'take_profit'}
                    )
                    close_signals.append(signal)

        return close_signals

    def _calculate_position_size(self, signal: Signal) -> float:
        """
        Calculate position size based on risk management.

        Args:
            signal: Entry signal

        Returns:
            Position size
        """
        if signal.price is None or signal.price == 0:
            return 0

        # Risk-based sizing
        risk_amount = self.capital * self.risk_per_trade

        if signal.stop_loss:
            risk_per_unit = abs(signal.price - signal.stop_loss)
        else:
            # Default: assume 2% stop
            risk_per_unit = signal.price * 0.02

        if risk_per_unit > 0:
            size = risk_amount / risk_per_unit
        else:
            size = (self.capital * 0.1) / signal.price  # 10% of capital

        return size

    def calculate_metrics(self) -> StrategyMetrics:
        """
        Calculate strategy performance metrics.

        Returns:
            Updated StrategyMetrics
        """
        if not self.trade_history:
            return self.metrics

        pnls = [t['pnl'] for t in self.trade_history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        self.metrics.total_trades = len(pnls)
        self.metrics.winning_trades = len(wins)
        self.metrics.losing_trades = len(losses)
        self.metrics.total_pnl = sum(pnls)

        # Win rate
        self.metrics.win_rate = safe_divide(
            self.metrics.winning_trades,
            self.metrics.total_trades,
            default=0
        )

        # Average trade
        self.metrics.avg_trade = np.mean(pnls) if pnls else 0
        self.metrics.avg_win = np.mean(wins) if wins else 0
        self.metrics.avg_loss = np.mean(losses) if losses else 0

        # Profit factor
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        self.metrics.profit_factor = safe_divide(total_wins, total_losses, default=0)

        # Total return
        self.metrics.total_return = safe_divide(
            self.capital - self.initial_capital,
            self.initial_capital,
            default=0
        )

        # Sharpe ratio (using daily P&L)
        if len(self._daily_pnl) > 1:
            daily_returns = np.array(self._daily_pnl) / self.initial_capital
            if np.std(daily_returns) > 0:
                self.metrics.sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

        # Max drawdown
        if pnls:
            cumulative = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (running_max - cumulative) / (running_max + self.initial_capital)
            self.metrics.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Consecutive wins/losses
        self.metrics.max_consecutive_wins = self._count_max_consecutive(pnls, positive=True)
        self.metrics.max_consecutive_losses = self._count_max_consecutive(pnls, positive=False)

        return self.metrics

    def _count_max_consecutive(self, values: List[float], positive: bool = True) -> int:
        """Count maximum consecutive positive/negative values."""
        max_count = 0
        current_count = 0

        for v in values:
            if (positive and v > 0) or (not positive and v < 0):
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def get_summary(self) -> Dict:
        """
        Get strategy summary.

        Returns:
            Dictionary with strategy state and metrics
        """
        return {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'state': self.state.value,
            'capital': self.capital,
            'initial_capital': self.initial_capital,
            'n_positions': len(self.positions),
            'positions': {s: {'side': p.side, 'size': p.size, 'pnl': p.unrealized_pnl}
                         for s, p in self.positions.items()},
            'metrics': {
                'total_trades': self.metrics.total_trades,
                'win_rate': self.metrics.win_rate,
                'total_pnl': self.metrics.total_pnl,
                'total_return': self.metrics.total_return,
                'sharpe_ratio': self.metrics.sharpe_ratio,
                'max_drawdown': self.metrics.max_drawdown,
                'profit_factor': self.metrics.profit_factor
            }
        }

    def reset(self) -> None:
        """Reset strategy to initial state."""
        self.capital = self.initial_capital
        self.positions.clear()
        self.pending_orders.clear()
        self.trade_history.clear()
        self.pnl_history.clear()
        self._daily_pnl.clear()
        self.metrics = StrategyMetrics()
        self.state = StrategyState.IDLE

        self.logger.info(f"Strategy {self.name} reset")

    def validate_signal(self, signal: Signal) -> bool:
        """
        Validate a trading signal.

        Override in subclasses for custom validation.

        Args:
            signal: Signal to validate

        Returns:
            True if signal is valid
        """
        if signal.signal_type == SignalType.NO_SIGNAL:
            return False

        if signal.price is not None and signal.price < 0:
            return False

        if signal.size is not None and signal.size < 0:
            return False

        return True


class StrategyRegistry:
    """
    Registry for strategy classes.

    Allows dynamic registration and retrieval of strategy implementations.
    """

    _strategies: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a strategy class."""
        def decorator(strategy_class: type) -> type:
            cls._strategies[name] = strategy_class
            return strategy_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """Get a strategy class by name."""
        return cls._strategies.get(name)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all registered strategy names."""
        return list(cls._strategies.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[BaseStrategy]:
        """Create a strategy instance by name."""
        strategy_class = cls.get(name)
        if strategy_class:
            return strategy_class(**kwargs)
        return None
