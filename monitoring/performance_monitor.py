"""
Performance Monitor for Quant Research Lab.

Provides comprehensive performance tracking:
    - Strategy performance
    - System resource usage
    - Execution latency
    - Throughput metrics
    - Real-time snapshots

Performance monitoring is essential for:
    - Strategy optimization
    - System capacity planning
    - Latency tracking
    - Health monitoring

Example usage:
    ```python
    from monitoring.performance_monitor import PerformanceMonitor

    monitor = PerformanceMonitor()

    # Track trade
    monitor.record_trade('strategy_1', pnl=100, latency_ms=45)

    # Get snapshot
    snapshot = monitor.get_snapshot()
    print(f"Total P&L: {snapshot.total_pnl}")
    ```
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from monitoring.metrics_collector import MetricsCollector


@dataclass
class PerformanceSnapshot:
    """
    Performance snapshot at a point in time.

    Attributes:
        timestamp: Snapshot timestamp
        total_pnl: Total profit/loss
        realized_pnl: Realized P&L
        unrealized_pnl: Unrealized P&L
        total_trades: Total number of trades
        winning_trades: Number of winning trades
        losing_trades: Number of losing trades
        win_rate: Win rate percentage
        avg_trade_pnl: Average trade P&L
        max_drawdown: Maximum drawdown
        sharpe_ratio: Sharpe ratio
        sortino_ratio: Sortino ratio
        avg_latency_ms: Average execution latency
        throughput: Trades per minute
        equity: Current equity
        positions: Number of open positions
    """
    timestamp: datetime = field(default_factory=datetime.utcnow)
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_trade_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    avg_latency_ms: float = 0.0
    throughput: float = 0.0
    equity: float = 0.0
    positions: int = 0
    daily_return: float = 0.0
    weekly_return: float = 0.0
    monthly_return: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_pnl': self.total_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_trade_pnl': self.avg_trade_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'avg_latency_ms': self.avg_latency_ms,
            'throughput': self.throughput,
            'equity': self.equity,
            'positions': self.positions,
            'daily_return': self.daily_return,
            'weekly_return': self.weekly_return,
            'monthly_return': self.monthly_return
        }


@dataclass
class StrategyPerformance:
    """Performance metrics for a single strategy."""
    name: str
    total_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown: float = 0.0
    latencies: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)


class PerformanceMonitor:
    """
    Performance Monitoring System.

    Tracks:
        - Strategy performance
        - System metrics
        - Execution statistics
        - Portfolio health

    Attributes:
        metrics_collector: Optional metrics collector for integration
    """

    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        history_size: int = 10000
    ):
        """
        Initialize Performance Monitor.

        Args:
            metrics_collector: Optional metrics collector
            history_size: Maximum history to keep
        """
        self.metrics_collector = metrics_collector
        self.history_size = history_size
        self.logger = get_logger('performance_monitor')

        # Strategy tracking
        self._strategies: Dict[str, StrategyPerformance] = {}

        # History
        self._equity_curve: List[Dict] = []
        self._trade_history: List[Dict] = []
        self._latency_history: List[float] = []
        self._return_history: List[float] = []

        # Current state
        self._current_equity: float = 0.0
        self._peak_equity: float = 0.0
        self._max_drawdown: float = 0.0
        self._start_time: datetime = datetime.utcnow()

        # Thread safety
        self._lock = threading.RLock()

    def record_trade(
        self,
        strategy: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        pnl: float = 0.0,
        latency_ms: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a trade execution.

        Args:
            strategy: Strategy name
            symbol: Trading symbol
            side: Trade side
            quantity: Trade quantity
            price: Execution price
            pnl: Realized P&L
            latency_ms: Execution latency
            timestamp: Trade timestamp
        """
        timestamp = timestamp or datetime.utcnow()

        with self._lock:
            # Initialize strategy if needed
            if strategy not in self._strategies:
                self._strategies[strategy] = StrategyPerformance(name=strategy)

            strat = self._strategies[strategy]
            strat.total_trades += 1
            strat.total_pnl += pnl

            if pnl > 0:
                strat.winning_trades += 1
            elif pnl < 0:
                strat.losing_trades += 1

            if latency_ms is not None:
                strat.latencies.append(latency_ms)
                self._latency_history.append(latency_ms)

            # Record trade
            trade = {
                'timestamp': timestamp,
                'strategy': strategy,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'pnl': pnl,
                'latency_ms': latency_ms
            }
            strat.trades.append(trade)
            self._trade_history.append(trade)

            # Trim history
            if len(self._trade_history) > self.history_size:
                self._trade_history = self._trade_history[-self.history_size:]

            # Update metrics collector
            if self.metrics_collector:
                self.metrics_collector.counter('trades_total', 1, {'strategy': strategy})
                self.metrics_collector.histogram('trade_pnl', pnl, {'strategy': strategy})
                if latency_ms:
                    self.metrics_collector.histogram('trade_latency_ms', latency_ms)

    def update_equity(
        self,
        equity: float,
        unrealized_pnl: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Update equity curve.

        Args:
            equity: Current equity
            unrealized_pnl: Unrealized P&L
            timestamp: Timestamp
        """
        timestamp = timestamp or datetime.utcnow()

        with self._lock:
            prev_equity = self._current_equity
            self._current_equity = equity

            # Track peak and drawdown
            if equity > self._peak_equity:
                self._peak_equity = equity

            if self._peak_equity > 0:
                drawdown = 1 - equity / self._peak_equity
                self._max_drawdown = max(self._max_drawdown, drawdown)

            # Calculate return
            daily_return = 0.0
            if prev_equity > 0:
                daily_return = (equity - prev_equity) / prev_equity
                self._return_history.append(daily_return)

            # Record equity
            self._equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'unrealized_pnl': unrealized_pnl,
                'return': daily_return
            })

            # Trim history
            if len(self._equity_curve) > self.history_size:
                self._equity_curve = self._equity_curve[-self.history_size:]

            # Update metrics collector
            if self.metrics_collector:
                self.metrics_collector.gauge('equity', equity)
                self.metrics_collector.gauge('unrealized_pnl', unrealized_pnl)
                self.metrics_collector.gauge('max_drawdown', self._max_drawdown)

    def record_signal(
        self,
        strategy: str,
        symbol: str,
        signal_type: str,
        latency_ms: Optional[float] = None
    ) -> None:
        """
        Record a signal generation.

        Args:
            strategy: Strategy name
            symbol: Symbol
            signal_type: Signal type
            latency_ms: Generation latency
        """
        if self.metrics_collector:
            self.metrics_collector.counter('signals_total', 1, {
                'strategy': strategy,
                'symbol': symbol,
                'type': signal_type
            })
            if latency_ms:
                self.metrics_collector.histogram('signal_latency_ms', latency_ms)

    def get_snapshot(self) -> PerformanceSnapshot:
        """
        Get current performance snapshot.

        Returns:
            PerformanceSnapshot with current metrics
        """
        with self._lock:
            # Aggregate strategy metrics
            total_trades = sum(s.total_trades for s in self._strategies.values())
            winning_trades = sum(s.winning_trades for s in self._strategies.values())
            losing_trades = sum(s.losing_trades for s in self._strategies.values())
            total_pnl = sum(s.total_pnl for s in self._strategies.values())

            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

            # Calculate latency
            avg_latency = np.mean(self._latency_history) if self._latency_history else 0

            # Calculate throughput
            if self._trade_history:
                time_span = (datetime.utcnow() - self._start_time).total_seconds() / 60
                throughput = len(self._trade_history) / time_span if time_span > 0 else 0
            else:
                throughput = 0

            # Calculate returns
            returns = np.array(self._return_history) if self._return_history else np.array([0])
            daily_return = returns[-1] if len(returns) > 0 else 0

            # Calculate Sharpe
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                downside_returns = returns[returns < 0]
                downside_std = np.std(downside_returns) if len(downside_returns) > 1 else np.std(returns)
                sortino = np.mean(returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
            else:
                sharpe = 0
                sortino = 0

            # Calculate period returns
            equity_curve = self._equity_curve
            if len(equity_curve) > 1:
                current_equity = equity_curve[-1]['equity']

                # Daily return (last 24h)
                day_ago = datetime.utcnow() - timedelta(days=1)
                day_data = [e for e in equity_curve if e['timestamp'] > day_ago]
                if day_data:
                    daily_return = (current_equity - day_data[0]['equity']) / day_data[0]['equity']
                else:
                    daily_return = 0

                # Weekly return
                week_ago = datetime.utcnow() - timedelta(weeks=1)
                week_data = [e for e in equity_curve if e['timestamp'] > week_ago]
                if week_data:
                    weekly_return = (current_equity - week_data[0]['equity']) / week_data[0]['equity']
                else:
                    weekly_return = 0

                # Monthly return
                month_ago = datetime.utcnow() - timedelta(days=30)
                month_data = [e for e in equity_curve if e['timestamp'] > month_ago]
                if month_data:
                    monthly_return = (current_equity - month_data[0]['equity']) / month_data[0]['equity']
                else:
                    monthly_return = 0
            else:
                daily_return = weekly_return = monthly_return = 0

            return PerformanceSnapshot(
                timestamp=datetime.utcnow(),
                total_pnl=total_pnl,
                realized_pnl=total_pnl,
                unrealized_pnl=self._equity_curve[-1].get('unrealized_pnl', 0) if self._equity_curve else 0,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                avg_trade_pnl=avg_pnl,
                max_drawdown=self._max_drawdown,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                avg_latency_ms=avg_latency,
                throughput=throughput,
                equity=self._current_equity,
                positions=0,  # Would need position tracking
                daily_return=daily_return,
                weekly_return=weekly_return,
                monthly_return=monthly_return
            )

    def get_strategy_performance(self, strategy: str) -> Optional[StrategyPerformance]:
        """Get performance for a specific strategy."""
        return self._strategies.get(strategy)

    def get_all_strategies(self) -> Dict[str, StrategyPerformance]:
        """Get all strategy performances."""
        return dict(self._strategies)

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        with self._lock:
            if not self._equity_curve:
                return pd.DataFrame(columns=['timestamp', 'equity'])

            df = pd.DataFrame(self._equity_curve)
            df.set_index('timestamp', inplace=True)
            return df

    def get_trade_history(
        self,
        strategy: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get trade history.

        Args:
            strategy: Optional strategy filter
            limit: Maximum trades to return

        Returns:
            DataFrame with trade history
        """
        with self._lock:
            trades = self._trade_history.copy()

        if strategy:
            trades = [t for t in trades if t['strategy'] == strategy]

        trades = trades[-limit:]

        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)
        df.set_index('timestamp', inplace=True)
        return df

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        with self._lock:
            latencies = self._latency_history.copy()

        if not latencies:
            return {'avg': 0, 'min': 0, 'max': 0, 'p99': 0}

        return {
            'avg': float(np.mean(latencies)),
            'min': float(np.min(latencies)),
            'max': float(np.max(latencies)),
            'p50': float(np.percentile(latencies, 50)),
            'p90': float(np.percentile(latencies, 90)),
            'p99': float(np.percentile(latencies, 99))
        }

    def reset(self) -> None:
        """Reset all tracking."""
        with self._lock:
            self._strategies.clear()
            self._equity_curve.clear()
            self._trade_history.clear()
            self._latency_history.clear()
            self._return_history.clear()
            self._current_equity = 0.0
            self._peak_equity = 0.0
            self._max_drawdown = 0.0
            self._start_time = datetime.utcnow()

        self.logger.info("Performance monitor reset")


__all__ = [
    'PerformanceSnapshot',
    'StrategyPerformance',
    'PerformanceMonitor'
]
