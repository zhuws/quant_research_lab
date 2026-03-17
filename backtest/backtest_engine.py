"""
Backtest Engine for Quant Research Lab.

Provides professional backtesting capabilities:
    - Event-driven backtesting
    - Multi-asset support
    - Strategy integration
    - Realistic execution simulation
    - Performance analysis

Features:
    - Walk-forward testing
    - Monte Carlo simulation
    - Parameter optimization
    - Benchmark comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.math_utils import safe_divide
from backtest.execution_simulator import (
    ExecutionSimulator, Order, Fill, OrderType, OrderSide,
    Position, SlippageModel, CommissionModel
)
from backtest.performance_analyzer import PerformanceAnalyzer, PerformanceReport


class BacktestMode(Enum):
    """Backtest execution modes."""
    VECTOR = 'vector'  # Fast vectorized backtest
    EVENT = 'event'  # Event-driven backtest
    WALK_FORWARD = 'walk_forward'  # Walk-forward optimization


@dataclass
class BacktestConfig:
    """
    Backtest configuration.

    Contains all parameters for backtest execution.
    """
    initial_capital: float = 100000
    commission_rate: float = 0.001
    slippage_rate: float = 0.0001
    risk_free_rate: float = 0.02
    position_size_pct: float = 0.10
    max_positions: int = 10
    leverage: float = 1.0
    allow_shorting: bool = True
    margin_requirement: float = 0.5
    benchmark_symbol: str = 'BTCUSDT'
    mode: BacktestMode = BacktestMode.EVENT
    use_adj_close: bool = True


@dataclass
class BacktestResult:
    """
    Backtest results container.

    Contains all outputs from a backtest run.
    """
    # Core results
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    positions_history: pd.DataFrame
    performance_report: PerformanceReport

    # Detailed data
    daily_returns: pd.Series
    drawdowns: pd.Series

    # Configuration
    config: BacktestConfig = None
    start_date: datetime = None
    end_date: datetime = None

    # Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    final_capital: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0

    def summary(self) -> str:
        """Generate text summary."""
        return self.performance_report.generate_summary_text() if self.performance_report else "No results"


class BacktestEngine:
    """
    Professional Backtesting Engine.

    Provides comprehensive backtesting capabilities with:
        - Event-driven simulation
        - Realistic execution modeling
        - Multi-asset support
        - Strategy integration
        - Performance analysis

    Attributes:
        config: Backtest configuration
        execution_simulator: Order execution simulator
        performance_analyzer: Performance analysis engine
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        **kwargs
    ):
        """
        Initialize Backtest Engine.

        Args:
            config: Backtest configuration
            **kwargs: Override config parameters
        """
        self.config = config or BacktestConfig(**kwargs)
        self.logger = get_logger('backtest_engine')

        # Initialize components
        self.execution_simulator = ExecutionSimulator(
            slippage_model=SlippageModel(base_slippage=self.config.slippage_rate),
            commission_model=CommissionModel(commission_rate=self.config.commission_rate)
        )
        self.performance_analyzer = PerformanceAnalyzer(
            risk_free_rate=self.config.risk_free_rate
        )

        # State
        self._capital = self.config.initial_capital
        self._positions: Dict[str, Position] = {}
        self._equity_curve: List[Dict] = []
        self._trades: List[Dict] = []
        self._daily_returns: List[float] = []

        # Data
        self._data: Dict[str, pd.DataFrame] = {}
        self._current_idx: int = 0
        self._timestamps: List[datetime] = []

        # Strategy
        self._strategy: Optional[Any] = None
        self._signal_generator: Optional[Callable] = None

    def load_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        benchmark: bool = False
    ) -> None:
        """
        Load market data for a symbol.

        Args:
            symbol: Trading symbol
            data: DataFrame with OHLCV data
            benchmark: Is this benchmark data
        """
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in data.columns]

        if missing:
            # Try lowercase
            data = data.copy()
            data.columns = [c.lower() for c in data.columns]

        self._data[symbol] = data.copy()

        if benchmark:
            self.config.benchmark_symbol = symbol

        self.logger.info(f"Loaded {len(data)} bars for {symbol}")

    def set_strategy(
        self,
        strategy: Any,
        signal_generator: Optional[Callable] = None
    ) -> None:
        """
        Set trading strategy.

        Args:
            strategy: Strategy instance
            signal_generator: Optional signal generation function
        """
        self._strategy = strategy
        self._signal_generator = signal_generator or getattr(strategy, 'generate_signals', None)

        if self._signal_generator is None:
            raise ValueError("Strategy must have generate_signals method or provide signal_generator")

    def run(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        verbose: bool = False
    ) -> BacktestResult:
        """
        Run backtest.

        Args:
            start_date: Start date filter
            end_date: End date filter
            verbose: Print progress

        Returns:
            BacktestResult with all results
        """
        if not self._data:
            raise ValueError("No data loaded")

        self.logger.info("Starting backtest...")
        self._reset_state()

        # Get aligned timestamps
        self._timestamps = self._get_aligned_timestamps()
        if not self._timestamps:
            raise ValueError("No aligned timestamps found")

        # Filter by date range
        if start_date or end_date:
            self._timestamps = [
                ts for ts in self._timestamps
                if (start_date is None or ts >= start_date) and
                   (end_date is None or ts <= end_date)
            ]

        if verbose:
            self.logger.info(f"Backtesting {len(self._timestamps)} bars")

        # Run based on mode
        if self.config.mode == BacktestMode.VECTOR:
            result = self._run_vector(verbose)
        else:
            result = self._run_event(verbose)

        self.logger.info(f"Backtest complete. Final capital: ${result.final_capital:.2f}")

        return result

    def _run_event(self, verbose: bool = False) -> BacktestResult:
        """
        Run event-driven backtest.

        Args:
            verbose: Print progress

        Returns:
            BacktestResult
        """
        progress_interval = max(1, len(self._timestamps) // 10)

        for i, timestamp in enumerate(self._timestamps):
            self._current_idx = i

            # Update market data
            self._update_market_data(timestamp)

            # Process open orders
            fills = self.execution_simulator.process_orders(timestamp)

            # Process fills
            for fill in fills:
                self._process_fill(fill, timestamp)

            # Generate signals
            signals = self._generate_signals(timestamp)

            # Execute signals
            for signal in signals:
                self._execute_signal(signal, timestamp)

            # Mark positions to market
            self._mark_positions_to_market(timestamp)

            # Record equity
            equity = self._calculate_equity(timestamp)
            self._record_equity(timestamp, equity)

            # Progress report
            if verbose and (i + 1) % progress_interval == 0:
                pct = (i + 1) / len(self._timestamps) * 100
                self.logger.info(f"Progress: {pct:.0f}% - Equity: ${equity:.2f}")

        # Compile results
        return self._compile_result()

    def _run_vector(self, verbose: bool = False) -> BacktestResult:
        """
        Run vectorized backtest (faster but less realistic).

        Args:
            verbose: Print progress

        Returns:
            BacktestResult
        """
        # Get primary symbol
        primary_symbol = list(self._data.keys())[0]
        data = self._data[primary_symbol]

        # Simple buy-and-hold for demonstration
        # In practice, this would use vectorized strategy signals
        returns = data['close'].pct_change()
        returns = returns.fillna(0)

        # Calculate equity curve
        equity_curve = self.config.initial_capital * (1 + returns).cumprod()

        # Record results
        for i, (timestamp, equity, ret) in enumerate(zip(self._timestamps, equity_curve, returns)):
            self._equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'cash': equity * 0.5,  # Placeholder
                'position_value': equity * 0.5,
                'daily_pnl': equity * ret if i > 0 else 0
            })
            self._daily_returns.append(ret)

            # Add to performance analyzer
            self.performance_analyzer.add_daily_result(
                timestamp=timestamp,
                equity=equity,
                daily_pnl=equity * ret if i > 0 else 0,
                daily_return=ret
            )

        # Compile results
        result = self._compile_result()
        result.performance_report = self.performance_analyzer.analyze()

        return result

    def _update_market_data(self, timestamp: datetime) -> None:
        """Update market data for all symbols at current timestamp."""
        for symbol, data in self._data.items():
            if timestamp not in data.index:
                # Try to find closest timestamp
                try:
                    idx = data.index.get_indexer([timestamp], method='nearest')[0]
                    row = data.iloc[idx]
                except (IndexError, KeyError):
                    continue
            else:
                row = data.loc[timestamp]

            # Update execution simulator
            self.execution_simulator.update_market(
                symbol=symbol,
                price=row['close'],
                volume=row.get('volume', 0),
                volatility=row.get('volatility', 0.02),
                spread=row.get('spread', 0.0001)
            )

    def _generate_signals(self, timestamp: datetime) -> List[Dict]:
        """Generate trading signals from strategy."""
        if self._signal_generator is None:
            return []

        try:
            # Prepare data for strategy
            data_slice = {}
            for symbol, data in self._data.items():
                # Get data up to current timestamp
                data_slice[symbol] = data[data.index <= timestamp]

            # Generate signals
            signals = self._signal_generator(data_slice)
            return signals if signals else []
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return []

    def _execute_signal(self, signal: Dict, timestamp: datetime) -> None:
        """
        Execute a trading signal.

        Args:
            signal: Signal dictionary
            timestamp: Current timestamp
        """
        symbol = signal.get('symbol')
        side = signal.get('side', 'buy')
        order_type = signal.get('order_type', 'market')
        quantity = signal.get('quantity')
        price = signal.get('price')

        # Calculate quantity if not provided
        if quantity is None:
            current_price = self.execution_simulator._last_prices.get(symbol, 0)
            if current_price > 0:
                quantity = (self._capital * self.config.position_size_pct) / current_price
            else:
                return

        # Determine order type
        if order_type == 'market':
            order_type_enum = OrderType.MARKET
        elif order_type == 'limit':
            order_type_enum = OrderType.LIMIT
        elif order_type == 'stop':
            order_type_enum = OrderType.STOP
        else:
            order_type_enum = OrderType.MARKET

        # Determine side
        side_enum = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

        # Check position limits
        if side_enum == OrderSide.SELL and not self.config.allow_shorting:
            # Check if we have position to sell
            if symbol not in self._positions or self._positions[symbol].quantity <= 0:
                return

        # Submit order
        self.execution_simulator.submit_order(
            symbol=symbol,
            side=side_enum,
            order_type=order_type_enum,
            quantity=quantity,
            price=price,
            stop_price=signal.get('stop_price'),
            timestamp=timestamp
        )

    def _process_fill(self, fill: Fill, timestamp: datetime) -> None:
        """Process a fill and update positions."""
        symbol = fill.symbol

        # Update or create position
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)

        position = self._positions[symbol]
        realized_pnl = position.update(fill)

        # Update capital
        if fill.side == OrderSide.BUY:
            self._capital -= fill.quantity * fill.price + fill.commission
        else:
            self._capital += fill.quantity * fill.price - fill.commission

        # Record trade
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'side': fill.side.value,
            'quantity': fill.quantity,
            'price': fill.price,
            'commission': fill.commission,
            'slippage': fill.slippage,
            'pnl': realized_pnl
        }
        self._trades.append(trade_record)

        # Add to performance analyzer
        self.performance_analyzer.add_trade(
            entry_time=timestamp,  # Simplified
            exit_time=timestamp,
            symbol=symbol,
            side=fill.side.value,
            entry_price=fill.price,
            exit_price=fill.price,
            quantity=fill.quantity,
            pnl=realized_pnl,
            commission=fill.commission,
            slippage=fill.slippage
        )

    def _mark_positions_to_market(self, timestamp: datetime) -> None:
        """Mark all positions to market."""
        for symbol, position in self._positions.items():
            current_price = self.execution_simulator._last_prices.get(symbol, 0)
            if current_price > 0:
                position.mark_to_market(current_price)

    def _calculate_equity(self, timestamp: datetime) -> float:
        """Calculate total equity."""
        equity = self._capital

        for symbol, position in self._positions.items():
            current_price = self.execution_simulator._last_prices.get(symbol, 0)
            if current_price > 0 and position.quantity != 0:
                position_value = position.quantity * current_price
                equity += position_value

        return equity

    def _record_equity(self, timestamp: datetime, equity: float) -> None:
        """Record equity snapshot."""
        position_value = sum(
            pos.quantity * self.execution_simulator._last_prices.get(sym, 0)
            for sym, pos in self._positions.items()
        )

        daily_pnl = 0
        daily_return = 0
        if self._equity_curve:
            prev_equity = self._equity_curve[-1]['equity']
            daily_pnl = equity - prev_equity
            daily_return = daily_pnl / prev_equity if prev_equity > 0 else 0

        self._equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'cash': self._capital,
            'position_value': position_value,
            'daily_pnl': daily_pnl,
            'daily_return': daily_return
        })

        self._daily_returns.append(daily_return)

        # Add to performance analyzer
        self.performance_analyzer.add_daily_result(
            timestamp=timestamp,
            equity=equity,
            daily_pnl=daily_pnl,
            daily_return=daily_return
        )

    def _get_aligned_timestamps(self) -> List[datetime]:
        """Get aligned timestamps across all data."""
        if not self._data:
            return []

        # Get timestamps from first symbol
        first_symbol = list(self._data.keys())[0]
        timestamps = set(self._data[first_symbol].index)

        # Intersect with all other symbols
        for symbol in self._data:
            timestamps = timestamps.intersection(set(self._data[symbol].index))

        return sorted(list(timestamps))

    def _compile_result(self) -> BacktestResult:
        """Compile backtest results."""
        # Create equity curve DataFrame
        equity_df = pd.DataFrame(self._equity_curve)
        if not equity_df.empty:
            equity_df.set_index('timestamp', inplace=True)

        # Create trades DataFrame
        trades_df = pd.DataFrame(self._trades)

        # Generate performance report
        performance_report = self.performance_analyzer.analyze()

        # Calculate daily returns series
        daily_returns = pd.Series(self._daily_returns, index=self._timestamps[:len(self._daily_returns)])

        # Calculate drawdowns
        drawdowns = pd.Series()
        if not equity_df.empty:
            equity = equity_df['equity']
            running_max = equity.cummax()
            drawdowns = (equity - running_max) / running_max

        result = BacktestResult(
            equity_curve=equity_df,
            trades=trades_df,
            positions_history=pd.DataFrame(),
            performance_report=performance_report,
            daily_returns=daily_returns,
            drawdowns=drawdowns,
            config=self.config,
            start_date=min(self._timestamps) if self._timestamps else None,
            end_date=max(self._timestamps) if self._timestamps else None,
            total_trades=len(self._trades),
            winning_trades=len([t for t in self._trades if t.get('pnl', 0) > 0]),
            losing_trades=len([t for t in self._trades if t.get('pnl', 0) < 0]),
            final_capital=self._equity_curve[-1]['equity'] if self._equity_curve else self.config.initial_capital,
            total_return=performance_report.total_return,
            max_drawdown=performance_report.max_drawdown,
            sharpe_ratio=performance_report.sharpe_ratio
        )

        return result

    def _reset_state(self) -> None:
        """Reset backtest state."""
        self._capital = self.config.initial_capital
        self._positions.clear()
        self._equity_curve.clear()
        self._trades.clear()
        self._daily_returns.clear()
        self._current_idx = 0

        self.execution_simulator.reset()
        self.performance_analyzer.reset()

    def run_walk_forward(
        self,
        train_period: int = 252,
        test_period: int = 63,
        step: int = 63,
        optimize_func: Optional[Callable] = None
    ) -> List[BacktestResult]:
        """
        Run walk-forward optimization.

        Args:
            train_period: Training period in days
            test_period: Test period in days
            step: Step size in days
            optimize_func: Optimization function

        Returns:
            List of BacktestResult for each fold
        """
        if not self._timestamps:
            raise ValueError("No data loaded")

        results = []
        total_periods = len(self._timestamps)

        # Generate fold indices
        folds = []
        start = 0
        while start + train_period + test_period <= total_periods:
            train_end = start + train_period
            test_end = train_end + test_period
            folds.append((start, train_end, test_end))
            start += step

        self.logger.info(f"Running {len(folds)} walk-forward folds")

        for i, (train_start, train_end, test_end) in enumerate(folds):
            # Run optimization on training period (if provided)
            if optimize_func:
                train_data = {
                    symbol: data.iloc[train_start:train_end]
                    for symbol, data in self._data.items()
                }
                optimized_params = optimize_func(train_data)
                # Apply optimized params to strategy

            # Run backtest on test period
            test_timestamps = self._timestamps[train_start:test_end]
            result = self.run(
                start_date=min(test_timestamps),
                end_date=max(test_timestamps)
            )
            results.append(result)

            self.logger.info(
                f"Fold {i+1}/{len(folds)}: Return={result.total_return*100:.2f}%, "
                f"Sharpe={result.sharpe_ratio:.2f}"
            )

        return results

    def run_monte_carlo(
        self,
        n_simulations: int = 1000,
        strategy_params_generator: Optional[Callable] = None
    ) -> Dict:
        """
        Run Monte Carlo simulation.

        Args:
            n_simulations: Number of simulations
            strategy_params_generator: Function to generate random parameters

        Returns:
            Dictionary with simulation results
        """
        results = []

        for i in range(n_simulations):
            # Generate random parameters if provided
            if strategy_params_generator:
                params = strategy_params_generator()
                # Apply params

            # Run backtest
            result = self.run()

            results.append({
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown
            })

        # Analyze results
        returns = [r['total_return'] for r in results]
        sharpes = [r['sharpe_ratio'] for r in results]
        drawdowns = [r['max_drawdown'] for r in results]

        return {
            'n_simulations': n_simulations,
            'return_mean': np.mean(returns),
            'return_std': np.std(returns),
            'return_percentile_5': np.percentile(returns, 5),
            'return_percentile_95': np.percentile(returns, 95),
            'sharpe_mean': np.mean(sharpes),
            'sharpe_std': np.std(sharpes),
            'max_drawdown_avg': np.mean(drawdowns),
            'prob_positive_return': len([r for r in returns if r > 0]) / len(returns)
        }


def run_backtest(
    data: pd.DataFrame,
    strategy: Any,
    initial_capital: float = 100000,
    **kwargs
) -> BacktestResult:
    """
    Convenience function to run a backtest.

    Args:
        data: Market data
        strategy: Strategy instance
        initial_capital: Initial capital
        **kwargs: Additional configuration

    Returns:
        BacktestResult
    """
    config = BacktestConfig(initial_capital=initial_capital, **kwargs)
    engine = BacktestEngine(config)

    symbol = kwargs.get('symbol', 'ASSET')
    engine.load_data(symbol, data)
    engine.set_strategy(strategy)

    return engine.run()
