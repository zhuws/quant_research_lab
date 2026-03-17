"""
Walk-Forward Validation for Quant Research Lab.

Provides professional walk-forward validation capabilities:
    - Rolling walk-forward analysis
    - Anchored walk-forward analysis
    - Parameter optimization on training windows
    - Out-of-sample testing
    - Performance aggregation

Walk-forward validation is essential for:
    - Avoiding overfitting in strategy development
    - Robust parameter selection
    - Realistic performance estimation
    - Strategy stability assessment

Example usage:
    ```python
    from backtest.walk_forward import WalkForwardValidator, WalkForwardConfig

    config = WalkForwardConfig(
        train_period=252,  # 1 year training
        test_period=63,    # 3 months testing
        step=63,           # Move forward 3 months each fold
        anchored=False     # Rolling window
    )

    validator = WalkForwardValidator(config)
    results = validator.run(data, strategy)
    ```
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.math_utils import safe_divide


class WalkForwardMode(Enum):
    """Walk-forward validation modes."""
    ROLLING = 'rolling'  # Fixed-size rolling training window
    ANCHORED = 'anchored'  # Expanding training window from start


class OptimizationTarget(Enum):
    """Optimization target metrics."""
    SHARPE = 'sharpe'
    RETURN = 'return'
    IC = 'ic'
    SORTINO = 'sortino'
    CALMAR = 'calmar'


@dataclass
class WalkForwardConfig:
    """
    Configuration for walk-forward validation.

    Attributes:
        train_period: Number of periods for training window
        test_period: Number of periods for test window
        step: Number of periods to step forward between folds
        mode: Rolling or anchored walk-forward
        optimization_target: Metric to optimize during training
        n_optimization_trials: Number of parameter optimization trials
        min_train_samples: Minimum samples required for training
        commission_rate: Trading commission rate
        slippage_rate: Slippage rate
        initial_capital: Starting capital for backtests
        allow_shorting: Whether to allow short positions
        verbose: Print progress messages
    """
    train_period: int = 252
    test_period: int = 63
    step: int = 63
    mode: WalkForwardMode = WalkForwardMode.ROLLING
    optimization_target: OptimizationTarget = OptimizationTarget.SHARPE
    n_optimization_trials: int = 20
    min_train_samples: int = 100
    commission_rate: float = 0.001
    slippage_rate: float = 0.0001
    initial_capital: float = 100000
    allow_shorting: bool = True
    verbose: bool = True

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.train_period < self.min_train_samples:
            raise ValueError(
                f"train_period ({self.train_period}) must be >= "
                f"min_train_samples ({self.min_train_samples})"
            )
        if self.test_period < 1:
            raise ValueError(f"test_period must be >= 1, got {self.test_period}")
        if self.step < 1:
            raise ValueError(f"step must be >= 1, got {self.step}")


@dataclass
class FoldResult:
    """
    Result for a single walk-forward fold.

    Attributes:
        fold_number: Fold index
        train_start: Training period start index
        train_end: Training period end index
        test_start: Test period start index
        test_end: Test period end index
        train_metrics: Performance metrics on training data
        test_metrics: Performance metrics on test data
        optimized_params: Parameters optimized on training data
        equity_curve: Equity curve during test period
        trades: List of trades during test period
        train_dates: Actual dates for training period
        test_dates: Actual dates for test period
    """
    fold_number: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    optimized_params: Dict[str, Any] = field(default_factory=dict)
    equity_curve: Optional[pd.DataFrame] = None
    trades: Optional[pd.DataFrame] = None
    train_dates: Optional[Tuple[datetime, datetime]] = None
    test_dates: Optional[Tuple[datetime, datetime]] = None

    def summary(self) -> str:
        """Generate summary string for this fold."""
        return (
            f"Fold {self.fold_number}: "
            f"Train Sharpe={self.train_metrics.get('sharpe_ratio', 0):.2f}, "
            f"Test Sharpe={self.test_metrics.get('sharpe_ratio', 0):.2f}, "
            f"Test Return={self.test_metrics.get('total_return', 0)*100:.2f}%"
        )


@dataclass
class WalkForwardResult:
    """
    Complete results from walk-forward validation.

    Attributes:
        config: Configuration used
        fold_results: Results for each fold
        aggregated_metrics: Aggregated performance metrics
        parameters_stability: Analysis of parameter stability across folds
        performance_degradation: Train vs test performance comparison
        recommendations: Generated recommendations
        execution_time: Total execution time in seconds
    """
    config: WalkForwardConfig
    fold_results: List[FoldResult] = field(default_factory=list)
    aggregated_metrics: Dict[str, float] = field(default_factory=dict)
    parameters_stability: Dict[str, Any] = field(default_factory=dict)
    performance_degradation: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    execution_time: float = 0.0

    def summary(self) -> str:
        """Generate summary of walk-forward results."""
        if not self.fold_results:
            return "No fold results available"

        lines = [
            "=" * 60,
            "WALK-FORWARD VALIDATION RESULTS",
            "=" * 60,
            f"Folds: {len(self.fold_results)}",
            f"Aggregated Test Sharpe: {self.aggregated_metrics.get('avg_test_sharpe', 0):.2f}",
            f"Aggregated Test Return: {self.aggregated_metrics.get('avg_test_return', 0)*100:.2f}%",
            f"Performance Degradation: {self.performance_degradation.get('sharpe_degradation', 0)*100:.1f}%",
            "",
            "FOLD DETAILS:"
        ]

        for fold in self.fold_results:
            lines.append(f"  {fold.summary()}")

        if self.recommendations:
            lines.append("")
            lines.append("RECOMMENDATIONS:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            'config': {
                'train_period': self.config.train_period,
                'test_period': self.config.test_period,
                'step': self.config.step,
                'mode': self.config.mode.value,
                'optimization_target': self.config.optimization_target.value
            },
            'aggregated_metrics': self.aggregated_metrics,
            'parameters_stability': self.parameters_stability,
            'performance_degradation': self.performance_degradation,
            'recommendations': self.recommendations,
            'execution_time': self.execution_time,
            'n_folds': len(self.fold_results),
            'fold_summaries': [fold.summary() for fold in self.fold_results]
        }

    def save(self, filepath: str) -> None:
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class WalkForwardValidator:
    """
    Professional Walk-Forward Validation Engine.

    Provides comprehensive walk-forward validation with:
        - Rolling and anchored modes
        - Parameter optimization
        - Performance degradation analysis
        - Parameter stability assessment

    Walk-forward validation simulates how a strategy would have performed
    if it had been developed and deployed in a realistic manner, avoiding
    the pitfalls of overfitting to historical data.

    Example:
        ```python
        config = WalkForwardConfig(
            train_period=252,
            test_period=63,
            step=63
        )
        validator = WalkForwardValidator(config)
        results = validator.run(data, strategy)
        print(results.summary())
        ```
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        """
        Initialize Walk-Forward Validator.

        Args:
            config: Walk-forward configuration
        """
        self.config = config or WalkForwardConfig()
        self.config.validate()
        self.logger = get_logger('walk_forward_validator')

        # Internal state
        self._data: Optional[pd.DataFrame] = None
        self._strategy: Optional[Any] = None
        self._optimize_func: Optional[Callable] = None
        self._backtest_func: Optional[Callable] = None

    def run(
        self,
        data: pd.DataFrame,
        strategy: Any,
        optimize_func: Optional[Callable] = None,
        backtest_func: Optional[Callable] = None
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.

        Args:
            data: Market data DataFrame with OHLCV columns
            strategy: Strategy instance with generate_signals method
            optimize_func: Optional function to optimize parameters
                          Signature: func(train_data, strategy) -> Dict[str, Any]
            backtest_func: Optional custom backtest function
                          Signature: func(data, strategy, params) -> Dict[str, float]

        Returns:
            WalkForwardResult with comprehensive validation results
        """
        start_time = time.time()

        self._data = data.copy()
        self._strategy = strategy
        self._optimize_func = optimize_func
        self._backtest_func = backtest_func

        # Validate data
        self._validate_data()

        # Generate fold indices
        folds = self._generate_folds()

        if not folds:
            raise ValueError(
                f"Insufficient data for walk-forward validation. "
                f"Need at least {self.config.train_period + self.config.test_period} samples, "
                f"got {len(data)}"
            )

        self.logger.info(
            f"Running walk-forward validation with {len(folds)} folds "
            f"({self.config.mode.value} mode)"
        )

        # Run validation for each fold
        fold_results = []
        for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
            if self.config.verbose:
                self.logger.info(f"Processing fold {i+1}/{len(folds)}")

            fold_result = self._run_fold(i, train_start, train_end, test_start, test_end)
            fold_results.append(fold_result)

            if self.config.verbose:
                self.logger.info(f"  {fold_result.summary()}")

        # Compile results
        result = WalkForwardResult(
            config=self.config,
            fold_results=fold_results,
            execution_time=time.time() - start_time
        )

        # Calculate aggregated metrics
        result.aggregated_metrics = self._aggregate_metrics(fold_results)

        # Analyze parameter stability
        result.parameters_stability = self._analyze_parameter_stability(fold_results)

        # Calculate performance degradation
        result.performance_degradation = self._calculate_degradation(fold_results)

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)

        self.logger.info(f"Walk-forward validation complete in {result.execution_time:.2f}s")

        return result

    def _validate_data(self) -> None:
        """Validate input data."""
        if self._data is None or self._data.empty:
            raise ValueError("Data cannot be empty")

        required_columns = ['close']
        for col in required_columns:
            if col not in self._data.columns:
                raise ValueError(f"Missing required column: {col}")

        if len(self._data) < self.config.train_period + self.config.test_period:
            raise ValueError(
                f"Insufficient data: need at least "
                f"{self.config.train_period + self.config.test_period} samples, "
                f"got {len(self._data)}"
            )

    def _generate_folds(self) -> List[Tuple[int, int, int, int]]:
        """
        Generate fold indices for walk-forward validation.

        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        n_samples = len(self._data)
        folds = []

        if self.config.mode == WalkForwardMode.ROLLING:
            # Rolling window: fixed-size training window
            start = 0
            while start + self.config.train_period + self.config.test_period <= n_samples:
                train_start = start
                train_end = start + self.config.train_period
                test_start = train_end
                test_end = test_start + self.config.test_period

                folds.append((train_start, train_end, test_start, test_end))
                start += self.config.step

        else:  # ANCHORED
            # Anchored: expanding training window from start
            train_end = self.config.train_period
            while train_end + self.config.test_period <= n_samples:
                train_start = 0
                test_start = train_end
                test_end = test_start + self.config.test_period

                folds.append((train_start, train_end, test_start, test_end))
                train_end += self.config.step

        return folds

    def _run_fold(
        self,
        fold_number: int,
        train_start: int,
        train_end: int,
        test_start: int,
        test_end: int
    ) -> FoldResult:
        """
        Run validation for a single fold.

        Args:
            fold_number: Fold index
            train_start: Training start index
            train_end: Training end index
            test_start: Test start index
            test_end: Test end index

        Returns:
            FoldResult with metrics and data
        """
        # Split data
        train_data = self._data.iloc[train_start:train_end].copy()
        test_data = self._data.iloc[test_start:test_end].copy()

        # Get dates if available
        train_dates = None
        test_dates = None
        if hasattr(self._data, 'index') and isinstance(self._data.index, pd.DatetimeIndex):
            train_dates = (self._data.index[train_start], self._data.index[train_end - 1])
            test_dates = (self._data.index[test_start], self._data.index[min(test_end - 1, len(self._data) - 1)])

        # Optimize parameters on training data
        optimized_params = {}
        if self._optimize_func is not None:
            optimized_params = self._optimize_func(train_data, self._strategy)
        else:
            optimized_params = self._default_optimization(train_data)

        # Run backtests
        train_metrics = self._run_backtest(train_data, optimized_params)
        test_metrics = self._run_backtest(test_data, optimized_params)

        return FoldResult(
            fold_number=fold_number,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            optimized_params=optimized_params,
            train_dates=train_dates,
            test_dates=test_dates
        )

    def _default_optimization(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Default parameter optimization using simple grid search.

        Args:
            train_data: Training data

        Returns:
            Optimized parameters dictionary
        """
        # Get strategy's default parameters
        if hasattr(self._strategy, 'get_default_params'):
            return self._strategy.get_default_params()
        elif hasattr(self._strategy, 'get_params'):
            return self._strategy.get_params()
        else:
            return {}

    def _run_backtest(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Run backtest with given parameters.

        Args:
            data: Market data
            params: Strategy parameters

        Returns:
            Dictionary of performance metrics
        """
        if self._backtest_func is not None:
            return self._backtest_func(data, self._strategy, params)

        # Default backtest implementation
        return self._default_backtest(data, params)

    def _default_backtest(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Default backtest implementation.

        Args:
            data: Market data
            params: Strategy parameters

        Returns:
            Dictionary of performance metrics
        """
        # Apply parameters to strategy if possible
        if hasattr(self._strategy, 'set_params'):
            self._strategy.set_params(params)

        # Generate signals
        try:
            if hasattr(self._strategy, 'generate_signals'):
                signals = self._strategy.generate_signals(data)
            elif hasattr(self._strategy, 'generate_signal'):
                signals = self._strategy.generate_signal(data)
            else:
                signals = None
        except Exception as e:
            self.logger.warning(f"Error generating signals: {e}")
            signals = None

        # Calculate returns
        if signals is not None and len(signals) > 0:
            returns = self._calculate_returns(data, signals)
        else:
            # Use buy-and-hold as fallback
            returns = data['close'].pct_change().fillna(0)

        # Calculate metrics
        metrics = self._calculate_metrics(returns, data)

        return metrics

    def _calculate_returns(
        self,
        data: pd.DataFrame,
        signals: Union[pd.Series, pd.DataFrame, List[Dict]]
    ) -> pd.Series:
        """
        Calculate strategy returns from signals.

        Args:
            data: Market data
            signals: Trading signals

        Returns:
            Series of strategy returns
        """
        if isinstance(signals, list):
            # Convert list of dicts to series
            signal_series = pd.Series(0.0, index=data.index)
            for signal in signals:
                if isinstance(signal, dict):
                    idx = signal.get('timestamp') or signal.get('index')
                    pos = signal.get('position', signal.get('signal', 0))
                    if idx is not None and idx in signal_series.index:
                        signal_series.loc[idx] = pos
            signals = signal_series
        elif isinstance(signals, pd.DataFrame):
            if 'signal' in signals.columns:
                signals = signals['signal']
            elif 'position' in signals.columns:
                signals = signals['position']
            else:
                signals = signals.iloc[:, 0]

        # Align signals with data
        if len(signals) != len(data):
            signals = signals.reindex(data.index, method='ffill').fillna(0)

        # Calculate returns
        market_returns = data['close'].pct_change().fillna(0)
        strategy_returns = signals.shift(1) * market_returns  # Lag signals by 1

        # Apply commission
        commission = self.config.commission_rate
        trades = signals.diff().abs()
        strategy_returns = strategy_returns - trades * commission

        return strategy_returns.fillna(0)

    def _calculate_metrics(
        self,
        returns: pd.Series,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.

        Args:
            returns: Strategy returns
            data: Original market data

        Returns:
            Dictionary of metrics
        """
        if len(returns) == 0 or returns.std() == 0:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'ic': 0.0
            }

        # Total return
        total_return = (1 + returns).prod() - 1

        # Sharpe ratio (annualized assuming daily data)
        periods_per_year = 252
        if len(returns) > 1:
            sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year)
        else:
            sharpe = 0.0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.001
        sortino = returns.mean() / downside_std * np.sqrt(periods_per_year) if downside_std > 0 else 0.0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        winning_days = (returns > 0).sum()
        total_days = (returns != 0).sum()
        win_rate = winning_days / total_days if total_days > 0 else 0.0

        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Information coefficient (correlation with future returns)
        future_returns = data['close'].pct_change().shift(-1).dropna()
        if len(future_returns) > 0 and len(returns) > 0:
            aligned = pd.concat([returns, future_returns], axis=1).dropna()
            if len(aligned) > 1:
                ic = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            else:
                ic = 0.0
        else:
            ic = 0.0

        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'ic': float(ic)
        }

    def _aggregate_metrics(self, fold_results: List[FoldResult]) -> Dict[str, float]:
        """
        Aggregate metrics across all folds.

        Args:
            fold_results: List of fold results

        Returns:
            Aggregated metrics dictionary
        """
        if not fold_results:
            return {}

        # Collect test metrics
        test_sharpes = [f.test_metrics.get('sharpe_ratio', 0) for f in fold_results]
        test_returns = [f.test_metrics.get('total_return', 0) for f in fold_results]
        test_drawdowns = [f.test_metrics.get('max_drawdown', 0) for f in fold_results]
        test_ics = [f.test_metrics.get('ic', 0) for f in fold_results]

        train_sharpes = [f.train_metrics.get('sharpe_ratio', 0) for f in fold_results]

        return {
            'avg_test_sharpe': float(np.mean(test_sharpes)),
            'std_test_sharpe': float(np.std(test_sharpes)),
            'avg_test_return': float(np.mean(test_returns)),
            'std_test_return': float(np.std(test_returns)),
            'avg_test_drawdown': float(np.mean(test_drawdowns)),
            'avg_test_ic': float(np.mean(test_ics)),
            'avg_train_sharpe': float(np.mean(train_sharpes)),
            'positive_test_folds': sum(1 for r in test_returns if r > 0),
            'total_folds': len(fold_results),
            'test_sharpe_consistency': sum(1 for s in test_sharpes if s > 0) / len(test_sharpes) if test_sharpes else 0
        }

    def _analyze_parameter_stability(self, fold_results: List[FoldResult]) -> Dict[str, Any]:
        """
        Analyze stability of optimized parameters across folds.

        Args:
            fold_results: List of fold results

        Returns:
            Parameter stability analysis
        """
        if not fold_results:
            return {}

        # Collect all parameter values across folds
        param_values = {}
        for fold in fold_results:
            for param, value in fold.optimized_params.items():
                if param not in param_values:
                    param_values[param] = []
                param_values[param].append(value)

        stability_analysis = {}
        for param, values in param_values.items():
            if len(values) > 1:
                # Calculate stability metrics
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    stability_analysis[param] = {
                        'mean': float(np.mean(numeric_values)),
                        'std': float(np.std(numeric_values)),
                        'min': float(np.min(numeric_values)),
                        'max': float(np.max(numeric_values)),
                        'cv': float(np.std(numeric_values) / np.mean(numeric_values)) if np.mean(numeric_values) != 0 else 0,
                        'values': numeric_values
                    }

        # Overall stability score (lower CV = more stable)
        if stability_analysis:
            cvs = [s['cv'] for s in stability_analysis.values() if s['cv'] > 0]
            avg_cv = np.mean(cvs) if cvs else 0
            stability_score = max(0, 1 - avg_cv)  # Higher is more stable
        else:
            stability_score = 1.0

        return {
            'parameters': stability_analysis,
            'stability_score': float(stability_score),
            'is_stable': stability_score > 0.7
        }

    def _calculate_degradation(self, fold_results: List[FoldResult]) -> Dict[str, float]:
        """
        Calculate performance degradation from train to test.

        High degradation indicates overfitting.

        Args:
            fold_results: List of fold results

        Returns:
            Degradation metrics
        """
        if not fold_results:
            return {}

        train_sharpes = [f.train_metrics.get('sharpe_ratio', 0) for f in fold_results]
        test_sharpes = [f.test_metrics.get('sharpe_ratio', 0) for f in fold_results]

        train_returns = [f.train_metrics.get('total_return', 0) for f in fold_results]
        test_returns = [f.test_metrics.get('total_return', 0) for f in fold_results]

        avg_train_sharpe = np.mean(train_sharpes)
        avg_test_sharpe = np.mean(test_sharpes)
        avg_train_return = np.mean(train_returns)
        avg_test_return = np.mean(test_returns)

        # Calculate degradation (positive = degradation)
        sharpe_degradation = (avg_train_sharpe - avg_test_sharpe) / abs(avg_train_sharpe) if avg_train_sharpe != 0 else 0
        return_degradation = (avg_train_return - avg_test_return) / abs(avg_train_return) if avg_train_return != 0 else 0

        return {
            'sharpe_degradation': float(sharpe_degradation),
            'return_degradation': float(return_degradation),
            'avg_train_sharpe': float(avg_train_sharpe),
            'avg_test_sharpe': float(avg_test_sharpe),
            'is_overfitting': sharpe_degradation > 0.3  # >30% degradation indicates overfitting
        }

    def _generate_recommendations(self, result: WalkForwardResult) -> List[str]:
        """
        Generate recommendations based on validation results.

        Args:
            result: Walk-forward validation result

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check for overfitting
        if result.performance_degradation.get('is_overfitting', False):
            recommendations.append(
                "High performance degradation detected. Consider simplifying the strategy "
                "or reducing the number of optimized parameters to avoid overfitting."
            )

        # Check parameter stability
        if not result.parameters_stability.get('is_stable', True):
            recommendations.append(
                "Parameter values vary significantly across folds. Consider using more "
                "robust parameter values or reducing parameter sensitivity."
            )

        # Check Sharpe ratio
        avg_sharpe = result.aggregated_metrics.get('avg_test_sharpe', 0)
        if avg_sharpe < 0:
            recommendations.append(
                "Negative average test Sharpe ratio. The strategy underperforms "
                "risk-free rate. Consider revising the strategy logic."
            )
        elif avg_sharpe < 0.5:
            recommendations.append(
                "Low Sharpe ratio on test data. Consider improving risk-adjusted returns."
            )
        elif avg_sharpe > 2.0:
            recommendations.append(
                "Exceptionally high Sharpe ratio. Verify there are no data issues "
                "or look-ahead bias in the strategy."
            )

        # Check consistency
        consistency = result.aggregated_metrics.get('test_sharpe_consistency', 0)
        if consistency < 0.5:
            recommendations.append(
                f"Only {consistency*100:.0f}% of folds have positive Sharpe ratio. "
                "Strategy performance is inconsistent across time periods."
            )

        # Check drawdown
        avg_dd = result.aggregated_metrics.get('avg_test_drawdown', 0)
        if avg_dd < -0.2:
            recommendations.append(
                f"Average max drawdown of {avg_dd*100:.1f}% is significant. "
                "Consider adding risk management controls."
            )

        # Check IC
        avg_ic = result.aggregated_metrics.get('avg_test_ic', 0)
        if abs(avg_ic) < 0.02:
            recommendations.append(
                "Low information coefficient. The strategy's predictive power is weak."
            )

        # Positive feedback
        if avg_sharpe > 1.0 and consistency > 0.7 and not result.performance_degradation.get('is_overfitting', False):
            recommendations.append(
                "Strategy shows robust performance with good consistency across folds."
            )

        return recommendations


class WalkForwardOptimizer(WalkForwardValidator):
    """
    Walk-Forward Validation with integrated parameter optimization.

    Extends WalkForwardValidator with automatic parameter optimization
    during the training phase of each fold.
    """

    def __init__(
        self,
        config: Optional[WalkForwardConfig] = None,
        param_grid: Optional[Dict[str, List[Any]]] = None
    ):
        """
        Initialize Walk-Forward Optimizer.

        Args:
            config: Walk-forward configuration
            param_grid: Parameter grid for optimization
        """
        super().__init__(config)
        self.param_grid = param_grid or {}

    def run_with_optimization(
        self,
        data: pd.DataFrame,
        strategy: Any,
        param_grid: Optional[Dict[str, List[Any]]] = None
    ) -> WalkForwardResult:
        """
        Run walk-forward validation with parameter optimization.

        Args:
            data: Market data
            strategy: Strategy instance
            param_grid: Parameter grid for optimization

        Returns:
            WalkForwardResult
        """
        param_grid = param_grid or self.param_grid

        if not param_grid:
            self.logger.warning("No parameter grid provided, running without optimization")
            return self.run(data, strategy)

        def optimize_func(train_data: pd.DataFrame, strategy: Any) -> Dict[str, Any]:
            """Optimize parameters using grid search."""
            best_params = {}
            best_score = -np.inf

            # Get all parameter combinations
            from itertools import product
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())

            for combo in product(*param_values):
                params = dict(zip(param_names, combo))

                # Apply parameters
                if hasattr(strategy, 'set_params'):
                    strategy.set_params(params)

                # Run backtest
                metrics = self._run_backtest(train_data, params)

                # Get optimization target
                target = self.config.optimization_target.value
                score = metrics.get(target, metrics.get('sharpe_ratio', 0))

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

            return best_params

        return self.run(data, strategy, optimize_func=optimize_func)


def run_walk_forward(
    data: pd.DataFrame,
    strategy: Any,
    train_period: int = 252,
    test_period: int = 63,
    step: int = 63,
    **kwargs
) -> WalkForwardResult:
    """
    Convenience function to run walk-forward validation.

    Args:
        data: Market data DataFrame
        strategy: Strategy instance
        train_period: Training period length
        test_period: Test period length
        step: Step size between folds
        **kwargs: Additional configuration parameters

    Returns:
        WalkForwardResult
    """
    config = WalkForwardConfig(
        train_period=train_period,
        test_period=test_period,
        step=step,
        **kwargs
    )

    validator = WalkForwardValidator(config)
    return validator.run(data, strategy)


# Export classes
__all__ = [
    'WalkForwardMode',
    'OptimizationTarget',
    'WalkForwardConfig',
    'FoldResult',
    'WalkForwardResult',
    'WalkForwardValidator',
    'WalkForwardOptimizer',
    'run_walk_forward'
]
