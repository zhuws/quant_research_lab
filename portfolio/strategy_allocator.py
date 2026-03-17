"""
Strategy Allocator for Quant Research Lab.

Manages capital allocation across multiple trading strategies:
    - Performance-based allocation
    - Risk parity across strategies
    - Dynamic rebalancing
    - Strategy ranking and selection
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


class AllocationMethod(Enum):
    """Strategy allocation methods."""
    EQUAL = 'equal'
    RISK_PARITY = 'risk_parity'
    SHARPE_WEIGHTED = 'sharpe_weighted'
    PERFORMANCE_BASED = 'performance_based'
    KELLY = 'kelly'
    VOLATILITY_TARGETING = 'volatility_targeting'
    MAX_SHARPE = 'max_sharpe'


@dataclass
class StrategyPerformance:
    """Performance metrics for a single strategy."""
    name: str
    returns: pd.Series = field(default_factory=pd.Series)
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_return: float = 0.0
    volatility: float = 0.0
    total_trades: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    def update_metrics(self, risk_free_rate: float = 0.02):
        """Calculate and update performance metrics."""
        if self.returns.empty:
            return

        # Annualized return
        self.avg_return = self.returns.mean() * 252

        # Annualized volatility
        self.volatility = self.returns.std() * np.sqrt(252)

        # Sharpe ratio
        if self.volatility > 0:
            self.sharpe_ratio = (self.avg_return - risk_free_rate) / self.volatility
        else:
            self.sharpe_ratio = 0

        # Sortino ratio
        downside_returns = self.returns[self.returns < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std() * np.sqrt(252)
            self.sortino_ratio = safe_divide(
                self.avg_return - risk_free_rate,
                downside_vol,
                default=0
            )

        # Max drawdown
        cumulative = (1 + self.returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        self.max_drawdown = abs(drawdown.min())

        # Calmar ratio
        if self.max_drawdown > 0:
            self.calmar_ratio = safe_divide(self.avg_return, self.max_drawdown, default=0)

        # Win rate
        if len(self.returns) > 0:
            self.win_rate = (self.returns > 0).sum() / len(self.returns)

        # Profit factor
        gains = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        self.profit_factor = safe_divide(gains, losses, default=0)

        self.last_updated = datetime.now()


class StrategyAllocator:
    """
    Multi-Strategy Capital Allocator.

    Allocates capital across multiple trading strategies based on
    performance, risk, and correlation.

    Features:
        - Multiple allocation methods
        - Dynamic rebalancing
        - Performance tracking
        - Risk budgeting
        - Strategy selection

    Attributes:
        allocation_method: Method for allocating capital
        rebalance_frequency: How often to rebalance (days)
        min_strategy_weight: Minimum weight per strategy
        max_strategy_weight: Maximum weight per strategy
        lookback_period: Lookback period for performance calculation
    """

    def __init__(
        self,
        allocation_method: str = 'risk_parity',
        rebalance_frequency: int = 7,
        min_strategy_weight: float = 0.0,
        max_strategy_weight: float = 1.0,
        lookback_period: int = 252,
        risk_free_rate: float = 0.02,
        target_volatility: Optional[float] = None
    ):
        """
        Initialize Strategy Allocator.

        Args:
            allocation_method: Allocation method name
            rebalance_frequency: Days between rebalancing
            min_strategy_weight: Minimum allocation to any strategy
            max_strategy_weight: Maximum allocation to any strategy
            lookback_period: Days to look back for performance
            risk_free_rate: Annual risk-free rate
            target_volatility: Target portfolio volatility (for vol targeting)
        """
        self.logger = get_logger('strategy_allocator')

        try:
            self.allocation_method = AllocationMethod(allocation_method)
        except ValueError:
            self.logger.warning(f"Unknown method {allocation_method}, using risk_parity")
            self.allocation_method = AllocationMethod.RISK_PARITY

        self.rebalance_frequency = rebalance_frequency
        self.min_strategy_weight = min_strategy_weight
        self.max_strategy_weight = max_strategy_weight
        self.lookback_period = lookback_period
        self.risk_free_rate = risk_free_rate
        self.target_volatility = target_volatility

        # Strategy registry
        self._strategies: Dict[str, StrategyPerformance] = {}

        # Current allocation
        self._current_weights: Dict[str, float] = {}
        self._last_rebalance: Optional[datetime] = None

        # Performance history
        self._weight_history: List[Tuple[datetime, Dict[str, float]]] = []

    def register_strategy(
        self,
        name: str,
        returns: Optional[pd.Series] = None
    ) -> None:
        """
        Register a trading strategy.

        Args:
            name: Strategy name
            returns: Historical returns (optional)
        """
        if name in self._strategies:
            self.logger.warning(f"Strategy {name} already registered, updating")
            self._strategies[name].returns = returns or pd.Series()
            self._strategies[name].update_metrics(self.risk_free_rate)
        else:
            self._strategies[name] = StrategyPerformance(
                name=name,
                returns=returns or pd.Series()
            )
            self._strategies[name].update_metrics(self.risk_free_rate)

        # Initialize with equal weight if not set
        if name not in self._current_weights:
            n_strategies = len(self._strategies)
            equal_weight = 1.0 / n_strategies
            self._current_weights = {s: equal_weight for s in self._strategies}

        self.logger.info(f"Registered strategy: {name}")

    def update_strategy_returns(
        self,
        name: str,
        returns: pd.Series
    ) -> None:
        """
        Update strategy with new returns.

        Args:
            name: Strategy name
            returns: New returns data
        """
        if name not in self._strategies:
            self.register_strategy(name, returns)
        else:
            # Append new returns
            if self._strategies[name].returns.empty:
                self._strategies[name].returns = returns
            else:
                self._strategies[name].returns = pd.concat([
                    self._strategies[name].returns,
                    returns
                ])

            # Trim to lookback period
            if len(self._strategies[name].returns) > self.lookback_period:
                self._strategies[name].returns = self._strategies[name].returns.tail(self.lookback_period)

            # Update metrics
            self._strategies[name].update_metrics(self.risk_free_rate)

    def allocate(
        self,
        strategy_returns: Optional[Dict[str, pd.Series]] = None,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate strategy allocations.

        Args:
            strategy_returns: Dictionary of strategy returns (if different from stored)
            custom_weights: Custom weights to use (overrides allocation method)

        Returns:
            Dictionary of strategy weights
        """
        if custom_weights is not None:
            self._current_weights = self._normalize_weights(custom_weights)
            return self._current_weights

        # Update strategy returns if provided
        if strategy_returns:
            for name, returns in strategy_returns.items():
                self.update_strategy_returns(name, returns)

        if not self._strategies:
            self.logger.warning("No strategies registered")
            return {}

        # Dispatch to allocation method
        method_map = {
            AllocationMethod.EQUAL: self._equal_allocation,
            AllocationMethod.RISK_PARITY: self._risk_parity_allocation,
            AllocationMethod.SHARPE_WEIGHTED: self._sharpe_weighted_allocation,
            AllocationMethod.PERFORMANCE_BASED: self._performance_based_allocation,
            AllocationMethod.KELLY: self._kelly_allocation,
            AllocationMethod.VOLATILITY_TARGETING: self._volatility_targeting_allocation,
            AllocationMethod.MAX_SHARPE: self._max_sharpe_allocation
        }

        weights = method_map[self.allocation_method]()

        # Apply constraints
        weights = self._apply_constraints(weights)

        # Store weights
        self._current_weights = weights
        self._last_rebalance = datetime.now()
        self._weight_history.append((datetime.now(), weights.copy()))

        self.logger.info(f"Allocated using {self.allocation_method.value}: {weights}")

        return weights

    def _equal_allocation(self) -> Dict[str, float]:
        """Equal weight allocation."""
        n_strategies = len(self._strategies)
        weight = 1.0 / n_strategies
        return {name: weight for name in self._strategies}

    def _risk_parity_allocation(self) -> Dict[str, float]:
        """Risk parity allocation (inverse volatility weighting)."""
        weights = {}
        total_inv_vol = 0

        for name, perf in self._strategies.items():
            if perf.volatility > 0:
                inv_vol = 1.0 / perf.volatility
            else:
                inv_vol = 1.0
            weights[name] = inv_vol
            total_inv_vol += inv_vol

        # Normalize
        for name in weights:
            weights[name] /= total_inv_vol

        return weights

    def _sharpe_weighted_allocation(self) -> Dict[str, float]:
        """Allocate based on Sharpe ratios."""
        weights = {}
        total_sharpe = 0

        for name, perf in self._strategies.items():
            # Use positive Sharpe only
            sharpe = max(0, perf.sharpe_ratio)
            weights[name] = sharpe
            total_sharpe += sharpe

        if total_sharpe == 0:
            return self._equal_allocation()

        # Normalize
        for name in weights:
            weights[name] /= total_sharpe

        return weights

    def _performance_based_allocation(self) -> Dict[str, float]:
        """Allocate based on composite performance score."""
        weights = {}
        total_score = 0

        for name, perf in self._strategies.items():
            # Composite score: weighted combination of metrics
            score = (
                0.3 * max(0, perf.sharpe_ratio) +
                0.2 * max(0, perf.sortino_ratio) +
                0.2 * max(0, perf.calmar_ratio) +
                0.15 * perf.win_rate +
                0.15 * min(perf.profit_factor, 3) / 3
            )
            weights[name] = score
            total_score += score

        if total_score == 0:
            return self._equal_allocation()

        # Normalize
        for name in weights:
            weights[name] /= total_score

        return weights

    def _kelly_allocation(self) -> Dict[str, float]:
        """Kelly criterion allocation."""
        weights = {}

        for name, perf in self._strategies.items():
            if perf.volatility > 0:
                # Kelly fraction = excess_return / variance
                excess_return = perf.avg_return - self.risk_free_rate
                kelly = safe_divide(excess_return, perf.volatility ** 2, default=0)

                # Apply half-Kelly for safety
                kelly = max(0, min(0.5 * kelly, self.max_strategy_weight))
            else:
                kelly = self.min_strategy_weight

            weights[name] = kelly

        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            for name in weights:
                weights[name] /= total
        else:
            return self._equal_allocation()

        return weights

    def _volatility_targeting_allocation(self) -> Dict[str, float]:
        """Allocate to target portfolio volatility."""
        if self.target_volatility is None:
            return self._risk_parity_allocation()

        # Start with risk parity weights
        base_weights = self._risk_parity_allocation()

        # Calculate portfolio volatility
        returns_df = self._get_returns_dataframe()
        if returns_df.empty:
            return base_weights

        weight_array = np.array([base_weights.get(s, 0) for s in returns_df.columns])
        cov_matrix = returns_df.cov() * 252
        port_vol = np.sqrt(weight_array @ cov_matrix.values @ weight_array)

        # Scale weights to hit target volatility
        if port_vol > 0:
            scale = self.target_volatility / port_vol
            scaled_weights = {name: min(w * scale, self.max_strategy_weight)
                          for name, w in base_weights.items()}
            return self._normalize_weights(scaled_weights)

        return base_weights

    def _max_sharpe_allocation(self) -> Dict[str, float]:
        """Maximize portfolio Sharpe ratio."""
        returns_df = self._get_returns_dataframe()
        if returns_df.empty:
            return self._equal_allocation()

        from portfolio.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer(
            risk_free_rate=self.risk_free_rate,
            max_weight=self.max_strategy_weight,
            min_weight=self.min_strategy_weight
        )

        result = optimizer.optimize(returns_df, method='sharpe')
        return result.get('weights', self._equal_allocation())

    def _get_returns_dataframe(self) -> pd.DataFrame:
        """Get aligned returns DataFrame for all strategies."""
        returns_dict = {}
        for name, perf in self._strategies.items():
            if not perf.returns.empty:
                returns_dict[name] = perf.returns

        if not returns_dict:
            return pd.DataFrame()

        df = pd.DataFrame(returns_dict)
        return df.dropna()

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1."""
        total = sum(weights.values())
        if total == 0:
            return self._equal_allocation()
        return {name: w / total for name, w in weights.items()}

    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply min/max weight constraints."""
        constrained = {}
        for name, w in weights.items():
            w = max(self.min_strategy_weight, min(w, self.max_strategy_weight))
            constrained[name] = w

        # Renormalize
        return self._normalize_weights(constrained)

    def should_rebalance(self) -> bool:
        """Check if rebalancing is needed."""
        if self._last_rebalance is None:
            return True

        days_since_rebalance = (datetime.now() - self._last_rebalance).days
        return days_since_rebalance >= self.rebalance_frequency

    def get_current_weights(self) -> Dict[str, float]:
        """Get current strategy weights."""
        return self._current_weights.copy()

    def get_strategy_metrics(self) -> pd.DataFrame:
        """Get performance metrics for all strategies."""
        metrics = []
        for name, perf in self._strategies.items():
            metrics.append({
                'strategy': name,
                'sharpe_ratio': perf.sharpe_ratio,
                'sortino_ratio': perf.sortino_ratio,
                'calmar_ratio': perf.calmar_ratio,
                'max_drawdown': perf.max_drawdown,
                'win_rate': perf.win_rate,
                'profit_factor': perf.profit_factor,
                'avg_return': perf.avg_return,
                'volatility': perf.volatility,
                'total_trades': perf.total_trades
            })

        return pd.DataFrame(metrics)

    def rank_strategies(
        self,
        metric: str = 'sharpe_ratio',
        ascending: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Rank strategies by performance metric.

        Args:
            metric: Metric to rank by
            ascending: Sort ascending

        Returns:
            List of (strategy_name, metric_value) tuples
        """
        rankings = []
        for name, perf in self._strategies.items():
            value = getattr(perf, metric, 0)
            rankings.append((name, value))

        rankings.sort(key=lambda x: x[1], reverse=not ascending)
        return rankings

    def select_top_strategies(
        self,
        n: int = 3,
        metric: str = 'sharpe_ratio'
    ) -> List[str]:
        """
        Select top N strategies by metric.

        Args:
            n: Number of strategies to select
            metric: Performance metric

        Returns:
            List of strategy names
        """
        rankings = self.rank_strategies(metric)
        return [name for name, _ in rankings[:n]]

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix between strategies."""
        returns_df = self._get_returns_dataframe()
        if returns_df.empty:
            return pd.DataFrame()
        return returns_df.corr()

    def get_diversification_score(self) -> float:
        """
        Calculate portfolio diversification score.

        Higher score = more diversified.

        Returns:
            Diversification score (0-1)
        """
        corr_matrix = self.calculate_correlation_matrix()
        if corr_matrix.empty:
            return 0

        # Average absolute correlation (lower is better)
        n = len(corr_matrix)
        if n < 2:
            return 1

        # Get upper triangle (excluding diagonal)
        upper_tri = corr_matrix.values[np.triu_indices(n, k=1)]
        avg_abs_corr = np.mean(np.abs(upper_tri))

        # Diversification score (1 - average correlation)
        return 1 - avg_abs_corr

    def get_weight_history(self) -> pd.DataFrame:
        """Get historical weight allocations."""
        if not self._weight_history:
            return pd.DataFrame()

        records = []
        for timestamp, weights in self._weight_history:
            record = {'timestamp': timestamp}
            record.update(weights)
            records.append(record)

        return pd.DataFrame(records)

    def generate_report(self) -> Dict:
        """
        Generate comprehensive strategy allocation report.

        Returns:
            Dictionary with full allocation analysis
        """
        metrics_df = self.get_strategy_metrics()
        corr_matrix = self.calculate_correlation_matrix()

        return {
            'current_weights': self._current_weights,
            'strategy_metrics': metrics_df.to_dict('records'),
            'correlation_matrix': corr_matrix.to_dict() if not corr_matrix.empty else {},
            'diversification_score': self.get_diversification_score(),
            'allocation_method': self.allocation_method.value,
            'last_rebalance': self._last_rebalance.isoformat() if self._last_rebalance else None,
            'n_strategies': len(self._strategies),
            'top_performers': self.select_top_strategies(3)
        }


def create_strategy_allocator(
    strategy_names: List[str],
    method: str = 'risk_parity',
    **kwargs
) -> StrategyAllocator:
    """
    Convenience function to create and initialize a strategy allocator.

    Args:
        strategy_names: List of strategy names to register
        method: Allocation method
        **kwargs: Additional arguments for StrategyAllocator

    Returns:
        Initialized StrategyAllocator
    """
    allocator = StrategyAllocator(allocation_method=method, **kwargs)

    for name in strategy_names:
        allocator.register_strategy(name)

    return allocator
