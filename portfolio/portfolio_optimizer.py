"""
Portfolio Optimizer for Quant Research Lab.

Implements multiple portfolio optimization methods:
    - Mean-Variance Optimization (Markowitz)
    - Risk Parity (Equal Risk Contribution)
    - Maximum Sharpe Ratio
    - Minimum Variance
    - Hierarchical Risk Parity (HRP)
    - Black-Litterman Model
    - Maximum Diversification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import warnings
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.math_utils import safe_divide


class PortfolioOptimizer:
    """
    Portfolio Optimization Engine.

    Provides multiple optimization methods for portfolio construction:
        - Mean-variance optimization
        - Risk parity
        - Sharpe maximization
        - Minimum variance
        - Hierarchical risk parity
        - Black-Litterman

    Attributes:
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        max_weight: Maximum weight per asset
        min_weight: Minimum weight per asset
        target_return: Target return for constrained optimization
        target_risk: Target risk for constrained optimization
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None,
        optimization_method: str = 'sharpe'
    ):
        """
        Initialize Portfolio Optimizer.

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
            max_weight: Maximum weight per asset (default: 1.0)
            min_weight: Minimum weight per asset (default: 0.0)
            target_return: Target portfolio return for optimization
            target_risk: Target portfolio risk for optimization
            optimization_method: Default optimization method
        """
        self.logger = get_logger('portfolio_optimizer')
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.target_return = target_return
        self.target_risk = target_risk
        self.optimization_method = optimization_method

        # Cache for optimization results
        self._last_weights = None
        self._last_metrics = None

    def optimize(
        self,
        returns: pd.DataFrame,
        method: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Optimize portfolio weights using specified method.

        Args:
            returns: DataFrame of asset returns (columns = assets)
            method: Optimization method ('mean_variance', 'risk_parity',
                    'sharpe', 'min_variance', 'hrp', 'max_diversification')
            **kwargs: Additional method-specific arguments

        Returns:
            Dictionary with weights and metrics
        """
        method = method or self.optimization_method

        if returns.empty:
            self.logger.error("Empty returns DataFrame")
            return {'weights': {}, 'metrics': {}}

        # Calculate expected returns and covariance matrix
        expected_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252  # Annualized

        # Dispatch to appropriate optimization method
        method_map = {
            'mean_variance': self._mean_variance_optimization,
            'risk_parity': self._risk_parity_optimization,
            'sharpe': self._max_sharpe_optimization,
            'min_variance': self._min_variance_optimization,
            'hrp': self._hrp_optimization,
            'max_diversification': self._max_diversification_optimization,
            'black_litterman': self._black_litterman_optimization
        }

        if method not in method_map:
            self.logger.error(f"Unknown optimization method: {method}")
            return {'weights': {}, 'metrics': {}}

        self.logger.info(f"Running {method} optimization for {len(returns.columns)} assets")

        # Run optimization
        result = method_map[method](
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            returns=returns,
            **kwargs
        )

        # Calculate portfolio metrics
        metrics = self._calculate_portfolio_metrics(
            weights=result['weights'],
            expected_returns=expected_returns,
            cov_matrix=cov_matrix
        )

        result['metrics'] = metrics
        self._last_weights = result['weights']
        self._last_metrics = metrics

        return result

    def _mean_variance_optimization(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        target_return: Optional[float] = None,
        **kwargs
    ) -> Dict:
        """
        Mean-Variance Optimization (Markowitz).

        Minimizes portfolio variance for a given target return.

        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            target_return: Target portfolio return (annualized)

        Returns:
            Dictionary with optimal weights
        """
        n_assets = len(expected_returns)
        target_return = target_return or self.target_return

        # Objective: minimize portfolio variance
        def portfolio_variance(weights):
            return weights @ cov_matrix.values @ weights

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]

        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: w @ expected_returns.values - target_return
            })

        # Bounds
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))

        # Initial guess: equal weights
        initial_weights = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )

        if not result.success:
            self.logger.warning(f"Optimization failed: {result.message}")
            return {'weights': dict(zip(expected_returns.index, initial_weights))}

        weights = result.x
        weights_dict = dict(zip(expected_returns.index, weights))

        return {'weights': weights_dict}

    def _max_sharpe_optimization(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        **kwargs
    ) -> Dict:
        """
        Maximum Sharpe Ratio Optimization.

        Maximizes the Sharpe ratio of the portfolio.

        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix

        Returns:
            Dictionary with optimal weights
        """
        n_assets = len(expected_returns)
        daily_rf = self.risk_free_rate / 252

        # Objective: minimize negative Sharpe ratio
        def negative_sharpe(weights):
            port_return = weights @ expected_returns.values
            port_volatility = np.sqrt(weights @ cov_matrix.values @ weights)
            if port_volatility == 0:
                return 1e10
            sharpe = (port_return - self.risk_free_rate) / port_volatility
            return -sharpe

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        # Bounds
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))

        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )

        if not result.success:
            self.logger.warning(f"Sharpe optimization failed: {result.message}")

        weights = result.x
        # Clean up small weights
        weights[weights < 1e-6] = 0
        weights = weights / weights.sum()  # Renormalize

        weights_dict = dict(zip(expected_returns.index, weights))

        return {'weights': weights_dict}

    def _min_variance_optimization(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        **kwargs
    ) -> Dict:
        """
        Minimum Variance Portfolio Optimization.

        Minimizes portfolio variance without return constraint.

        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix

        Returns:
            Dictionary with optimal weights
        """
        n_assets = len(expected_returns)

        # Objective: minimize portfolio variance
        def portfolio_variance(weights):
            return weights @ cov_matrix.values @ weights

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        # Bounds
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))

        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )

        if not result.success:
            self.logger.warning(f"Min variance optimization failed: {result.message}")

        weights = result.x
        weights[weights < 1e-6] = 0
        weights = weights / weights.sum()

        weights_dict = dict(zip(expected_returns.index, weights))

        return {'weights': weights_dict}

    def _risk_parity_optimization(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        target_risk_contributions: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict:
        """
        Risk Parity (Equal Risk Contribution) Optimization.

        Each asset contributes equally to portfolio risk.

        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            target_risk_contributions: Target risk contribution per asset

        Returns:
            Dictionary with optimal weights
        """
        n_assets = len(expected_returns)

        # Default: equal risk contribution
        if target_risk_contributions is None:
            target_risk = np.ones(n_assets) / n_assets
        else:
            target_risk = np.array([
                target_risk_contributions.get(asset, 1/n_assets)
                for asset in expected_returns.index
            ])
            target_risk = target_risk / target_risk.sum()

        def risk_contribution_objective(weights):
            """Objective: minimize squared difference from target risk contributions."""
            # Portfolio variance
            port_var = weights @ cov_matrix.values @ weights
            port_vol = np.sqrt(port_var)

            # Marginal risk contribution
            marginal_contrib = cov_matrix.values @ weights

            # Risk contribution
            risk_contrib = weights * marginal_contrib / port_vol

            # Normalize
            risk_contrib = risk_contrib / risk_contrib.sum()

            # Squared error from target
            return np.sum((risk_contrib - target_risk) ** 2)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        # Bounds
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))

        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            risk_contribution_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )

        if not result.success:
            self.logger.warning(f"Risk parity optimization failed: {result.message}")

        weights = result.x
        weights[weights < 1e-6] = 0
        weights = weights / weights.sum()

        weights_dict = dict(zip(expected_returns.index, weights))

        return {'weights': weights_dict}

    def _hrp_optimization(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        returns: pd.DataFrame,
        linkage_method: str = 'ward',
        **kwargs
    ) -> Dict:
        """
        Hierarchical Risk Parity (HRP) Optimization.

        Uses hierarchical clustering to build a diversified portfolio.

        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            returns: Original returns DataFrame
            linkage_method: Clustering linkage method

        Returns:
            Dictionary with optimal weights
        """
        n_assets = len(expected_returns)

        # Step 1: Calculate correlation distance matrix
        corr = returns.corr()
        dist = np.sqrt(0.5 * (1 - corr))

        # Step 2: Hierarchical clustering
        dist_values = squareform(dist.values)
        link = linkage(dist_values, method=linkage_method)

        # Step 3: Quasi-diagonalization
        sort_idx = list(leaves_list(link))

        # Step 4: Recursive bisection
        weights = np.ones(n_assets)

        # Cluster variances
        cluster_items = [sort_idx]

        while cluster_items:
            cluster_items = [
                cluster[start:end]
                for cluster in cluster_items
                for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
                if 1 < len(cluster)
            ]

            for cluster in cluster_items:
                # Split cluster into two halves
                half = len(cluster) // 2
                left_items = cluster[:half]
                right_items = cluster[half:]

                # Calculate variance of each half
                left_cov = cov_matrix.iloc[left_items, left_items]
                right_cov = cov_matrix.iloc[right_items, right_items]

                left_var = self._cluster_variance(left_cov)
                right_var = self._cluster_variance(right_cov)

                # Allocate based on inverse variance
                total_var = left_var + right_var
                left_weight = right_var / total_var
                right_weight = left_var / total_var

                # Update weights
                weights[left_items] *= left_weight
                weights[right_items] *= right_weight

        weights_dict = dict(zip(expected_returns.index, weights))

        return {'weights': weights_dict}

    def _cluster_variance(self, cov_matrix: pd.DataFrame) -> float:
        """Calculate variance of a cluster with inverse-variance weighting."""
        ivp_weights = 1.0 / np.diag(cov_matrix)
        ivp_weights = ivp_weights / ivp_weights.sum()
        return ivp_weights @ cov_matrix.values @ ivp_weights

    def _max_diversification_optimization(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        **kwargs
    ) -> Dict:
        """
        Maximum Diversification Optimization.

        Maximizes the diversification ratio.

        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix

        Returns:
            Dictionary with optimal weights
        """
        n_assets = len(expected_returns)

        # Asset volatilities
        volatilities = np.sqrt(np.diag(cov_matrix.values))

        def negative_diversification_ratio(weights):
            """Negative diversification ratio (to minimize)."""
            weighted_vol = weights @ volatilities
            port_vol = np.sqrt(weights @ cov_matrix.values @ weights)
            if port_vol == 0:
                return 1e10
            return -weighted_vol / port_vol

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        # Bounds
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))

        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            negative_diversification_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )

        if not result.success:
            self.logger.warning(f"Max diversification optimization failed: {result.message}")

        weights = result.x
        weights[weights < 1e-6] = 0
        weights = weights / weights.sum()

        weights_dict = dict(zip(expected_returns.index, weights))

        return {'weights': weights_dict}

    def _black_litterman_optimization(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        market_weights: Optional[np.ndarray] = None,
        views: Optional[Dict] = None,
        view_confidences: Optional[List[float]] = None,
        tau: float = 0.05,
        **kwargs
    ) -> Dict:
        """
        Black-Litterman Model.

        Combines market equilibrium returns with investor views.

        Args:
            expected_returns: Expected returns (used as prior if no market weights)
            cov_matrix: Covariance matrix
            market_weights: Market capitalization weights
            views: Dictionary of views {'asset': (relative_asset, view_return)}
            view_confidences: Confidence in each view
            tau: Scaling parameter for prior covariance

        Returns:
            Dictionary with optimal weights
        """
        n_assets = len(expected_returns)

        # Market equilibrium returns (if market weights provided)
        if market_weights is not None:
            # Implied equilibrium returns
            risk_aversion = 3.0  # Typical value
            pi = risk_aversion * cov_matrix.values @ market_weights
        else:
            # Use provided expected returns as prior
            pi = expected_returns.values

        # If no views, return market weights or equal weights
        if views is None or len(views) == 0:
            if market_weights is not None:
                return {'weights': dict(zip(expected_returns.index, market_weights))}
            else:
                equal_weights = np.ones(n_assets) / n_assets
                return {'weights': dict(zip(expected_returns.index, equal_weights))}

        # Build views matrix
        P = np.zeros((len(views), n_assets))  # Views matrix
        Q = np.zeros(len(views))  # Views vector

        assets_list = list(expected_returns.index)

        for i, (asset, view) in enumerate(views.items()):
            asset_idx = assets_list.index(asset)
            if isinstance(view, tuple):
                relative_asset, view_return = view
                relative_idx = assets_list.index(relative_asset)
                P[i, asset_idx] = 1
                P[i, relative_idx] = -1
                Q[i] = view_return
            else:
                P[i, asset_idx] = 1
                Q[i] = view

        # Views uncertainty matrix (Omega)
        if view_confidences is not None:
            omega = np.diag([1/c for c in view_confidences])
        else:
            # Default: proportional to view variance
            omega = np.diag(np.diag(P @ cov_matrix.values @ P.T)) * tau

        # Black-Litterman formula
        tau_sigma = tau * cov_matrix.values

        # Posterior expected returns
        M = np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(omega) @ P
        posterior_returns = np.linalg.inv(M) @ (
            np.linalg.inv(tau_sigma) @ pi + P.T @ np.linalg.inv(omega) @ Q
        )

        # Posterior covariance
        posterior_cov = np.linalg.inv(M) + cov_matrix.values

        # Optimize with posterior estimates
        posterior_returns_series = pd.Series(posterior_returns, index=expected_returns.index)
        posterior_cov_df = pd.DataFrame(
            posterior_cov,
            index=cov_matrix.index,
            columns=cov_matrix.columns
        )

        # Use maximum Sharpe with posterior estimates
        return self._max_sharpe_optimization(
            expected_returns=posterior_returns_series,
            cov_matrix=posterior_cov_df
        )

    def _calculate_portfolio_metrics(
        self,
        weights: Dict[str, float],
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> Dict:
        """
        Calculate portfolio performance metrics.

        Args:
            weights: Asset weights
            expected_returns: Expected returns
            cov_matrix: Covariance matrix

        Returns:
            Dictionary of portfolio metrics
        """
        # Convert weights to array
        weight_array = np.array([weights.get(asset, 0) for asset in expected_returns.index])

        # Expected return (annualized)
        port_return = weight_array @ expected_returns.values

        # Portfolio variance and volatility (annualized)
        port_variance = weight_array @ cov_matrix.values @ weight_array
        port_volatility = np.sqrt(port_variance)

        # Sharpe ratio
        sharpe_ratio = safe_divide(
            port_return - self.risk_free_rate,
            port_volatility,
            default=0
        )

        # Diversification ratio
        volatilities = np.sqrt(np.diag(cov_matrix.values))
        weighted_vol = weight_array @ volatilities
        diversification_ratio = safe_divide(weighted_vol, port_volatility, default=1)

        # Effective number of assets (inverse Herfindahl)
        herfindahl = np.sum(weight_array ** 2)
        effective_assets = safe_divide(1, herfindahl, default=1)

        # Maximum drawdown estimate (from variance)
        max_drawdown_estimate = 2 * port_volatility  # Approximate

        # Risk contributions
        marginal_risk = cov_matrix.values @ weight_array
        risk_contributions = weight_array * marginal_risk / port_volatility
        risk_contributions_pct = risk_contributions / risk_contributions.sum() if risk_contributions.sum() > 0 else risk_contributions

        return {
            'expected_return': port_return,
            'volatility': port_volatility,
            'variance': port_variance,
            'sharpe_ratio': sharpe_ratio,
            'diversification_ratio': diversification_ratio,
            'effective_assets': effective_assets,
            'max_drawdown_estimate': max_drawdown_estimate,
            'risk_contributions': dict(zip(expected_returns.index, risk_contributions_pct)),
            'weights': weights
        }

    def efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_points: int = 50,
        min_return: Optional[float] = None,
        max_return: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generate efficient frontier points.

        Args:
            returns: DataFrame of asset returns
            n_points: Number of points on the frontier
            min_return: Minimum return (default: min variance portfolio return)
            max_return: Maximum return (default: max asset return)

        Returns:
            DataFrame with frontier points (return, volatility, sharpe)
        """
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # Get min variance portfolio
        min_var_result = self._min_variance_optimization(expected_returns, cov_matrix)
        min_var_weights = np.array([
            min_var_result['weights'].get(asset, 0)
            for asset in expected_returns.index
        ])
        min_var_return = min_var_weights @ expected_returns.values

        # Get max return
        max_asset_return = expected_returns.max()

        # Set return range
        min_return = min_return or min_var_return
        max_return = max_return or max_asset_return

        # Generate frontier points
        target_returns = np.linspace(min_return, max_return, n_points)

        frontier_points = []

        for target_ret in target_returns:
            try:
                result = self._mean_variance_optimization(
                    expected_returns,
                    cov_matrix,
                    target_return=target_ret
                )
                weights = result['weights']
                weight_array = np.array([
                    weights.get(asset, 0)
                    for asset in expected_returns.index
                ])

                port_return = weight_array @ expected_returns.values
                port_vol = np.sqrt(weight_array @ cov_matrix.values @ weight_array)
                sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

                frontier_points.append({
                    'target_return': target_ret,
                    'return': port_return,
                    'volatility': port_vol,
                    'sharpe_ratio': sharpe
                })
            except Exception as e:
                self.logger.debug(f"Failed to find frontier point at {target_ret}: {e}")
                continue

        return pd.DataFrame(frontier_points)

    def rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        threshold: float = 0.05
    ) -> Dict:
        """
        Calculate rebalancing trades.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            threshold: Rebalance threshold (drift percentage)

        Returns:
            Dictionary with trade instructions
        """
        trades = {}
        total_drift = 0

        all_assets = set(current_weights.keys()) | set(target_weights.keys())

        for asset in all_assets:
            current = current_weights.get(asset, 0)
            target = target_weights.get(asset, 0)
            drift = abs(current - target)
            total_drift += drift

            if drift > threshold:
                trades[asset] = {
                    'current_weight': current,
                    'target_weight': target,
                    'trade_size': target - current,
                    'drift': drift
                }

        return {
            'trades': trades,
            'total_drift': total_drift,
            'needs_rebalance': len(trades) > 0
        }

    def get_risk_budget(
        self,
        weights: Dict[str, float],
        cov_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate risk budget (risk contribution per asset).

        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix

        Returns:
            Dictionary of risk contributions
        """
        weight_array = np.array([weights.get(asset, 0) for asset in cov_matrix.index])
        port_vol = np.sqrt(weight_array @ cov_matrix.values @ weight_array)

        if port_vol == 0:
            return {asset: 0 for asset in cov_matrix.index}

        marginal_risk = cov_matrix.values @ weight_array
        risk_contrib = weight_array * marginal_risk / port_vol
        risk_contrib_pct = risk_contrib / risk_contrib.sum()

        return dict(zip(cov_matrix.index, risk_contrib_pct))

    def monte_carlo_simulation(
        self,
        weights: Dict[str, float],
        returns: pd.DataFrame,
        n_simulations: int = 10000,
        time_horizon: int = 252
    ) -> Dict:
        """
        Monte Carlo simulation for portfolio performance.

        Args:
            weights: Portfolio weights
            returns: Historical returns DataFrame
            n_simulations: Number of simulations
            time_horizon: Time horizon in days

        Returns:
            Dictionary with simulation results
        """
        weight_array = np.array([weights.get(asset, 0) for asset in returns.columns])
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values

        # Simulate returns
        simulated_returns = np.random.multivariate_normal(
            mean_returns,
            cov_matrix,
            size=(n_simulations, time_horizon)
        )

        # Calculate portfolio returns
        portfolio_returns = simulated_returns @ weight_array

        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + portfolio_returns, axis=1)

        # Calculate final values (starting at 1)
        final_values = cumulative_returns[:, -1]

        # Calculate statistics
        results = {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_25': np.percentile(final_values, 25),
            'percentile_75': np.percentile(final_values, 75),
            'percentile_95': np.percentile(final_values, 95),
            'prob_loss': np.mean(final_values < 1),
            'prob_double': np.mean(final_values > 2),
            'var_95': 1 - np.percentile(final_values, 5),
            'cvar_95': 1 - np.mean(final_values[final_values <= np.percentile(final_values, 5)])
        }

        return results


def optimize_portfolio(
    returns: pd.DataFrame,
    method: str = 'sharpe',
    **kwargs
) -> Dict:
    """
    Convenience function for portfolio optimization.

    Args:
        returns: DataFrame of asset returns
        method: Optimization method
        **kwargs: Additional arguments

    Returns:
        Optimization result dictionary
    """
    optimizer = PortfolioOptimizer(optimization_method=method)
    return optimizer.optimize(returns, method=method, **kwargs)
