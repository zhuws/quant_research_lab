"""
Factor Evaluator for Quant Research Lab.
Evaluates alpha factors using IC, Sharpe, turnover, and other metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.math_utils import winsorize


@dataclass
class FactorEvaluation:
    """
    Results of factor evaluation.

    Attributes:
        name: Factor name
        ic: Information Coefficient (correlation with forward returns)
        ic_ir: IC Information Ratio (IC mean / IC std)
        ic_tstat: IC t-statistic
        sharpe: Sharpe ratio of factor-based strategy
        turnover: Average turnover
        fitness: Overall fitness score
        decay: Alpha decay profile
    """
    name: str
    ic: float = 0.0
    ic_ir: float = 0.0
    ic_tstat: float = 0.0
    sharpe: float = 0.0
    turnover: float = 0.0
    fitness: float = 0.0
    decay: Optional[List[float]] = None
    stability: float = 0.0


class FactorEvaluator:
    """
    Evaluate alpha factors using various metrics.

    Metrics calculated:
        - Information Coefficient (IC)
        - IC Information Ratio
        - Factor Sharpe Ratio
        - Turnover
        - Alpha Decay
        - Factor Stability
    """

    def __init__(
        self,
        forward_horizons: List[int] = None,
        ic_method: str = 'spearman',
        quantile_bins: int = 5,
        winsorize_limits: Tuple[float, float] = (0.01, 0.99)
    ):
        """
        Initialize Factor Evaluator.

        Args:
            forward_horizons: Forward return horizons to evaluate
            ic_method: Correlation method ('spearman', 'pearson')
            quantile_bins: Number of quantile bins for quantile analysis
            winsorize_limits: Winsorization limits for factor values
        """
        self.logger = get_logger('factor_evaluator')
        self.forward_horizons = forward_horizons or [1, 5, 10, 20]
        self.ic_method = ic_method
        self.quantile_bins = quantile_bins
        self.winsorize_limits = winsorize_limits

    def evaluate_factor(
        self,
        factor_values: pd.Series,
        prices: pd.DataFrame,
        horizon: int = 10,
        include_decay: bool = True
    ) -> FactorEvaluation:
        """
        Evaluate a single factor.

        Args:
            factor_values: Series with factor values
            prices: DataFrame with OHLCV data
            horizon: Forward return horizon
            include_decay: Whether to calculate decay profile

        Returns:
            FactorEvaluation object with results
        """
        result = FactorEvaluation(name=factor_values.name or 'factor')

        # Clean factor values
        factor_clean = winsorize(factor_values.dropna(), limits=self.winsorize_limits)

        # Calculate forward returns
        forward_returns = self._calculate_forward_returns(prices['close'], horizon)
        forward_returns = forward_returns.dropna()

        # Align index
        common_idx = factor_clean.index.intersection(forward_returns.index)
        if len(common_idx) < 100:
            self.logger.warning(f"Insufficient data for evaluation: {len(common_idx)}")
            return result

        factor_clean = factor_clean.loc[common_idx]
        forward_returns = forward_returns.loc[common_idx]

        # Calculate IC
        ic_result = self._calculate_ic(factor_clean, forward_returns)
        result.ic = ic_result['ic']
        result.ic_ir = ic_result['ic_ir']
        result.ic_tstat = ic_result['ic_tstat']

        # Calculate rolling IC for stability
        result.stability = self._calculate_ic_stability(factor_clean, forward_returns)

        # Calculate Sharpe ratio of quantile strategy
        result.sharpe = self._calculate_quantile_sharpe(factor_clean, forward_returns)

        # Calculate turnover
        result.turnover = self._calculate_turnover(factor_clean)

        # Calculate fitness score
        result.fitness = self._calculate_fitness(result)

        # Calculate decay profile
        if include_decay:
            result.decay = self._calculate_decay(factor_clean, prices['close'])

        return result

    def evaluate_factors(
        self,
        factors_df: pd.DataFrame,
        prices: pd.DataFrame,
        horizon: int = 10,
        n_jobs: int = 1
    ) -> pd.DataFrame:
        """
        Evaluate multiple factors.

        Args:
            factors_df: DataFrame with factor values (columns are factors)
            prices: DataFrame with OHLCV data
            horizon: Forward return horizon
            n_jobs: Number of parallel jobs (not implemented, kept for API compatibility)

        Returns:
            DataFrame with evaluation results
        """
        results = []

        for col in factors_df.columns:
            try:
                eval_result = self.evaluate_factor(
                    factors_df[col],
                    prices,
                    horizon=horizon,
                    include_decay=False
                )
                results.append({
                    'factor': eval_result.name,
                    'ic': eval_result.ic,
                    'ic_ir': eval_result.ic_ir,
                    'ic_tstat': eval_result.ic_tstat,
                    'sharpe': eval_result.sharpe,
                    'turnover': eval_result.turnover,
                    'stability': eval_result.stability,
                    'fitness': eval_result.fitness
                })
            except Exception as e:
                self.logger.warning(f"Error evaluating factor {col}: {e}")

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('fitness', ascending=False)

        self.logger.info(f"Evaluated {len(results_df)} factors")

        return results_df

    def _calculate_forward_returns(
        self,
        close: pd.Series,
        horizon: int
    ) -> pd.Series:
        """Calculate forward returns."""
        forward_returns = np.log(close.shift(-horizon) / close)
        return forward_returns

    def _calculate_ic(
        self,
        factor: pd.Series,
        returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate Information Coefficient."""
        # Align data
        factor = factor.dropna()
        returns = returns.dropna()
        common_idx = factor.index.intersection(returns.index)

        if len(common_idx) < 30:
            return {'ic': 0, 'ic_ir': 0, 'ic_tstat': 0}

        factor = factor.loc[common_idx]
        returns = returns.loc[common_idx]

        # Calculate correlation
        if self.ic_method == 'spearman':
            corr, pvalue = stats.spearmanr(factor, returns)
        else:
            corr, pvalue = stats.pearsonr(factor, returns)

        # Calculate rolling IC for IR
        window = min(252, len(factor) // 4)
        if window > 20:
            rolling_ic = factor.rolling(window).corr(returns)
            ic_std = rolling_ic.std()
            ic_ir = corr / ic_std if ic_std > 0 else 0
        else:
            ic_ir = 0

        # T-statistic
        n = len(factor)
        tstat = corr * np.sqrt((n - 2) / (1 - corr**2 + 0.0001))

        return {
            'ic': corr if not np.isnan(corr) else 0,
            'ic_ir': ic_ir if not np.isnan(ic_ir) else 0,
            'ic_tstat': tstat if not np.isnan(tstat) else 0
        }

    def _calculate_ic_stability(
        self,
        factor: pd.Series,
        returns: pd.Series,
        window: int = 252
    ) -> float:
        """Calculate IC stability over time."""
        # Calculate rolling IC
        min_periods = max(20, window // 4)
        rolling_ic = factor.rolling(window, min_periods=min_periods).corr(returns)

        # Stability = fraction of time IC is positive (or negative consistently)
        if len(rolling_ic.dropna()) == 0:
            return 0

        # Calculate how often IC has the same sign as mean IC
        mean_ic = rolling_ic.mean()
        if np.isnan(mean_ic) or mean_ic == 0:
            return 0

        consistency = (np.sign(rolling_ic.dropna()) == np.sign(mean_ic)).mean()
        return consistency

    def _calculate_quantile_sharpe(
        self,
        factor: pd.Series,
        returns: pd.Series
    ) -> float:
        """Calculate Sharpe ratio of quantile strategy."""
        try:
            # Create quantile groups
            quantiles = pd.qcut(factor, self.quantile_bins, labels=False, duplicates='drop')

            # Long top quantile, short bottom quantile
            top_q = quantiles == quantiles.max()
            bottom_q = quantiles == quantiles.min()

            # Calculate returns
            long_returns = returns.loc[top_q]
            short_returns = returns.loc[bottom_q]

            # Strategy returns
            strategy_returns = long_returns.mean() - short_returns.mean()

            # Sharpe (annualized)
            if len(long_returns) > 1:
                std_returns = (long_returns.std() + short_returns.std()) / 2
                if std_returns > 0:
                    sharpe = strategy_returns / std_returns * np.sqrt(252 * 24 * 60)  # For 1-minute data
                    return sharpe

            return 0

        except Exception as e:
            self.logger.debug(f"Error calculating quantile Sharpe: {e}")
            return 0

    def _calculate_turnover(self, factor: pd.Series) -> float:
        """Calculate factor turnover."""
        # Normalize factor to [-1, 1] range
        factor_normalized = (factor - factor.min()) / (factor.max() - factor.min() + 0.0001) * 2 - 1

        # Calculate position changes
        position_changes = factor_normalized.diff().abs()

        # Turnover = average absolute position change
        turnover = position_changes.mean()

        return turnover if not np.isnan(turnover) else 0

    def _calculate_fitness(
        self,
        evaluation: FactorEvaluation,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate overall fitness score."""
        weights = weights or {
            'ic': 0.4,
            'ic_ir': 0.2,
            'sharpe': 0.2,
            'stability': 0.1,
            'turnover_penalty': 0.1
        }

        # IC component (prefer high absolute IC)
        ic_score = abs(evaluation.ic) * np.sign(evaluation.ic)

        # IC IR component
        ic_ir_score = evaluation.ic_ir if not np.isnan(evaluation.ic_ir) else 0

        # Sharpe component
        sharpe_score = evaluation.sharpe if not np.isnan(evaluation.sharpe) else 0

        # Stability component
        stability_score = evaluation.stability

        # Turnover penalty (prefer lower turnover)
        turnover_penalty = -min(evaluation.turnover, 1) * weights['turnover_penalty']

        # Combine
        fitness = (
            weights['ic'] * ic_score +
            weights['ic_ir'] * min(ic_ir_score, 5) / 5 +
            weights['sharpe'] * min(max(sharpe_score, -5), 5) / 5 +
            weights['stability'] * stability_score +
            turnover_penalty
        )

        return fitness

    def _calculate_decay(
        self,
        factor: pd.Series,
        close: pd.Series,
        max_horizon: int = 60
    ) -> List[float]:
        """Calculate alpha decay profile."""
        decay = []

        for h in range(1, min(max_horizon + 1, len(factor) - 1)):
            forward_ret = np.log(close.shift(-h) / close)
            common_idx = factor.index.intersection(forward_ret.dropna().index)

            if len(common_idx) > 30:
                if self.ic_method == 'spearman':
                    corr, _ = stats.spearmanr(
                        factor.loc[common_idx],
                        forward_ret.loc[common_idx]
                    )
                else:
                    corr, _ = stats.pearsonr(
                        factor.loc[common_idx],
                        forward_ret.loc[common_idx]
                    )
                decay.append(corr if not np.isnan(corr) else 0)
            else:
                decay.append(0)

        return decay

    def calculate_factor_correlation(
        self,
        factors_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between factors.

        Args:
            factors_df: DataFrame with factor values

        Returns:
            Correlation matrix
        """
        return factors_df.corr(method=self.ic_method)

    def calculate_factor_autocorrelation(
        self,
        factor: pd.Series,
        max_lag: int = 20
    ) -> pd.Series:
        """
        Calculate factor autocorrelation at different lags.

        Args:
            factor: Factor values
            max_lag: Maximum lag

        Returns:
            Series with autocorrelation values
        """
        autocorr = {}
        for lag in range(1, max_lag + 1):
            autocorr[lag] = factor.autocorr(lag=lag)

        return pd.Series(autocorr)

    def analyze_quantile_returns(
        self,
        factor: pd.Series,
        returns: pd.Series,
        n_quantiles: int = 5
    ) -> pd.DataFrame:
        """
        Analyze returns by factor quantile.

        Args:
            factor: Factor values
            returns: Forward returns
            n_quantiles: Number of quantiles

        Returns:
            DataFrame with quantile analysis
        """
        try:
            quantiles = pd.qcut(factor, n_quantiles, labels=False, duplicates='drop')

            results = []
            for q in range(n_quantiles):
                mask = quantiles == q
                if mask.sum() > 0:
                    q_returns = returns.loc[mask]
                    results.append({
                        'quantile': q + 1,
                        'count': mask.sum(),
                        'mean_return': q_returns.mean(),
                        'std_return': q_returns.std(),
                        'sharpe': q_returns.mean() / q_returns.std() if q_returns.std() > 0 else 0,
                        'win_rate': (q_returns > 0).mean()
                    })

            return pd.DataFrame(results)

        except Exception as e:
            self.logger.warning(f"Error in quantile analysis: {e}")
            return pd.DataFrame()

    def calculate_congestion(
        self,
        factor: pd.Series,
        window: int = 20
    ) -> float:
        """
        Calculate factor congestion (how crowded the signal is).

        Args:
            factor: Factor values
            window: Rolling window

        Returns:
            Congestion score (0 = not crowded, 1 = very crowded)
        """
        # Congestion = 1 - uniqueness of factor values
        # High congestion means many similar values
        rolling_std = factor.rolling(window).std()
        overall_std = factor.std()

        if overall_std == 0:
            return 1.0

        # Low rolling std relative to overall means high congestion
        congestion = 1 - (rolling_std.mean() / overall_std)
        return max(0, min(1, congestion))

    def calculate_monotonicity(
        self,
        factor: pd.Series,
        returns: pd.Series,
        n_groups: int = 10
    ) -> float:
        """
        Calculate monotonicity of returns across factor quantiles.

        Args:
            factor: Factor values
            returns: Forward returns
            n_groups: Number of groups

        Returns:
            Monotonicity score (1 = perfectly monotonic, -1 = reversed, 0 = random)
        """
        try:
            # Group returns by factor quantiles
            quantiles = pd.qcut(factor, n_groups, labels=False, duplicates='drop')
            group_means = []

            for q in range(n_groups):
                mask = quantiles == q
                if mask.sum() > 0:
                    group_means.append(returns.loc[mask].mean())

            # Calculate Spearman correlation of group number vs mean return
            if len(group_means) >= 3:
                corr, _ = stats.spearmanr(range(len(group_means)), group_means)
                return corr if not np.isnan(corr) else 0

            return 0

        except Exception:
            return 0

    def generate_evaluation_report(
        self,
        evaluation: FactorEvaluation,
        factor: pd.Series,
        prices: pd.DataFrame
    ) -> str:
        """
        Generate a detailed evaluation report for a factor.

        Args:
            evaluation: Factor evaluation results
            factor: Factor values
            prices: Price data

        Returns:
            Formatted report string
        """
        report = []
        report.append(f"\n{'='*60}")
        report.append(f"Factor Evaluation Report: {evaluation.name}")
        report.append(f"{'='*60}")
        report.append(f"\n--- Information Coefficient ---")
        report.append(f"IC:           {evaluation.ic:.4f}")
        report.append(f"IC IR:        {evaluation.ic_ir:.4f}")
        report.append(f"IC t-stat:    {evaluation.ic_tstat:.4f}")
        report.append(f"Stability:    {evaluation.stability:.4f}")

        report.append(f"\n--- Performance ---")
        report.append(f"Sharpe:       {evaluation.sharpe:.4f}")
        report.append(f"Turnover:     {evaluation.turnover:.4f}")

        report.append(f"\n--- Overall ---")
        report.append(f"Fitness:      {evaluation.fitness:.4f}")

        # Add decay profile summary
        if evaluation.decay:
            report.append(f"\n--- Alpha Decay ---")
            report.append(f"Peak IC:      {max(evaluation.decay):.4f} at horizon {evaluation.decay.index(max(evaluation.decay)) + 1}")
            report.append(f"Half-life:    {self._estimate_halflife(evaluation.decay)} bars")

        report.append(f"\n{'='*60}")

        return '\n'.join(report)

    def _estimate_halflife(self, decay: List[float]) -> int:
        """Estimate half-life from decay profile."""
        if not decay or decay[0] == 0:
            return 0

        peak = abs(decay[0])
        half_peak = peak / 2

        for i, val in enumerate(decay):
            if abs(val) <= half_peak:
                return i + 1

        return len(decay)
