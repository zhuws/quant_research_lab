"""
Alpha Discovery for Quant Research Lab.
Automated discovery and evaluation of alpha factors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import warnings
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from research.factor_library import FactorLibrary, Factor, GeneticFactorGenerator
from research.factor_evaluator import FactorEvaluator, FactorEvaluation


@dataclass
class DiscoveryResult:
    """
    Results of alpha discovery run.

    Attributes:
        factors: List of discovered factor evaluations
        best_factor: Best factor name
        best_ic: Best IC value
        best_sharpe: Best Sharpe ratio
        timestamp: Discovery timestamp
    """
    factors: List[FactorEvaluation]
    best_factor: str = ''
    best_ic: float = 0.0
    best_sharpe: float = 0.0
    timestamp: str = ''


class AlphaDiscovery:
    """
    Automated alpha discovery system.

    Capabilities:
        - Generate candidate factors from library
        - Evaluate factors using multiple metrics
        - Rank and select best factors
        - Generate novel factors through genetic programming
        - Test factor robustness across time periods
    """

    def __init__(
        self,
        forward_horizon: int = 10,
        ic_threshold: float = 0.01,
        fitness_threshold: float = 0.1,
        max_factors: int = 100
    ):
        """
        Initialize Alpha Discovery.

        Args:
            forward_horizon: Forward return horizon for evaluation
            ic_threshold: Minimum IC to consider a factor
            fitness_threshold: Minimum fitness score
            max_factors: Maximum number of factors to evaluate
        """
        self.logger = get_logger('alpha_discovery')
        self.forward_horizon = forward_horizon
        self.ic_threshold = ic_threshold
        self.fitness_threshold = fitness_threshold
        self.max_factors = max_factors

        # Initialize components
        self.factor_library = FactorLibrary()
        self.evaluator = FactorEvaluator(forward_horizons=[forward_horizon])
        self.genetic_generator = GeneticFactorGenerator()

        # Results storage
        self._discovered_factors: Dict[str, FactorEvaluation] = {}
        self._factor_rankings: pd.DataFrame = pd.DataFrame()

    def run(
        self,
        data: pd.DataFrame,
        n_factors: int = 100,
        categories: Optional[List[str]] = None,
        include_genetic: bool = False,
        genetic_population: int = 50,
        verbose: bool = True
    ) -> DiscoveryResult:
        """
        Run alpha discovery process.

        Args:
            data: DataFrame with OHLCV and features
            n_factors: Number of factors to generate/evaluate
            categories: Factor categories to include
            include_genetic: Whether to include genetic factor generation
            genetic_population: Population size for genetic generation
            verbose: Print progress

        Returns:
            DiscoveryResult with discovered factors
        """
        self.logger.info(f"Starting alpha discovery with {len(data)} bars")

        # Get factors from library
        if categories:
            library_factors = []
            for cat in categories:
                library_factors.extend(self.factor_library.get_factors_by_category(cat))
        else:
            library_factors = self.factor_library.get_all_factors()

        self.logger.info(f"Library contains {len(library_factors)} factors")

        # Calculate library factors
        factor_values = pd.DataFrame(index=data.index)
        successful_factors = []

        for factor in library_factors[:min(len(library_factors), n_factors)]:
            try:
                values = self.factor_library.calculate_factor(data, factor)
                if not values.isna().all():
                    factor_values[factor.name] = values
                    successful_factors.append(factor)
            except Exception as e:
                self.logger.debug(f"Skipping factor {factor.name}: {e}")

        # Generate genetic factors if requested
        if include_genetic:
            genetic_factors = self.genetic_generator.generate_population(genetic_population)
            for factor in genetic_factors:
                try:
                    values = self.factor_library.calculate_factor(data, factor)
                    if not values.isna().all():
                        factor_values[factor.name] = values
                        successful_factors.append(factor)
                except Exception as e:
                    self.logger.debug(f"Skipping genetic factor: {e}")

        self.logger.info(f"Calculated {len(factor_values.columns)} factor values")

        # Evaluate all factors
        evaluations = []
        for col in factor_values.columns:
            try:
                eval_result = self.evaluator.evaluate_factor(
                    factor_values[col],
                    data,
                    horizon=self.forward_horizon,
                    include_decay=True
                )

                # Filter by thresholds
                if abs(eval_result.ic) >= self.ic_threshold and eval_result.fitness >= self.fitness_threshold:
                    evaluations.append(eval_result)
                    self._discovered_factors[col] = eval_result

            except Exception as e:
                self.logger.debug(f"Error evaluating {col}: {e}")

        # Sort by fitness
        evaluations.sort(key=lambda x: x.fitness, reverse=True)

        # Create rankings DataFrame
        self._factor_rankings = pd.DataFrame([
            {
                'factor': e.name,
                'ic': e.ic,
                'ic_ir': e.ic_ir,
                'sharpe': e.sharpe,
                'turnover': e.turnover,
                'stability': e.stability,
                'fitness': e.fitness
            }
            for e in evaluations
        ])

        # Get best factor
        best = evaluations[0] if evaluations else None

        result = DiscoveryResult(
            factors=evaluations,
            best_factor=best.name if best else '',
            best_ic=best.ic if best else 0,
            best_sharpe=best.sharpe if best else 0,
            timestamp=datetime.now().isoformat()
        )

        if verbose:
            self._print_summary(result)

        self.logger.info(f"Discovered {len(evaluations)} significant factors")

        return result

    def discover_combinations(
        self,
        data: pd.DataFrame,
        base_factors: List[str],
        max_combinations: int = 50
    ) -> DiscoveryResult:
        """
        Discover new factors by combining existing ones.

        Args:
            data: DataFrame with OHLCV data
            base_factors: List of base factor names to combine
            max_combinations: Maximum combinations to generate

        Returns:
            DiscoveryResult with combined factors
        """
        self.logger.info(f"Discovering factor combinations from {len(base_factors)} base factors")

        # Generate combinations
        combined_factors = self.factor_library.generate_factor_combinations(
            base_factors,
            max_combinations=max_combinations
        )

        # Calculate and evaluate
        evaluations = []
        for factor in combined_factors:
            try:
                values = self.factor_library.calculate_factor(data, factor)
                if values.isna().all():
                    continue

                eval_result = self.evaluator.evaluate_factor(
                    values,
                    data,
                    horizon=self.forward_horizon
                )

                if abs(eval_result.ic) >= self.ic_threshold:
                    evaluations.append(eval_result)

            except Exception as e:
                self.logger.debug(f"Error with combination {factor.name}: {e}")

        evaluations.sort(key=lambda x: x.fitness, reverse=True)

        return DiscoveryResult(
            factors=evaluations,
            best_factor=evaluations[0].name if evaluations else '',
            best_ic=evaluations[0].ic if evaluations else 0,
            timestamp=datetime.now().isoformat()
        )

    def test_robustness(
        self,
        data: pd.DataFrame,
        factor_name: str,
        n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Test factor robustness across time periods.

        Args:
            data: DataFrame with OHLCV data
            factor_name: Factor to test
            n_splits: Number of time splits

        Returns:
            Dictionary with robustness metrics
        """
        self.logger.info(f"Testing robustness of {factor_name}")

        factor = self.factor_library.get_factor(factor_name)
        if factor is None:
            raise ValueError(f"Factor not found: {factor_name}")

        values = self.factor_library.calculate_factor(data, factor)

        # Split data into periods
        split_size = len(data) // n_splits
        ics = []
        sharp = []

        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size if i < n_splits - 1 else len(data)

            split_data = data.iloc[start_idx:end_idx]
            split_values = values.iloc[start_idx:end_idx]

            eval_result = self.evaluator.evaluate_factor(
                split_values,
                split_data,
                horizon=self.forward_horizon,
                include_decay=False
            )

            ics.append(eval_result.ic)
            sharp.append(eval_result.sharpe)

        # Calculate robustness metrics
        robustness = {
            'ic_mean': np.mean(ics),
            'ic_std': np.std(ics),
            'ic_min': np.min(ics),
            'ic_max': np.max(ics),
            'ic_positive_rate': sum(1 for ic in ics if ic > 0) / len(ics),
            'sharpe_mean': np.mean(sharp),
            'sharpe_std': np.std(sharp),
            'robustness_score': np.mean(ics) / (np.std(ics) + 0.0001)
        }

        return robustness

    def analyze_decay(
        self,
        data: pd.DataFrame,
        factor_name: str,
        max_horizon: int = 60
    ) -> Dict[str, Union[List[float], int]]:
        """
        Analyze alpha decay for a factor.

        Args:
            data: DataFrame with OHLCV data
            factor_name: Factor to analyze
            max_horizon: Maximum horizon to analyze

        Returns:
            Dictionary with decay profile and metrics
        """
        factor = self.factor_library.get_factor(factor_name)
        if factor is None:
            raise ValueError(f"Factor not found: {factor_name}")

        values = self.factor_library.calculate_factor(data, factor)

        eval_result = self.evaluator.evaluate_factor(
            values,
            data,
            horizon=self.forward_horizon,
            include_decay=True
        )

        decay = eval_result.decay or []

        # Find optimal holding period
        optimal_horizon = decay.index(max(decay)) + 1 if decay else self.forward_horizon

        return {
            'decay_profile': decay,
            'optimal_horizon': optimal_horizon,
            'peak_ic': max(decay) if decay else 0,
            'half_life': self._estimate_halflife(decay)
        }

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

    def get_top_factors(self, n: int = 10) -> List[FactorEvaluation]:
        """
        Get top N discovered factors.

        Args:
            n: Number of factors to return

        Returns:
            List of top factor evaluations
        """
        sorted_factors = sorted(
            self._discovered_factors.values(),
            key=lambda x: x.fitness,
            reverse=True
        )
        return sorted_factors[:n]

    def get_factor_rankings(self) -> pd.DataFrame:
        """Get DataFrame with all factor rankings."""
        return self._factor_rankings.copy()

    def save_discovery_results(self, filepath: str) -> None:
        """
        Save discovery results to file.

        Args:
            filepath: Output file path
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'forward_horizon': self.forward_horizon,
            'rankings': self._factor_rankings.to_dict('records'),
            'top_factors': [
                {
                    'name': f.name,
                    'ic': f.ic,
                    'ic_ir': f.ic_ir,
                    'sharpe': f.sharpe,
                    'fitness': f.fitness
                }
                for f in self.get_top_factors(50)
            ]
        }

        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Saved discovery results to {filepath}")

    def load_discovery_results(self, filepath: str) -> None:
        """
        Load discovery results from file.

        Args:
            filepath: Input file path
        """
        import json
        with open(filepath, 'r') as f:
            results = json.load(f)

        self._factor_rankings = pd.DataFrame(results['rankings'])

        for factor_data in results['top_factors']:
            eval_result = FactorEvaluation(
                name=factor_data['name'],
                ic=factor_data['ic'],
                ic_ir=factor_data['ic_ir'],
                sharpe=factor_data['sharpe'],
                fitness=factor_data['fitness']
            )
            self._discovered_factors[factor_data['name']] = eval_result

        self.logger.info(f"Loaded {len(self._discovered_factors)} factors from {filepath}")

    def _print_summary(self, result: DiscoveryResult) -> None:
        """Print discovery summary."""
        print("\n" + "="*70)
        print("ALPHA DISCOVERY SUMMARY")
        print("="*70)
        print(f"Forward Horizon: {self.forward_horizon} bars")
        print(f"Factors Evaluated: {len(result.factors)}")
        print(f"\nBest Factor: {result.best_factor}")
        print(f"  IC: {result.best_ic:.4f}")
        print(f"  Sharpe: {result.best_sharpe:.4f}")

        print("\n--- Top 10 Factors ---")
        print(f"{'Factor':<30} {'IC':>8} {'IC IR':>8} {'Sharpe':>8} {'Fitness':>8}")
        print("-"*70)

        for f in result.factors[:10]:
            print(f"{f.name:<30} {f.ic:>8.4f} {f.ic_ir:>8.4f} {f.sharpe:>8.4f} {f.fitness:>8.4f}")

        print("="*70 + "\n")


class FactorOptimizer:
    """
    Optimize factor parameters for better performance.
    """

    def __init__(
        self,
        discovery: AlphaDiscovery,
        param_ranges: Optional[Dict[str, List]] = None
    ):
        """
        Initialize Factor Optimizer.

        Args:
            discovery: AlphaDiscovery instance
            param_ranges: Parameter ranges to search
        """
        self.logger = get_logger('factor_optimizer')
        self.discovery = discovery
        self.param_ranges = param_ranges or {
            'window': [5, 10, 20, 30, 60],
            'threshold': [0.01, 0.02, 0.05, 0.1]
        }

    def optimize_factor(
        self,
        data: pd.DataFrame,
        base_expression: str,
        param_name: str,
        param_values: Optional[List] = None
    ) -> Tuple[str, float]:
        """
        Optimize a single parameter for a factor.

        Args:
            data: DataFrame with OHLCV data
            base_expression: Factor expression with {param} placeholder
            param_name: Parameter to optimize
            param_values: Values to test

        Returns:
            Tuple of (best expression, best IC)
        """
        param_values = param_values or self.param_ranges.get(param_name, [10, 20])
        best_ic = 0
        best_expr = base_expression

        for val in param_values:
            try:
                expr = base_expression.replace(f'{{{param_name}}}', str(val))
                factor = Factor(f'optimized_{param_name}_{val}', expr, 'optimized')

                values = self.discovery.factor_library.calculate_factor(data, factor)
                eval_result = self.discovery.evaluator.evaluate_factor(
                    values, data,
                    horizon=self.discovery.forward_horizon
                )

                if abs(eval_result.ic) > abs(best_ic):
                    best_ic = eval_result.ic
                    best_expr = expr

            except Exception as e:
                self.logger.debug(f"Error with {param_name}={val}: {e}")

        return best_expr, best_ic

    def grid_search(
        self,
        data: pd.DataFrame,
        base_expression: str,
        params: Dict[str, List]
    ) -> pd.DataFrame:
        """
        Grid search over multiple parameters.

        Args:
            data: DataFrame with OHLCV data
            base_expression: Factor expression with {param} placeholders
            params: Dictionary of parameter names to values

        Returns:
            DataFrame with results sorted by IC
        """
        from itertools import product

        results = []
        param_names = list(params.keys())
        param_combinations = list(product(*params.values()))

        for combo in param_combinations:
            expr = base_expression
            param_dict = {}

            for name, val in zip(param_names, combo):
                expr = expr.replace(f'{{{name}}}', str(val))
                param_dict[name] = val

            try:
                factor = Factor(f'grid_{combo}', expr, 'optimized')
                values = self.discovery.factor_library.calculate_factor(data, factor)
                eval_result = self.discovery.evaluator.evaluate_factor(
                    values, data,
                    horizon=self.discovery.forward_horizon
                )

                results.append({
                    **param_dict,
                    'ic': eval_result.ic,
                    'sharpe': eval_result.sharpe,
                    'fitness': eval_result.fitness
                })

            except Exception as e:
                self.logger.debug(f"Error with params {combo}: {e}")

        return pd.DataFrame(results).sort_values('ic', ascending=False)


def run_alpha_discovery(
    data: pd.DataFrame,
    n_factors: int = 100,
    horizon: int = 10,
    categories: Optional[List[str]] = None,
    include_genetic: bool = False
) -> DiscoveryResult:
    """
    Convenience function to run alpha discovery.

    Args:
        data: DataFrame with OHLCV data
        n_factors: Number of factors to evaluate
        horizon: Forward return horizon
        categories: Factor categories to include
        include_genetic: Whether to include genetic generation

    Returns:
        DiscoveryResult with discovered factors
    """
    discovery = AlphaDiscovery(forward_horizon=horizon)
    return discovery.run(
        data,
        n_factors=n_factors,
        categories=categories,
        include_genetic=include_genetic
    )
