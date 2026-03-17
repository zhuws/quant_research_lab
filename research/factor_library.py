"""
Factor Library for Quant Research Lab.
Defines and generates alpha factor expressions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.math_utils import winsorize


@dataclass
class Factor:
    """
    Represents a single alpha factor.

    Attributes:
        name: Factor name
        expression: String expression for calculation
        category: Factor category (momentum, volatility, etc.)
        params: Optional parameters
    """
    name: str
    expression: str
    category: str = 'custom'
    params: Optional[Dict] = None

    def __post_init__(self):
        self.params = self.params or {}


class FactorLibrary:
    """
    Library of alpha factors for research.

    Provides:
        - Pre-defined factor templates
        - Factor expression parser
        - Factor generation utilities
        - Factor categorization

    Contains ~100 pre-built factors organized by category.
    """

    def __init__(self):
        """Initialize Factor Library."""
        self.logger = get_logger('factor_library')
        self._factors: Dict[str, Factor] = {}
        self._categories: Dict[str, List[str]] = {}

        # Load default factors
        self._load_default_factors()

    def _load_default_factors(self):
        """Load default factor library."""
        # Momentum factors
        momentum_factors = [
            Factor('mom_1', 'close / close.shift(1) - 1', 'momentum'),
            Factor('mom_5', 'close / close.shift(5) - 1', 'momentum'),
            Factor('mom_10', 'close / close.shift(10) - 1', 'momentum'),
            Factor('mom_20', 'close / close.shift(20) - 1', 'momentum'),
            Factor('roc_5', '(close - close.shift(5)) / close.shift(5) * 100', 'momentum'),
            Factor('roc_10', '(close - close.shift(10)) / close.shift(10) * 100', 'momentum'),
            Factor('roc_20', '(close - close.shift(20)) / close.shift(20) * 100', 'momentum'),
            Factor('roc_60', '(close - close.shift(60)) / close.shift(60) * 100', 'momentum'),
            Factor('delta_close_5', 'close - close.shift(5)', 'momentum'),
            Factor('delta_close_10', 'close - close.shift(10)', 'momentum'),
            Factor('acceleration_5', '(close - close.shift(5)) - (close.shift(5) - close.shift(10))', 'momentum'),
            Factor('momentum_oscillator', 'close / close.rolling(20).mean() - 1', 'momentum'),
            Factor('price_rate_10', '(close - close.rolling(10).min()) / (close.rolling(10).max() - close.rolling(10).min())', 'momentum'),
            Factor('price_rate_20', '(close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min())', 'momentum'),
        ]

        for f in momentum_factors:
            self.add_factor(f)

        # Volume factors
        volume_factors = [
            Factor('volume_ratio_5', 'volume / volume.rolling(5).mean()', 'volume'),
            Factor('volume_ratio_10', 'volume / volume.rolling(10).mean()', 'volume'),
            Factor('volume_ratio_20', 'volume / volume.rolling(20).mean()', 'volume'),
            Factor('volume_change_1', 'volume.pct_change(1)', 'volume'),
            Factor('volume_change_5', 'volume.pct_change(5)', 'volume'),
            Factor('volume_acceleration', 'volume - 2 * volume.shift(1) + volume.shift(2)', 'volume'),
            Factor('volume_momentum', '(volume - volume.shift(5)) / volume.shift(5)', 'volume'),
            Factor('obv_trend', '(np.sign(close.diff()) * volume).cumsum().diff(5)', 'volume'),
            Factor('vwap_ratio', 'close / ((high + low + close) / 3).rolling(20).apply(lambda x: np.average(x, weights=volume.loc[x.index]) if len(x) > 0 else np.nan) - 1', 'volume'),
            Factor('adv_5', 'volume.rolling(5).mean()', 'volume'),
            Factor('adv_20', 'volume.rolling(20).mean()', 'volume'),
            Factor('volume_std_20', 'volume.rolling(20).std() / volume.rolling(20).mean()', 'volume'),
        ]

        for f in volume_factors:
            self.add_factor(f)

        # Volatility factors
        volatility_factors = [
            Factor('volatility_5', 'close.pct_change().rolling(5).std()', 'volatility'),
            Factor('volatility_10', 'close.pct_change().rolling(10).std()', 'volatility'),
            Factor('volatility_20', 'close.pct_change().rolling(20).std()', 'volatility'),
            Factor('volatility_60', 'close.pct_change().rolling(60).std()', 'volatility'),
            Factor('atr_ratio_14', '(high - low).rolling(14).mean() / close', 'volatility'),
            Factor('range_ratio', '(high - low) / close', 'volatility'),
            Factor('range_ratio_ma', '(high - low) / close.rolling(20).mean()', 'volatility'),
            Factor('parkinson_vol', 'np.sqrt((np.log(high / low) ** 2).rolling(20).mean() / (4 * np.log(2)))', 'volatility'),
            Factor('garman_klass', 'np.sqrt((0.5 * np.log(high / low) ** 2 - (2 * np.log(2) - 1) * np.log(close / open) ** 2).rolling(20).mean())', 'volatility'),
            Factor('vol_of_vol', 'close.pct_change().rolling(5).std().rolling(20).std()', 'volatility'),
            Factor('vol_ratio', 'close.pct_change().rolling(5).std() / close.pct_change().rolling(20).std()', 'volatility'),
            Factor('upper_shadow', '(high - np.maximum(close, open)) / (high - low + 0.0001)', 'volatility'),
            Factor('lower_shadow', '(np.minimum(close, open) - low) / (high - low + 0.0001)', 'volatility'),
        ]

        for f in volatility_factors:
            self.add_factor(f)

        # Mean reversion factors
        mean_reversion_factors = [
            Factor('zscore_20', '(close - close.rolling(20).mean()) / close.rolling(20).std()', 'mean_reversion'),
            Factor('zscore_60', '(close - close.rolling(60).mean()) / close.rolling(60).std()', 'mean_reversion'),
            Factor('bb_position', '(close - close.rolling(20).mean() + 2 * close.rolling(20).std()) / (4 * close.rolling(20).std())', 'mean_reversion'),
            Factor('price_deviation', '(close - close.rolling(20).mean()) / close.rolling(20).mean()', 'mean_reversion'),
            Factor('rsi_14', '100 - 100 / (1 + close.diff().clip(lower=0).rolling(14).mean() / (-close.diff().clip(upper=0)).rolling(14).mean())', 'mean_reversion'),
            Factor('rsi_7', '100 - 100 / (1 + close.diff().clip(lower=0).rolling(7).mean() / (-close.diff().clip(upper=0)).rolling(7).mean())', 'mean_reversion'),
            Factor('stoch_k', '(close - low.rolling(14).min()) / (high.rolling(14).max() - low.rolling(14).min()) * 100', 'mean_reversion'),
            Factor('stoch_d', '(close - low.rolling(14).min()) / (high.rolling(14).max() - low.rolling(14).min()).rolling(3).mean() * 100', 'mean_reversion'),
            Factor('cci_20', '((high + low + close) / 3 - (high + low + close).rolling(20).mean()) / (0.015 * (high + low + close).rolling(20).apply(lambda x: np.abs(x - x.mean()).mean()))', 'mean_reversion'),
            Factor('williams_r', '(high.rolling(14).max() - close) / (high.rolling(14).max() - low.rolling(14).min()) * -100', 'mean_reversion'),
        ]

        for f in mean_reversion_factors:
            self.add_factor(f)

        # Trend factors
        trend_factors = [
            Factor('price_above_sma_20', '(close > close.rolling(20).mean()).astype(int)', 'trend'),
            Factor('price_above_sma_60', '(close > close.rolling(60).mean()).astype(int)', 'trend'),
            Factor('sma_cross_5_20', 'close.rolling(5).mean() / close.rolling(20).mean() - 1', 'trend'),
            Factor('sma_cross_10_60', 'close.rolling(10).mean() / close.rolling(60).mean() - 1', 'trend'),
            Factor('ema_cross_5_20', 'close.ewm(span=5).mean() / close.ewm(span=20).mean() - 1', 'trend'),
            Factor('macd', 'close.ewm(span=12).mean() - close.ewm(span=26).mean()', 'trend'),
            Factor('macd_signal', '(close.ewm(span=12).mean() - close.ewm(span=26).mean()).ewm(span=9).mean()', 'trend'),
            Factor('macd_hist', '(close.ewm(span=12).mean() - close.ewm(span=26).mean()) - (close.ewm(span=12).mean() - close.ewm(span=26).mean()).ewm(span=9).mean()', 'trend'),
            Factor('adx', 'self._calculate_adx(close, high, low, 14)', 'trend'),
            Factor('di_plus', 'self._calculate_di_plus(high, low, close, 14)', 'trend'),
            Factor('di_minus', 'self._calculate_di_minus(high, low, close, 14)', 'trend'),
        ]

        for f in trend_factors:
            self.add_factor(f)

        # Liquidity factors
        liquidity_factors = [
            Factor('amihud_illiq', 'abs(close.pct_change()) / (volume + 1).rolling(20).mean()', 'liquidity'),
            Factor('kyle_lambda', 'abs(close.diff()) / np.sqrt(volume).rolling(20).mean()', 'liquidity'),
            Factor('spread_proxy', '(high - low) / (np.sqrt(volume) + 1)', 'liquidity'),
            Factor('roll_spread', 'np.sqrt(abs(close.diff().cov(close.diff().shift(1))))', 'liquidity'),
            Factor('volume_sync', 'abs(close.pct_change()) / volume.rolling(20).mean()', 'liquidity'),
        ]

        for f in liquidity_factors:
            self.add_factor(f)

        # Price pattern factors
        pattern_factors = [
            Factor('doji', '((abs(close - open) / (high - low + 0.0001)) < 0.1).astype(int)', 'pattern'),
            Factor('hammer', '(((np.minimum(close, open) - low) > 2 * abs(close - open)) & ((high - np.maximum(close, open)) < 0.5 * abs(close - open))).astype(int)', 'pattern'),
            Factor('engulfing_bull', '((close > open) & (close.shift(1) < open.shift(1)) & (open < close.shift(1)) & (close > open.shift(1))).astype(int)', 'pattern'),
            Factor('engulfing_bear', '((close < open) & (close.shift(1) > open.shift(1)) & (open > close.shift(1)) & (close < open.shift(1))).astype(int)', 'pattern'),
            Factor('inside_bar', '((high < high.shift(1)) & (low > low.shift(1))).astype(int)', 'pattern'),
            Factor('outside_bar', '((high > high.shift(1)) & (low < low.shift(1))).astype(int)', 'pattern'),
            Factor('gap_up', '((open > high.shift(1)) * (open - high.shift(1)) / close.shift(1))', 'pattern'),
            Factor('gap_down', '((open < low.shift(1)) * (low.shift(1) - open) / close.shift(1))', 'pattern'),
        ]

        for f in pattern_factors:
            self.add_factor(f)

        # Microstructure factors (require orderbook/trade data)
        microstructure_factors = [
            Factor('trade_intensity', 'volume / volume.rolling(5).mean()', 'microstructure'),
            Factor('order_imbalance', '(close - open) / (high - low + 0.0001)', 'microstructure'),
            Factor('effective_spread', '2 * abs(close - (high + low) / 2) / close', 'microstructure'),
            Factor('realized_spread', 'abs(close.shift(-1) - close) / close', 'microstructure'),
        ]

        for f in microstructure_factors:
            self.add_factor(f)

        # Cross-sectional factors
        cross_sectional_factors = [
            Factor('rank_volume', 'volume.rank(pct=True)', 'cross_sectional'),
            Factor('rank_return_5', 'close.pct_change(5).rank(pct=True)', 'cross_sectional'),
            Factor('rank_volatility', 'close.pct_change().rolling(20).std().rank(pct=True)', 'cross_sectional'),
        ]

        for f in cross_sectional_factors:
            self.add_factor(f)

        self.logger.info(f"Loaded {len(self._factors)} default factors in {len(self._categories)} categories")

    def add_factor(self, factor: Factor) -> None:
        """
        Add a factor to the library.

        Args:
            factor: Factor to add
        """
        self._factors[factor.name] = factor

        if factor.category not in self._categories:
            self._categories[factor.category] = []
        self._categories[factor.category].append(factor.name)

    def get_factor(self, name: str) -> Optional[Factor]:
        """
        Get a factor by name.

        Args:
            name: Factor name

        Returns:
            Factor if found, None otherwise
        """
        return self._factors.get(name)

    def get_factors_by_category(self, category: str) -> List[Factor]:
        """
        Get all factors in a category.

        Args:
            category: Category name

        Returns:
            List of factors in category
        """
        factor_names = self._categories.get(category, [])
        return [self._factors[name] for name in factor_names]

    def get_all_factors(self) -> List[Factor]:
        """Get all factors in the library."""
        return list(self._factors.values())

    def get_categories(self) -> List[str]:
        """Get all factor categories."""
        return list(self._categories.keys())

    def calculate_factor(
        self,
        df: pd.DataFrame,
        factor: Union[str, Factor],
        winsorize_factor: bool = True,
        winsorize_bounds: Tuple[float, float] = (0.01, 0.99)
    ) -> pd.Series:
        """
        Calculate a factor value from DataFrame.

        Args:
            df: DataFrame with OHLCV data
            factor: Factor name or Factor object
            winsorize_factor: Whether to winsorize the result
            winsorize_bounds: Winsorization bounds

        Returns:
            Series with factor values
        """
        if isinstance(factor, str):
            factor = self.get_factor(factor)
            if factor is None:
                raise ValueError(f"Factor not found: {factor}")

        try:
            # Create local namespace for evaluation
            local_vars = {
                'df': df,
                'np': np,
                'pd': pd,
                'close': df['close'],
                'open': df['open'],
                'high': df['high'],
                'low': df['low'],
                'volume': df['volume'],
                'self': self,
            }

            # Evaluate expression
            result = eval(factor.expression, {"__builtins__": {}}, local_vars)

            if isinstance(result, pd.Series):
                values = result
            else:
                values = pd.Series(result, index=df.index)

            # Winsorize if requested
            if winsorize_factor:
                values = winsorize(values, limits=winsorize_bounds)

            return values

        except Exception as e:
            self.logger.error(f"Error calculating factor {factor.name}: {e}")
            return pd.Series(index=df.index, dtype=float)

    def calculate_all_factors(
        self,
        df: pd.DataFrame,
        categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate all factors (or specified categories).

        Args:
            df: DataFrame with OHLCV data
            categories: Optional list of categories to include

        Returns:
            DataFrame with all factor values
        """
        if categories is None:
            factors = self.get_all_factors()
        else:
            factors = []
            for cat in categories:
                factors.extend(self.get_factors_by_category(cat))

        results = pd.DataFrame(index=df.index)

        for factor in factors:
            try:
                results[factor.name] = self.calculate_factor(df, factor)
            except Exception as e:
                self.logger.warning(f"Skipping factor {factor.name}: {e}")

        # Remove completely empty columns
        results = results.dropna(axis=1, how='all')

        self.logger.info(f"Calculated {len(results.columns)} factors")

        return results

    def generate_factor_combinations(
        self,
        base_factors: List[str],
        operations: List[str] = ['add', 'sub', 'mul', 'div'],
        max_combinations: int = 100
    ) -> List[Factor]:
        """
        Generate new factors by combining existing ones.

        Args:
            base_factors: List of base factor names
            operations: Operations to use for combination
            max_combinations: Maximum number of combinations to generate

        Returns:
            List of new combined factors
        """
        op_symbols = {
            'add': '+',
            'sub': '-',
            'mul': '*',
            'div': '/'
        }

        new_factors = []
        count = 0

        for i, f1_name in enumerate(base_factors):
            f1 = self.get_factor(f1_name)
            if f1 is None:
                continue

            for f2_name in base_factors[i+1:]:
                f2 = self.get_factor(f2_name)
                if f2 is None:
                    continue

                for op in operations:
                    if count >= max_combinations:
                        break

                    new_name = f"{f1_name}_{op}_{f2_name}"
                    new_expr = f"({f1.expression}) {op_symbols[op]} ({f2.expression})"
                    new_factor = Factor(
                        name=new_name,
                        expression=new_expr,
                        category='combined'
                    )
                    new_factors.append(new_factor)
                    count += 1

        self.logger.info(f"Generated {len(new_factors)} combined factors")
        return new_factors

    def generate_time_transforms(
        self,
        base_factor: str,
        windows: List[int] = [5, 10, 20]
    ) -> List[Factor]:
        """
        Generate time-transformed factors (rolling mean, std, etc.).

        Args:
            base_factor: Base factor name
            windows: Rolling window sizes

        Returns:
            List of transformed factors
        """
        f = self.get_factor(base_factor)
        if f is None:
            return []

        new_factors = []

        for window in windows:
            # Rolling mean
            mean_name = f"{base_factor}_mean_{window}"
            mean_expr = f"({f.expression}).rolling({window}).mean()"
            new_factors.append(Factor(mean_name, mean_expr, 'transformed'))

            # Rolling std
            std_name = f"{base_factor}_std_{window}"
            std_expr = f"({f.expression}).rolling({window}).std()"
            new_factors.append(Factor(std_name, std_expr, 'transformed'))

            # Rolling zscore
            zscore_name = f"{base_factor}_zscore_{window}"
            zscore_expr = f"(({f.expression}) - ({f.expression}).rolling({window}).mean()) / ({f.expression}).rolling({window}).std()"
            new_factors.append(Factor(zscore_name, zscore_expr, 'transformed'))

            # Rolling rank
            rank_name = f"{base_factor}_rank_{window}"
            rank_expr = f"({f.expression}).rolling({window}).rank(pct=True)"
            new_factors.append(Factor(rank_name, rank_expr, 'transformed'))

        self.logger.info(f"Generated {len(new_factors)} time-transformed factors from {base_factor}")
        return new_factors

    def _calculate_adx(self, close: pd.Series, high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        tr = np.maximum(
            high - low,
            np.maximum(
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            )
        )

        plus_dm = np.where(
            (high - high.shift(1)) > (low.shift(1) - low),
            np.maximum(high - high.shift(1), 0),
            0
        )
        minus_dm = np.where(
            (low.shift(1) - low) > (high - high.shift(1)),
            np.maximum(low.shift(1) - low, 0),
            0
        )

        atr = pd.Series(tr).rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.rolling(period).mean()

        return adx

    def _calculate_di_plus(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Plus Directional Indicator."""
        tr = np.maximum(
            high - low,
            np.maximum(
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            )
        )

        plus_dm = np.where(
            (high - high.shift(1)) > (low.shift(1) - low),
            np.maximum(high - high.shift(1), 0),
            0
        )

        atr = pd.Series(tr).rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr

        return plus_di

    def _calculate_di_minus(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Minus Directional Indicator."""
        tr = np.maximum(
            high - low,
            np.maximum(
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            )
        )

        minus_dm = np.where(
            (low.shift(1) - low) > (high - high.shift(1)),
            np.maximum(low.shift(1) - low, 0),
            0
        )

        atr = pd.Series(tr).rolling(period).mean()
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

        return minus_di


class GeneticFactorGenerator:
    """
    Generate new factors using genetic programming.

    Uses evolution to discover novel alpha expressions.
    """

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7
    ):
        """
        Initialize Genetic Factor Generator.

        Args:
            population_size: Number of individuals in population
            generations: Number of evolution generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        self.logger = get_logger('genetic_factor_generator')
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Define operators and terminals
        self.operators = ['+', '-', '*', '/', 'rank', 'ts_mean', 'ts_std', 'ts_max', 'ts_min']
        self.terminals = ['close', 'open', 'high', 'low', 'volume']
        self.windows = [5, 10, 20, 60]

    def generate_random_expression(self, max_depth: int = 3) -> str:
        """
        Generate a random factor expression.

        Args:
            max_depth: Maximum expression depth

        Returns:
            Random factor expression string
        """
        if max_depth == 0:
            # Return a terminal
            terminal = np.random.choice(self.terminals)
            return terminal

        # Decide: operator or terminal
        if np.random.random() < 0.3:
            return np.random.choice(self.terminals)

        # Choose an operator
        operator = np.random.choice(self.operators)

        if operator in ['+', '-', '*', '/']:
            left = self.generate_random_expression(max_depth - 1)
            right = self.generate_random_expression(max_depth - 1)
            return f"({left} {operator} {right})"
        else:
            # Unary operator with window
            operand = self.generate_random_expression(max_depth - 1)
            window = np.random.choice(self.windows)

            if operator == 'rank':
                return f"({operand}).rank(pct=True)"
            elif operator == 'ts_mean':
                return f"({operand}).rolling({window}).mean()"
            elif operator == 'ts_std':
                return f"({operand}).rolling({window}).std()"
            elif operator == 'ts_max':
                return f"({operand}).rolling({window}).max()"
            elif operator == 'ts_min':
                return f"({operand}).rolling({window}).min()"

        return np.random.choice(self.terminals)

    def generate_population(self, size: int) -> List[Factor]:
        """
        Generate a population of random factors.

        Args:
            size: Population size

        Returns:
            List of random Factor objects
        """
        population = []
        for i in range(size):
            expr = self.generate_random_expression()
            factor = Factor(
                name=f"genetic_{i}",
                expression=expr,
                category='genetic'
            )
            population.append(factor)

        return population
