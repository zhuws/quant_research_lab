"""
Research module for Quant Research Lab.
Alpha discovery, factor evaluation, and regime detection.
"""

from research.factor_library import Factor, FactorLibrary, GeneticFactorGenerator
from research.factor_evaluator import FactorEvaluation, FactorEvaluator
from research.alpha_discovery import AlphaDiscovery, DiscoveryResult, FactorOptimizer, run_alpha_discovery
from research.regime_detection import (
    MarketRegime,
    RegimeResult,
    RegimeDetector,
    AdaptiveRegimeStrategy,
    detect_regime
)

__all__ = [
    # Factor Library
    'Factor',
    'FactorLibrary',
    'GeneticFactorGenerator',

    # Factor Evaluation
    'FactorEvaluation',
    'FactorEvaluator',

    # Alpha Discovery
    'AlphaDiscovery',
    'DiscoveryResult',
    'FactorOptimizer',
    'run_alpha_discovery',

    # Regime Detection
    'MarketRegime',
    'RegimeResult',
    'RegimeDetector',
    'AdaptiveRegimeStrategy',
    'detect_regime'
]
