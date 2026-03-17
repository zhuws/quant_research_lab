"""
Features Module for Quant Research Lab.
Provides feature engineering capabilities for alpha research.

Modules:
    - feature_pipeline: Main orchestrator for feature generation
    - technical_features: Technical indicators (momentum, volume, price patterns)
    - volatility_features: Volatility measures (ATR, Bollinger, etc.)
    - orderflow_features: Trade flow and order flow features
    - liquidity_features: Orderbook and liquidity features
    - cross_exchange_features: Cross-exchange arbitrage features
"""

from .feature_pipeline import FeaturePipeline
from .technical_features import TechnicalFeatureGenerator
from .volatility_features import VolatilityFeatureGenerator
from .orderflow_features import OrderFlowFeatureGenerator
from .liquidity_features import LiquidityFeatureGenerator
from .cross_exchange_features import CrossExchangeFeatureGenerator

__all__ = [
    'FeaturePipeline',
    'TechnicalFeatureGenerator',
    'VolatilityFeatureGenerator',
    'OrderFlowFeatureGenerator',
    'LiquidityFeatureGenerator',
    'CrossExchangeFeatureGenerator'
]
