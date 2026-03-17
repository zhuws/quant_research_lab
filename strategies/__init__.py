"""
Trading Strategies Module for Quant Research Lab.

Provides various trading strategies including:
    - Cross-exchange arbitrage
    - Funding rate arbitrage
    - Triangular arbitrage
    - Statistical arbitrage
"""

from strategies.base_strategy import BaseStrategy, StrategyState, Signal
from strategies.cross_exchange_arbitrage import CrossExchangeArbitrage
from strategies.funding_arbitrage import FundingRateArbitrage
from strategies.momentum_strategy import MomentumStrategy

__all__ = [
    'BaseStrategy',
    'StrategyState',
    'Signal',
    'CrossExchangeArbitrage',
    'FundingRateArbitrage',
    'MomentumStrategy'
]
