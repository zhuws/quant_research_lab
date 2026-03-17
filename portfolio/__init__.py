"""
Portfolio Management Module for Quant Research Lab.

Provides portfolio optimization, strategy allocation, and capital management.

Components:
    - PortfolioOptimizer: Mean-variance, risk parity, Sharpe maximization
    - StrategyAllocator: Allocate capital across strategies
    - CapitalAllocator: Capital allocation and position sizing
"""

from portfolio.portfolio_optimizer import PortfolioOptimizer
from portfolio.strategy_allocator import StrategyAllocator
from portfolio.capital_allocator import CapitalAllocator

__all__ = ['PortfolioOptimizer', 'StrategyAllocator', 'CapitalAllocator']
