"""
Backtesting Module for Quant Research Lab.

Provides professional backtesting capabilities:
    - Event-driven backtesting engine
    - Realistic execution simulation
    - Slippage and transaction cost modeling
    - Performance analysis and reporting
    - Walk-forward validation

Components:
    - BacktestEngine: Core backtesting engine
    - ExecutionSimulator: Order execution simulation
    - PerformanceAnalyzer: Performance metrics and reporting
    - WalkForwardValidator: Walk-forward validation engine
"""

from backtest.backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from backtest.execution_simulator import ExecutionSimulator, Order, Fill
from backtest.performance_analyzer import PerformanceAnalyzer, PerformanceReport
from backtest.walk_forward import (
    WalkForwardValidator,
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardMode,
    FoldResult,
    run_walk_forward
)
from backtest.vectorized_engine import VectorizedBacktestEngine

__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    'ExecutionSimulator',
    'Order',
    'Fill',
    'PerformanceAnalyzer',
    'PerformanceReport',
    'WalkForwardValidator',
    'WalkForwardConfig',
    'WalkForwardResult',
    'WalkForwardMode',
    'FoldResult',
    'run_walk_forward',
    'VectorizedBacktestEngine'
]
