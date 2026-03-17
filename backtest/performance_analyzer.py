"""
Performance Analyzer for Quant Research Lab.

Provides comprehensive performance analysis:
    - Return metrics
    - Risk metrics
    - Trade statistics
    - Drawdown analysis
    - Benchmark comparison
    - Performance attribution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import warnings
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.math_utils import safe_divide


@dataclass
class PerformanceReport:
    """
    Comprehensive performance report.

    Contains all metrics and analysis results.
    """
    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    cumulative_return: float = 0.0
    monthly_return: float = 0.0
    daily_return_avg: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    annualized_volatility: float = 0.0
    downside_volatility: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    treynor_ratio: float = 0.0
    information_ratio: float = 0.0
    omega_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    avg_holding_period: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Execution metrics
    total_commission: float = 0.0
    avg_commission: float = 0.0
    total_slippage: float = 0.0
    avg_slippage: float = 0.0
    execution_quality: float = 0.0

    # Benchmark comparison
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    tracking_error: float = 0.0
    excess_return: float = 0.0

    # Time range
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    trading_days: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'return_metrics': {
                'total_return': self.total_return,
                'annualized_return': self.annualized_return,
                'cumulative_return': self.cumulative_return
            },
            'risk_metrics': {
                'volatility': self.volatility,
                'annualized_volatility': self.annualized_volatility,
                'max_drawdown': self.max_drawdown,
                'var_95': self.var_95,
                'cvar_95': self.cvar_95
            },
            'risk_adjusted_metrics': {
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'calmar_ratio': self.calmar_ratio,
                'omega_ratio': self.omega_ratio
            },
            'trade_statistics': {
                'total_trades': self.total_trades,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor,
                'avg_trade': self.avg_trade,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss
            },
            'execution_metrics': {
                'total_commission': self.total_commission,
                'total_slippage': self.total_slippage,
                'execution_quality': self.execution_quality
            },
            'benchmark_comparison': {
                'benchmark_return': self.benchmark_return,
                'alpha': self.alpha,
                'beta': self.beta,
                'tracking_error': self.tracking_error
            }
        }


@dataclass
class DrawdownPeriod:
    """Represents a drawdown period."""
    start_date: datetime
    end_date: Optional[datetime]
    peak_value: float
    trough_value: float
    drawdown: float
    duration: int
    recovery_date: Optional[datetime] = None


class PerformanceAnalyzer:
    """
    Performance Analysis Engine.

    Calculates comprehensive performance metrics from backtest results.

    Features:
        - Return and risk metrics
        - Trade statistics
        - Drawdown analysis
        - Benchmark comparison
        - Rolling metrics
        - Performance attribution

    Attributes:
        risk_free_rate: Annual risk-free rate
        trading_days_per_year: Trading days per year
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        trading_days_per_year: int = 252,
        benchmark_return: float = 0.08
    ):
        """
        Initialize Performance Analyzer.

        Args:
            risk_free_rate: Annual risk-free rate
            trading_days_per_year: Trading days per year
            benchmark_return: Annual benchmark return
        """
        self.logger = get_logger('performance_analyzer')
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.benchmark_return = benchmark_return

        # Daily returns storage
        self._returns: List[float] = []
        self._equity_curve: List[float] = []
        self._timestamps: List[datetime] = []

        # Trade storage
        self._trades: List[Dict] = []

        # Benchmark data
        self._benchmark_returns: List[float] = []

    def add_daily_result(
        self,
        timestamp: datetime,
        equity: float,
        daily_pnl: float,
        daily_return: float
    ) -> None:
        """
        Add a daily result.

        Args:
            timestamp: Date
            equity: End of day equity
            daily_pnl: Daily profit/loss
            daily_return: Daily return (decimal)
        """
        self._timestamps.append(timestamp)
        self._equity_curve.append(equity)
        self._returns.append(daily_return)

    def add_trade(
        self,
        entry_time: datetime,
        exit_time: datetime,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        pnl: float,
        commission: float = 0.0,
        slippage: float = 0.0
    ) -> None:
        """
        Add a completed trade.

        Args:
            entry_time: Trade entry time
            exit_time: Trade exit time
            symbol: Trading symbol
            side: Trade side
            entry_price: Entry price
            exit_price: Exit price
            quantity: Trade quantity
            pnl: Realized P&L
            commission: Commission paid
            slippage: Slippage incurred
        """
        trade = {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'commission': commission,
            'slippage': slippage,
            'holding_period': (exit_time - entry_time).total_seconds() / 3600
        }
        self._trades.append(trade)

    def add_benchmark_return(self, daily_return: float) -> None:
        """Add benchmark daily return."""
        self._benchmark_returns.append(daily_return)

    def analyze(self) -> PerformanceReport:
        """
        Generate comprehensive performance report.

        Returns:
            PerformanceReport with all metrics
        """
        report = PerformanceReport()

        if not self._returns:
            return report

        returns = np.array(self._returns)
        equity_curve = np.array(self._equity_curve)

        # Time range
        if self._timestamps:
            report.start_date = min(self._timestamps)
            report.end_date = max(self._timestamps)
            report.trading_days = len(self._timestamps)

        # Return metrics
        report.total_return = (equity_curve[-1] / equity_curve[0]) - 1 if len(equity_curve) > 0 else 0
        report.cumulative_return = np.prod(1 + returns) - 1
        report.annualized_return = (1 + report.total_return) ** (self.trading_days_per_year / len(returns)) - 1 if len(returns) > 0 else 0
        report.daily_return_avg = np.mean(returns)
        report.monthly_return = report.daily_return_avg * 21  # Approximate

        # Risk metrics
        report.volatility = np.std(returns)
        report.annualized_volatility = report.volatility * np.sqrt(self.trading_days_per_year)

        # Downside volatility
        negative_returns = returns[returns < 0]
        report.downside_volatility = np.std(negative_returns) * np.sqrt(self.trading_days_per_year) if len(negative_returns) > 0 else 0

        # Drawdown analysis
        drawdowns, drawdown_periods = self._calculate_drawdowns(equity_curve)
        report.max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        report.avg_drawdown = np.mean(drawdowns[drawdowns < 0]) if len(drawdowns[drawdowns < 0]) > 0 else 0

        if drawdown_periods:
            report.max_drawdown_duration = max(dp.duration for dp in drawdown_periods)

        # VaR and CVaR
        report.var_95 = np.percentile(returns, 5)
        report.cvar_95 = np.mean(returns[returns <= report.var_95]) if len(returns[returns <= report.var_95]) > 0 else 0

        # Risk-adjusted metrics
        daily_rf = self.risk_free_rate / self.trading_days_per_year
        excess_returns = returns - daily_rf

        report.sharpe_ratio = (
            np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.trading_days_per_year)
            if np.std(excess_returns) > 0 else 0
        )

        report.sortino_ratio = (
            np.mean(excess_returns) / report.downside_volatility
            if report.downside_volatility > 0 else 0
        )

        report.calmar_ratio = (
            report.annualized_return / abs(report.max_drawdown)
            if report.max_drawdown != 0 else 0
        )

        # Omega ratio
        threshold = daily_rf
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]
        report.omega_ratio = np.sum(gains) / np.sum(losses) if np.sum(losses) > 0 else 0

        # Trade statistics
        if self._trades:
            pnls = [t['pnl'] for t in self._trades]
            wins = [p for p in pnls if p > 0]
            losses_list = [p for p in pnls if p < 0]

            report.total_trades = len(self._trades)
            report.winning_trades = len(wins)
            report.losing_trades = len(losses_list)
            report.win_rate = len(wins) / len(pnls) if pnls else 0

            report.avg_trade = np.mean(pnls) if pnls else 0
            report.avg_win = np.mean(wins) if wins else 0
            report.avg_loss = np.mean(losses_list) if losses_list else 0
            report.max_win = max(pnls) if pnls else 0
            report.max_loss = min(pnls) if pnls else 0

            # Profit factor
            total_wins = sum(wins)
            total_losses = abs(sum(losses_list))
            report.profit_factor = total_wins / total_losses if total_losses > 0 else 0

            # Holding period
            holding_periods = [t['holding_period'] for t in self._trades]
            report.avg_holding_period = np.mean(holding_periods) if holding_periods else 0

            # Consecutive wins/losses
            report.max_consecutive_wins = self._count_consecutive(pnls, positive=True)
            report.max_consecutive_losses = self._count_consecutive(pnls, positive=False)

            # Execution metrics
            commissions = [t['commission'] for t in self._trades]
            slippages = [t['slippage'] for t in self._trades]

            report.total_commission = sum(commissions)
            report.avg_commission = np.mean(commissions) if commissions else 0
            report.total_slippage = sum(slippages)
            report.avg_slippage = np.mean(slippages) if slippages else 0

            # Execution quality (lower is better)
            total_execution_cost = report.total_commission + report.total_slippage
            gross_profit = abs(sum(pnls)) + total_execution_cost
            report.execution_quality = 1 - (total_execution_cost / gross_profit) if gross_profit > 0 else 0

        # Benchmark comparison
        if self._benchmark_returns:
            benchmark_returns = np.array(self._benchmark_returns)
            report.benchmark_return = np.prod(1 + benchmark_returns) - 1

            # Alpha and Beta
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)

            report.beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            report.alpha = report.annualized_return - (
                self.risk_free_rate + report.beta * (report.benchmark_return - self.risk_free_rate)
            )

            # Tracking error
            active_returns = returns - benchmark_returns
            report.tracking_error = np.std(active_returns) * np.sqrt(self.trading_days_per_year)

            # Information ratio
            report.information_ratio = (
                np.mean(active_returns) * self.trading_days_per_year / report.tracking_error
                if report.tracking_error > 0 else 0
            )

            # Excess return
            report.excess_return = report.total_return - report.benchmark_return

        # Treynor ratio (if beta is available)
        if report.beta != 0:
            report.treynor_ratio = (report.annualized_return - self.risk_free_rate) / report.beta

        return report

    def _calculate_drawdowns(
        self,
        equity_curve: np.ndarray
    ) -> Tuple[np.ndarray, List[DrawdownPeriod]]:
        """
        Calculate drawdowns from equity curve.

        Args:
            equity_curve: Array of equity values

        Returns:
            Tuple of (drawdown_series, list of drawdown periods)
        """
        if len(equity_curve) == 0:
            return np.array([]), []

        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_curve)

        # Calculate drawdowns
        drawdowns = (equity_curve - running_max) / running_max

        # Identify drawdown periods
        drawdown_periods = []
        in_drawdown = False
        dd_start = None
        dd_peak = 0
        dd_trough = 0
        dd_trough_idx = 0

        for i, (eq, dd) in enumerate(zip(equity_curve, drawdowns)):
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                dd_start = self._timestamps[i] if i < len(self._timestamps) else None
                dd_peak = running_max[i]
                dd_trough = eq
                dd_trough_idx = i
            elif dd < 0 and in_drawdown:
                # Continuing drawdown
                if eq < dd_trough:
                    dd_trough = eq
                    dd_trough_idx = i
            elif dd == 0 and in_drawdown:
                # End of drawdown
                drawdown_periods.append(DrawdownPeriod(
                    start_date=dd_start,
                    end_date=self._timestamps[dd_trough_idx] if dd_trough_idx < len(self._timestamps) else None,
                    peak_value=dd_peak,
                    trough_value=dd_trough,
                    drawdown=(dd_trough - dd_peak) / dd_peak,
                    duration=i - dd_trough_idx,
                    recovery_date=self._timestamps[i] if i < len(self._timestamps) else None
                ))
                in_drawdown = False

        # Handle case where still in drawdown at end
        if in_drawdown:
            drawdown_periods.append(DrawdownPeriod(
                start_date=dd_start,
                end_date=self._timestamps[dd_trough_idx] if dd_trough_idx < len(self._timestamps) else None,
                peak_value=dd_peak,
                trough_value=dd_trough,
                drawdown=(dd_trough - dd_peak) / dd_peak,
                duration=len(equity_curve) - dd_trough_idx,
                recovery_date=None
            ))

        return drawdowns, drawdown_periods

    def _count_consecutive(self, values: List[float], positive: bool = True) -> int:
        """Count maximum consecutive positive/negative values."""
        max_count = 0
        current_count = 0

        for v in values:
            if (positive and v > 0) or (not positive and v < 0):
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def get_rolling_metrics(
        self,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.

        Args:
            window: Rolling window size

        Returns:
            DataFrame with rolling metrics
        """
        if len(self._returns) < window:
            return pd.DataFrame()

        returns = np.array(self._returns)

        rolling_metrics = {
            'rolling_return': pd.Series(returns).rolling(window).mean() * window,
            'rolling_volatility': pd.Series(returns).rolling(window).std() * np.sqrt(self.trading_days_per_year),
            'rolling_sharpe': pd.Series(returns).rolling(window).apply(
                lambda x: np.mean(x) / np.std(x) * np.sqrt(self.trading_days_per_year) if np.std(x) > 0 else 0
            ),
            'rolling_max_drawdown': pd.Series(self._equity_curve).rolling(window).apply(
                lambda x: min((x - np.maximum.accumulate(x)) / np.maximum.accumulate(x))
            )
        }

        df = pd.DataFrame(rolling_metrics)
        if self._timestamps:
            df.index = self._timestamps[:len(df)]

        return df

    def get_monthly_returns(self) -> pd.DataFrame:
        """
        Calculate monthly returns.

        Returns:
            DataFrame with monthly returns
        """
        if not self._timestamps or not self._returns:
            return pd.DataFrame()

        df = pd.DataFrame({
            'date': self._timestamps,
            'return': self._returns
        })
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        monthly = df.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly.columns = ['monthly_return']

        return monthly

    def get_return_distribution(self) -> Dict:
        """
        Get return distribution statistics.

        Returns:
            Dictionary with distribution statistics
        """
        if not self._returns:
            return {}

        returns = np.array(self._returns)

        return {
            'mean': np.mean(returns),
            'median': np.median(returns),
            'std': np.std(returns),
            'skewness': self._skewness(returns),
            'kurtosis': self._kurtosis(returns),
            'min': np.min(returns),
            'max': np.max(returns),
            'percentile_5': np.percentile(returns, 5),
            'percentile_25': np.percentile(returns, 25),
            'percentile_75': np.percentile(returns, 75),
            'percentile_95': np.percentile(returns, 95)
        }

    def _skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness."""
        n = len(returns)
        if n < 3:
            return 0
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0
        return np.sum(((returns - mean) / std) ** 3) / n

    def _kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis."""
        n = len(returns)
        if n < 4:
            return 0
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0
        return np.sum(((returns - mean) / std) ** 4) / n - 3

    def generate_summary_text(self) -> str:
        """
        Generate text summary of performance.

        Returns:
            Formatted text summary
        """
        report = self.analyze()

        summary = f"""
========================================
        BACKTEST PERFORMANCE REPORT
========================================

PERIOD: {report.start_date or 'N/A'} to {report.end_date or 'N/A'}
Trading Days: {report.trading_days}

----------------------------------------
           RETURN METRICS
----------------------------------------
Total Return:          {report.total_return*100:>10.2f}%
Annualized Return:     {report.annualized_return*100:>10.2f}%
Cumulative Return:     {report.cumulative_return*100:>10.2f}%

----------------------------------------
            RISK METRICS
----------------------------------------
Annualized Volatility: {report.annualized_volatility*100:>10.2f}%
Max Drawdown:          {abs(report.max_drawdown)*100:>10.2f}%
VaR (95%):             {report.var_95*100:>10.2f}%
CVaR (95%):            {report.cvar_95*100:>10.2f}%

----------------------------------------
        RISK-ADJUSTED METRICS
----------------------------------------
Sharpe Ratio:          {report.sharpe_ratio:>10.2f}
Sortino Ratio:         {report.sortino_ratio:>10.2f}
Calmar Ratio:          {report.calmar_ratio:>10.2f}
Omega Ratio:           {report.omega_ratio:>10.2f}

----------------------------------------
          TRADE STATISTICS
----------------------------------------
Total Trades:          {report.total_trades:>10d}
Winning Trades:        {report.winning_trades:>10d}
Losing Trades:         {report.losing_trades:>10d}
Win Rate:              {report.win_rate*100:>10.1f}%
Profit Factor:         {report.profit_factor:>10.2f}
Avg Trade:             ${report.avg_trade:>9.2f}
Avg Win:               ${report.avg_win:>9.2f}
Avg Loss:              ${report.avg_loss:>9.2f}
Max Win:               ${report.max_win:>9.2f}
Max Loss:              ${report.max_loss:>9.2f}

----------------------------------------
         EXECUTION METRICS
----------------------------------------
Total Commission:      ${report.total_commission:>9.2f}
Total Slippage:        ${report.total_slippage:>9.2f}
Execution Quality:     {report.execution_quality*100:>10.1f}%
========================================
"""
        return summary

    def reset(self) -> None:
        """Reset analyzer state."""
        self._returns.clear()
        self._equity_curve.clear()
        self._timestamps.clear()
        self._trades.clear()
        self._benchmark_returns.clear()


def analyze_backtest(
    equity_curve: List[float],
    returns: List[float],
    timestamps: List[datetime],
    trades: Optional[List[Dict]] = None
) -> PerformanceReport:
    """
    Convenience function to analyze backtest results.

    Args:
        equity_curve: List of equity values
        returns: List of daily returns
        timestamps: List of timestamps
        trades: List of trade dictionaries

    Returns:
        PerformanceReport
    """
    analyzer = PerformanceAnalyzer()

    for i, (ts, eq, ret) in enumerate(zip(timestamps, equity_curve, returns)):
        daily_pnl = (equity_curve[i] - equity_curve[i-1]) if i > 0 else 0
        analyzer.add_daily_result(ts, eq, daily_pnl, ret)

    if trades:
        for trade in trades:
            analyzer.add_trade(**trade)

    return analyzer.analyze()
