"""
Vectorized Backtest Engine for Quant Research Lab.

Provides fast vectorized backtesting capabilities:
    - Efficient pandas/numpy operations
    - Quick strategy evaluation
    - Performance metrics calculation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from decimal import Decimal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.math_utils import safe_divide


class VectorizedBacktestEngine:
    """
    Vectorized Backtest Engine.

    Provides fast backtesting using vectorized operations.

    Attributes:
        initial_capital: Starting capital
        fee: Transaction fee rate
        logger: Logger instance
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        fee: float = 0.001
    ):
        """
        Initialize Vectorized Backtest Engine.

        Args:
            initial_capital: Starting capital (default: 100000)
            fee: Transaction fee rate (default: 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.fee = fee
        self.logger = get_logger('vectorized_backtest')

    def run(
        self,
        data: pd.DataFrame,
        strategy,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run backtest with given data and strategy.

        Args:
            data: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume
            strategy: Strategy instance with generate_signals method
            **kwargs: Additional parameters

        Returns:
            Dictionary with backtest results including 'summary' key
        """
        if data.empty:
            return {
                'summary': 'No data provided',
                'trades': [],
                'equity_curve': pd.DataFrame(),
                'metrics': {}
            }

        # Ensure timestamp column
        if 'timestamp' not in data.columns:
            if data.index.name == 'timestamp' or isinstance(data.index, pd.DatetimeIndex):
                data = data.reset_index()
                data.columns = ['timestamp'] + list(data.columns[1:])
            else:
                data['timestamp'] = pd.date_range(start='2024-01-01', periods=len(data), freq='1min')

        # Make a copy to avoid modifying original
        df = data.copy()

        # Convert Decimal types to float
        df = self._convert_decimals(df)

        # Generate signals
        try:
            signals = strategy.generate_signals(df)
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return {
                'summary': f'Error: {str(e)}',
                'trades': [],
                'equity_curve': pd.DataFrame(),
                'metrics': {}
            }

        if not signals:
            # Try to get signals from dataframe if strategy modifies it
            if hasattr(strategy, '_signals_df') and strategy._signals_df is not None:
                df = strategy._signals_df
            else:
                # No signals, return simple results
                return {
                    'summary': 'No trading signals generated',
                    'trades': [],
                    'equity_curve': pd.DataFrame({'equity': [self.initial_capital]}),
                    'metrics': {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
                }

        # Process signals into positions
        df = self._process_signals(df, signals if signals else [])

        # Calculate equity curve
        equity_curve = self._calculate_equity_curve(df)

        # Calculate metrics
        metrics = self._calculate_metrics(equity_curve)

        # Generate summary
        summary = self._generate_summary(metrics)

        return {
            'summary': summary,
            'trades': df[df['trade_type'].notna()].to_dict('records') if 'trade_type' in df.columns else [],
            'equity_curve': equity_curve,
            'metrics': metrics
        }

    def _convert_decimals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Decimal types to float for pandas operations.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with Decimal columns converted to float
        """
        for col in df.columns:
            if df[col].dtype == object:
                # Check if any value is a Decimal
                if df[col].apply(lambda x: isinstance(x, Decimal)).any():
                    df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
            # Also handle numeric columns that might have Decimal
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception:
                pass
        return df

    def _process_signals(
        self,
        df: pd.DataFrame,
        signals: List
    ) -> pd.DataFrame:
        """
        Process signals into position states.

        Args:
            df: OHLCV DataFrame
            signals: List of Signal objects

        Returns:
            DataFrame with position information
        """
        df = df.copy()

        # Initialize position columns
        df['position'] = 0
        df['trade_type'] = None

        # Process each signal
        for signal in signals:
            # Find the timestamp index for this signal
            if hasattr(signal, 'timestamp') and signal.timestamp is not None:
                mask = df['timestamp'] >= signal.timestamp
                if mask.any():
                    idx = mask.idxmax() if hasattr(mask, 'idxmax') else mask.index[0]

                    # Determine position change based on signal type
                    if hasattr(signal, 'signal_type'):
                        signal_type = signal.signal_type
                        if hasattr(signal_type, 'value'):
                            signal_type = signal_type.value

                        if signal_type in ['buy', 'BUY']:
                            df.loc[idx:, 'position'] = 1
                            df.loc[idx, 'trade_type'] = 'buy'
                        elif signal_type in ['sell', 'SELL']:
                            df.loc[idx:, 'position'] = -1
                            df.loc[idx, 'trade_type'] = 'sell'
                        elif signal_type in ['close_long', 'CLOSE_LONG']:
                            df.loc[idx:, 'position'] = 0
                            df.loc[idx, 'trade_type'] = 'close_long'
                        elif signal_type in ['close_short', 'CLOSE_SHORT']:
                            df.loc[idx:, 'position'] = 0
                            df.loc[idx, 'trade_type'] = 'close_short'

        return df

    def _calculate_equity_curve(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate equity curve from positions.

        Args:
            df: DataFrame with position column

        Returns:
            DataFrame with equity values
        """
        if 'close' not in df.columns:
            return pd.DataFrame({'equity': [self.initial_capital]})

        df = df.copy()

        # Calculate returns
        df['returns'] = df['close'].pct_change()

        # Calculate strategy returns (position * returns)
        df['strategy_returns'] = df['position'].shift(1) * df['returns']

        # Apply fees on position changes (fee as percentage of price change)
        df['position_change'] = df['position'].diff().abs()
        # Fee is applied when position changes, as a percentage of notional
        df['fee_cost'] = df['position_change'] * self.fee
        df.loc[df['fee_cost'] > 0, 'strategy_returns'] -= df.loc[df['fee_cost'] > 0, 'fee_cost']

        # Calculate equity curve
        df['equity'] = self.initial_capital * (1 + df['strategy_returns']).cumprod()

        # Fill NaN values
        df['equity'] = df['equity'].fillna(self.initial_capital)

        return df[['equity']]

    def _calculate_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics.

        Args:
            equity_curve: DataFrame with equity column

        Returns:
            Dictionary of metrics
        """
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0
            }

        equity = equity_curve['equity']

        # Total return
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 0 else 0

        # Calculate returns
        returns = equity.pct_change().dropna()

        # Annualized return (assuming minute data, ~525600 minutes per year)
        n_periods = len(equity)
        years = n_periods / 525600 if n_periods > 0 else 1
        annualized_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1 if total_return > -1 else 0

        # Sharpe ratio
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(525600)
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        rolling_max = equity.cummax()
        drawdowns = (equity - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': 0.5,  # Placeholder
            'total_trades': 0  # Placeholder
        }

    def _generate_summary(self, metrics: Dict[str, float]) -> str:
        """
        Generate summary string from metrics.

        Args:
            metrics: Dictionary of metrics

        Returns:
            Summary string
        """
        total_return_pct = metrics.get('total_return', 0) * 100
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd_pct = metrics.get('max_drawdown', 0) * 100

        return (
            f"Total Return: {total_return_pct:.2f}%, "
            f"Sharpe Ratio: {sharpe:.2f}, "
            f"Max Drawdown: {max_dd_pct:.2f}%"
        )
