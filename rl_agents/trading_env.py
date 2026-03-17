"""
Trading Environment for Reinforcement Learning.
Gym-compatible environment for trading simulation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger


class ActionType(Enum):
    """Trading action types."""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE_LONG = 3
    CLOSE_SHORT = 4


@dataclass
class Position:
    """
    Trading position information.

    Attributes:
        size: Position size (positive for long, negative for short)
        entry_price: Entry price
        unrealized_pnl: Unrealized profit/loss
    """
    size: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0

    def is_long(self) -> bool:
        return self.size > 0

    def is_short(self) -> bool:
        return self.size < 0

    def is_flat(self) -> bool:
        return abs(self.size) < 1e-8


@dataclass
class TradingState:
    """
    Trading state information.

    Attributes:
        position: Current position
        balance: Account balance
        equity: Total equity
        margin_used: Margin currently used
        unrealized_pnl: Unrealized P&L
        realized_pnl: Realized P&L
    """
    position: Position
    balance: float
    equity: float
    margin_used: float
    unrealized_pnl: float
    realized_pnl: float


class TradingEnvironment:
    """
    Trading environment for reinforcement learning.

    Compatible with OpenAI Gym interface for use with
    stable-baselines3 and other RL libraries.
    """

    metadata = {'render_modes': ['human', 'console']}

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        target_column: str = 'close',
        initial_balance: float = 100000.0,
        commission_rate: float = 0.0006,
        leverage: float = 10.0,
        max_position_size: float = 1.0,
        lookback_window: int = 20,
        reward_scaling: float = 1.0,
        penalty_factor: float = 0.01,
        render_mode: Optional[str] = None
    ):
        """
        Initialize Trading Environment.

        Args:
            data: DataFrame with OHLCV and features
            feature_columns: Columns to use as observations
            target_column: Price column for trading
            initial_balance: Starting balance
            commission_rate: Trading commission rate
            leverage: Maximum leverage
            max_position_size: Maximum position size (fraction)
            lookback_window: Number of bars for observation
            reward_scaling: Reward scaling factor
            penalty_factor: Penalty for invalid actions
            render_mode: Render mode
        """
        self.logger = get_logger('trading_env')
        self.data = data
        self.target_column = target_column
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.leverage = leverage
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        self.reward_scaling = reward_scaling
        self.penalty_factor = penalty_factor
        self.render_mode = render_mode

        # Determine feature columns
        if feature_columns is None:
            exclude_cols = {'timestamp', target_column}
            self.feature_columns = [
                col for col in data.columns
                if col not in exclude_cols
            ]
        else:
            self.feature_columns = feature_columns

        # Environment state
        self.current_step = 0
        self.position = Position()
        self.balance = initial_balance
        self.realized_pnl = 0.0
        self.trades_history: List[Dict] = []
        self.equity_history: List[float] = []

        # Define spaces
        self._define_spaces()

    def _define_spaces(self) -> None:
        """Define observation and action spaces."""
        # Observation: features + position info
        n_features = len(self.feature_columns)
        n_position_features = 4  # size, entry_price, pnl, margin

        self.observation_dim = (
            self.lookback_window * n_features + n_position_features
        )

        # Action space: continuous position size in [-1, 1]
        # -1: max short, 0: flat, 1: max long
        self.action_dim = 1
        self.action_low = -1.0
        self.action_high = 1.0

    @property
    def prices(self) -> pd.Series:
        """Get price series."""
        return self.data[self.target_column]

    @property
    def current_price(self) -> float:
        """Get current price."""
        return self.prices.iloc[self.current_step]

    @property
    def equity(self) -> float:
        """Calculate current equity."""
        unrealized = self._calculate_unrealized_pnl()
        return self.balance + unrealized

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.

        Args:
            seed: Random seed

        Returns:
            Initial observation and info dict
        """
        if seed is not None:
            np.random.seed(seed)

        self.current_step = self.lookback_window
        self.position = Position()
        self.balance = self.initial_balance
        self.realized_pnl = 0.0
        self.trades_history = []
        self.equity_history = [self.initial_balance]

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: Union[float, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            action: Action to take (position target in [-1, 1])

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Extract action value
        if isinstance(action, np.ndarray):
            action = float(action.flatten()[0])

        # Clip action to valid range
        target_position = np.clip(action, self.action_low, self.action_high)

        # Get current price
        current_price = self.current_price

        # Execute trade
        trade_result = self._execute_trade(target_position, current_price)

        # Move to next step
        self.current_step += 1

        # Check termination
        terminated = self.current_step >= len(self.data) - 1
        truncated = self.equity <= self.initial_balance * 0.1  # Stop loss

        # Calculate reward
        reward = self._calculate_reward(trade_result)

        # Update equity history
        self.equity_history.append(self.equity)

        # Get observation
        observation = self._get_observation() if not terminated else np.zeros(self.observation_dim)
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _execute_trade(
        self,
        target_position: float,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Execute trade to reach target position.

        Args:
            target_position: Target position size (-1 to 1)
            current_price: Current price

        Returns:
            Trade result dict
        """
        # Scale target position by max position size
        target_size = target_position * self.max_position_size

        # Current position
        current_size = self.position.size

        # Calculate trade size
        trade_size = target_size - current_size

        result = {
            'trade_size': 0,
            'trade_price': current_price,
            'commission': 0,
            'slippage': 0,
            'executed': False
        }

        if abs(trade_size) < 1e-8:
            return result

        # Apply slippage
        slippage = abs(trade_size) * current_price * 0.0001
        if trade_size > 0:
            execution_price = current_price + slippage
        else:
            execution_price = current_price - slippage

        # Calculate commission
        trade_value = abs(trade_size) * execution_price * self.leverage
        commission = trade_value * self.commission_rate

        # Update position
        new_size = current_size + trade_size

        # Calculate realized P&L if closing position
        realized_pnl = 0.0
        if (current_size > 0 and trade_size < 0) or (current_size < 0 and trade_size > 0):
            # Closing at least part of position
            close_size = min(abs(trade_size), abs(current_size))
            if current_size > 0:
                # Closing long
                realized_pnl = close_size * (execution_price - self.position.entry_price) * self.leverage
            else:
                # Closing short
                realized_pnl = close_size * (self.position.entry_price - execution_price) * self.leverage

            self.realized_pnl += realized_pnl
            self.balance += realized_pnl - commission
        else:
            self.balance -= commission

        # Update position
        if abs(new_size) > 1e-8:
            if abs(new_size) > abs(current_size):
                # Increasing position, update entry price
                self.position.entry_price = (
                    (current_size * self.position.entry_price + trade_size * execution_price) /
                    (current_size + trade_size)
                )
            self.position.size = new_size
        else:
            self.position = Position()

        # Record trade
        result['trade_size'] = trade_size
        result['trade_price'] = execution_price
        result['commission'] = commission
        result['slippage'] = slippage
        result['executed'] = True
        result['realized_pnl'] = realized_pnl

        self.trades_history.append({
            'step': self.current_step,
            'size': trade_size,
            'price': execution_price,
            'commission': commission,
            'pnl': realized_pnl
        })

        return result

    def _calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.position.is_flat():
            return 0.0

        current_price = self.current_price

        if self.position.is_long():
            return self.position.size * (current_price - self.position.entry_price) * self.leverage
        else:
            return -self.position.size * (self.position.entry_price - current_price) * self.leverage

    def _calculate_reward(self, trade_result: Dict) -> float:
        """
        Calculate reward for the step.

        Args:
            trade_result: Result of trade execution

        Returns:
            Reward value
        """
        # P&L based reward
        equity_change = self.equity - self.equity_history[-1] if self.equity_history else 0
        pnl_reward = equity_change / self.initial_balance

        # Risk-adjusted reward (Sharpe-like)
        if len(self.equity_history) > 10:
            returns = np.diff(self.equity_history[-10:]) / self.equity_history[-11:-1]
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                risk_reward = sharpe * 0.1
            else:
                risk_reward = 0
        else:
            risk_reward = 0

        # Trading cost penalty
        cost_penalty = trade_result['commission'] / self.initial_balance * 10

        # Position holding reward
        if not self.position.is_flat():
            # Reward for holding winning positions
            unrealized = self._calculate_unrealized_pnl()
            hold_reward = unrealized / self.initial_balance * 0.1
        else:
            hold_reward = 0

        # Combine rewards
        reward = (
            pnl_reward +
            risk_reward -
            cost_penalty +
            hold_reward
        ) * self.reward_scaling

        return reward

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.

        Returns:
            Observation array
        """
        # Get feature window
        start_idx = max(0, self.current_step - self.lookback_window + 1)
        end_idx = self.current_step + 1

        feature_window = self.data[self.feature_columns].iloc[start_idx:end_idx]

        # Pad if needed
        if len(feature_window) < self.lookback_window:
            padding = np.zeros((self.lookback_window - len(feature_window), len(self.feature_columns)))
            feature_window = np.vstack([padding, feature_window.values])
        else:
            feature_window = feature_window.values

        # Flatten features
        features = feature_window.flatten()

        # Position features
        position_features = np.array([
            self.position.size / self.max_position_size,
            self.position.entry_price / self.current_price - 1 if not self.position.is_flat() else 0,
            self._calculate_unrealized_pnl() / self.initial_balance,
            self.equity / self.initial_balance - 1
        ])

        # Combine
        observation = np.concatenate([features, position_features])

        return observation.astype(np.float32)

    def _get_info(self) -> Dict:
        """
        Get environment info.

        Returns:
            Info dictionary
        """
        return {
            'step': self.current_step,
            'balance': self.balance,
            'equity': self.equity,
            'position_size': self.position.size,
            'unrealized_pnl': self._calculate_unrealized_pnl(),
            'realized_pnl': self.realized_pnl,
            'n_trades': len(self.trades_history),
            'return': (self.equity - self.initial_balance) / self.initial_balance
        }

    def render(self) -> None:
        """Render the environment."""
        if self.render_mode == 'console':
            info = self._get_info()
            print(f"Step: {info['step']}, Equity: ${info['equity']:.2f}, "
                  f"Position: {info['position_size']:.4f}, "
                  f"Return: {info['return']*100:.2f}%")

    def close(self) -> None:
        """Close the environment."""
        pass

    def get_portfolio_stats(self) -> Dict[str, float]:
        """
        Get portfolio statistics.

        Returns:
            Dictionary with portfolio stats
        """
        if len(self.equity_history) < 2:
            return {}

        equity_array = np.array(self.equity_history)
        returns = np.diff(equity_array) / equity_array[:-1]

        total_return = (equity_array[-1] - equity_array[0]) / equity_array[0]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 * 60)  # Annualized

        # Max drawdown
        rolling_max = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)

        # Win rate
        winning_trades = [t for t in self.trades_history if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(self.trades_history) if self.trades_history else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_trades': len(self.trades_history),
            'final_equity': equity_array[-1]
        }


class MultiAssetTradingEnvironment(TradingEnvironment):
    """
    Multi-asset trading environment.

    Allows trading multiple assets simultaneously.
    """

    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        feature_columns: Optional[List[str]] = None,
        initial_balance: float = 100000.0,
        commission_rate: float = 0.0006,
        leverage: float = 10.0,
        max_position_size: float = 0.5,
        lookback_window: int = 20,
        **kwargs
    ):
        """
        Initialize Multi-Asset Trading Environment.

        Args:
            data_dict: Dictionary of symbol -> DataFrame
            feature_columns: Columns to use as features
            initial_balance: Starting balance
            commission_rate: Trading commission
            leverage: Maximum leverage
            max_position_size: Max position per asset
            lookback_window: Observation window
        """
        self.symbols = list(data_dict.keys())
        self.data_dict = data_dict
        self.n_assets = len(self.symbols)

        # Create combined data for observation
        combined_data = pd.DataFrame(index=list(data_dict.values())[0].index)
        for symbol, df in data_dict.items():
            for col in df.columns:
                if col != 'timestamp':
                    combined_data[f'{symbol}_{col}'] = df[col]

        super().__init__(
            data=combined_data,
            feature_columns=feature_columns,
            initial_balance=initial_balance,
            commission_rate=commission_rate,
            leverage=leverage,
            max_position_size=max_position_size,
            lookback_window=lookback_window,
            **kwargs
        )

        # Initialize positions for each asset
        self.positions: Dict[str, Position] = {
            symbol: Position() for symbol in self.symbols
        }

    @property
    def equity(self) -> float:
        """Calculate total equity."""
        unrealized = sum(
            self._calculate_asset_unrealized_pnl(symbol)
            for symbol in self.symbols
        )
        return self.balance + unrealized

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        obs, info = super().reset(seed=seed)

        # Reset positions
        self.positions = {symbol: Position() for symbol in self.symbols}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step with actions for each asset.

        Args:
            action: Array of position targets for each asset

        Returns:
            Standard step output
        """
        if len(action) != self.n_assets:
            raise ValueError(f"Expected {self.n_assets} actions, got {len(action)}")

        # Execute trades for each asset
        total_commission = 0
        total_realized_pnl = 0

        for i, symbol in enumerate(self.symbols):
            target_position = np.clip(action[i], -1, 1)
            current_price = self.data_dict[symbol]['close'].iloc[self.current_step]

            trade_result = self._execute_asset_trade(
                symbol, target_position, current_price
            )

            total_commission += trade_result['commission']
            total_realized_pnl += trade_result['realized_pnl']

        self.balance -= total_commission
        self.realized_pnl += total_realized_pnl

        # Move to next step
        self.current_step += 1

        terminated = self.current_step >= len(self.data) - 1
        truncated = self.equity <= self.initial_balance * 0.1

        reward = self._calculate_reward({'commission': total_commission})
        self.equity_history.append(self.equity)

        observation = self._get_observation() if not terminated else np.zeros(self.observation_dim)
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _execute_asset_trade(
        self,
        symbol: str,
        target_position: float,
        current_price: float
    ) -> Dict[str, Any]:
        """Execute trade for a specific asset."""
        position = self.positions[symbol]
        target_size = target_position * self.max_position_size
        current_size = position.size
        trade_size = target_size - current_size

        result = {
            'trade_size': 0,
            'commission': 0,
            'realized_pnl': 0
        }

        if abs(trade_size) < 1e-8:
            return result

        # Calculate commission
        trade_value = abs(trade_size) * current_price * self.leverage
        commission = trade_value * self.commission_rate

        # Calculate realized P&L
        realized_pnl = 0
        if (current_size > 0 and trade_size < 0) or (current_size < 0 and trade_size > 0):
            close_size = min(abs(trade_size), abs(current_size))
            if current_size > 0:
                realized_pnl = close_size * (current_price - position.entry_price) * self.leverage
            else:
                realized_pnl = close_size * (position.entry_price - current_price) * self.leverage

        # Update position
        new_size = current_size + trade_size
        if abs(new_size) > 1e-8:
            if abs(new_size) > abs(current_size):
                new_entry = (
                    (current_size * position.entry_price + trade_size * current_price) /
                    (current_size + trade_size)
                )
            else:
                new_entry = position.entry_price

            self.positions[symbol] = Position(
                size=new_size,
                entry_price=new_entry
            )
        else:
            self.positions[symbol] = Position()

        result['trade_size'] = trade_size
        result['commission'] = commission
        result['realized_pnl'] = realized_pnl

        return result

    def _calculate_asset_unrealized_pnl(self, symbol: str) -> float:
        """Calculate unrealized P&L for an asset."""
        position = self.positions[symbol]
        if position.is_flat():
            return 0.0

        current_price = self.data_dict[symbol]['close'].iloc[self.current_step]

        if position.is_long():
            return position.size * (current_price - position.entry_price) * self.leverage
        else:
            return -position.size * (position.entry_price - current_price) * self.leverage

    def _get_observation(self) -> np.ndarray:
        """Get multi-asset observation."""
        # This would include features for all assets
        # Simplified version for now
        features = []

        for symbol in self.symbols:
            df = self.data_dict[symbol]
            start_idx = max(0, self.current_step - self.lookback_window + 1)
            end_idx = self.current_step + 1

            window = df[['open', 'high', 'low', 'close', 'volume']].iloc[start_idx:end_idx]

            if len(window) < self.lookback_window:
                padding = np.zeros((self.lookback_window - len(window), 5))
                window = np.vstack([padding, window.values])
            else:
                window = window.values

            features.append(window.flatten())

        # Position info for each asset
        for symbol in self.symbols:
            pos = self.positions[symbol]
            pos_features = [
                pos.size / self.max_position_size,
                pos.entry_price / self.data_dict[symbol]['close'].iloc[self.current_step] - 1 if not pos.is_flat() else 0,
                self._calculate_asset_unrealized_pnl(symbol) / self.initial_balance
            ]
            features.append(pos_features)

        return np.concatenate([np.concatenate(f) if isinstance(f, np.ndarray) else f for f in features]).astype(np.float32)
