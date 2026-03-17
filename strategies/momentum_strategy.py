"""
Momentum Strategy for Quant Research Lab.

A simple momentum-based trading strategy that generates buy/sell signals
based on price momentum indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base_strategy import (
    BaseStrategy, Signal, SignalType, Position, StrategyState
)
from utils.logger import get_logger


class MomentumStrategy(BaseStrategy):
    """
    Momentum Trading Strategy.

    Generates signals based on price momentum using:
        - Moving average crossovers
        - RSI (Relative Strength Index)
        - Volume confirmation

    Attributes:
        fast_period: Fast moving average period
        slow_period: Slow moving average period
        rsi_period: RSI calculation period
        rsi_overbought: RSI overbought threshold
        rsi_oversold: RSI oversold threshold
    """

    def __init__(
        self,
        name: str = 'MomentumStrategy',
        capital: float = 100000,
        fast_period: int = 10,
        slow_period: int = 30,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        **kwargs
    ):
        """
        Initialize Momentum Strategy.

        Args:
            name: Strategy name
            capital: Initial capital
            fast_period: Fast MA period (default: 10)
            slow_period: Slow MA period (default: 30)
            rsi_period: RSI period (default: 14)
            rsi_overbought: RSI overbought level (default: 70)
            rsi_oversold: RSI oversold level (default: 30)
            **kwargs: Additional parameters passed to BaseStrategy
        """
        super().__init__(name=name, capital=capital, **kwargs)

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

        self.logger = get_logger('momentum_strategy')
        self._signals_df = None

        # For real-time signal generation
        self._price_buffer = []
        self._buffer_size = max(self.slow_period, self.rsi_period) + 5

    def generate_signal(
        self,
        data: dict,
        **kwargs
    ) -> Optional[Signal]:
        """
        Generate a trading signal from real-time market data.

        Args:
            data: Real-time market data dict (from WebSocket)
            **kwargs: Additional parameters

        Returns:
            Signal if one is generated, None otherwise
        """
        # Extract price from data
        price = None
        symbol = data.get('symbol', 'ETHUSDT')

        # Handle different data formats
        if 'close' in data:
            price = float(data['close'])
        elif 'price' in data:
            price = float(data['price'])
        elif 'p' in data:  # Binance ticker format
            price = float(data['p'])
        elif 'lastPrice' in data:
            price = float(data['lastPrice'])

        if price is None:
            return None

        # Update price buffer
        self._price_buffer.append(price)
        if len(self._price_buffer) > self._buffer_size:
            self._price_buffer = self._price_buffer[-self._buffer_size:]

        # Need enough data to calculate indicators
        if len(self._price_buffer) < self.slow_period:
            return None

        # Calculate indicators from buffer
        prices = pd.Series(self._price_buffer)

        ma_fast = prices.rolling(window=self.fast_period).mean().iloc[-1]
        ma_slow = prices.rolling(window=self.slow_period).mean().iloc[-1]

        # Calculate RSI
        rsi = self._calculate_rsi(prices, self.rsi_period).iloc[-1]

        # Generate signal
        signal_type = None

        # Buy signal: Fast MA above Slow MA + RSI not overbought
        if ma_fast > ma_slow and rsi < self.rsi_overbought:
            signal_type = SignalType.BUY

        # Sell signal: Fast MA below Slow MA + RSI not oversold
        elif ma_fast < ma_slow and rsi > self.rsi_oversold:
            signal_type = SignalType.SELL

        if signal_type and signal_type != SignalType.NO_SIGNAL:
            return Signal(
                symbol=symbol,
                signal_type=signal_type,
                price=price,
                timestamp=datetime.now(),
                strategy_id=self.strategy_id,
                metadata={
                    'ma_fast': ma_fast,
                    'ma_slow': ma_slow,
                    'rsi': rsi
                }
            )

        return None

    def generate_signals(
        self,
        data: pd.DataFrame,
        **kwargs
    ) -> List[Signal]:
        """
        Generate trading signals from market data.

        Args:
            data: OHLCV DataFrame
            **kwargs: Additional parameters

        Returns:
            List of Signal objects
        """
        if data.empty:
            return []

        df = data.copy()

        # Ensure we have the required columns
        if 'close' not in df.columns:
            self.logger.warning("No 'close' column in data")
            return []

        # Calculate indicators
        df = self._calculate_indicators(df)

        # Generate signals based on indicators
        signals = self._generate_signals_from_indicators(df)

        # Store processed dataframe for vectorized backtest
        self._signals_df = df

        return signals

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()

        # Moving Averages
        df['ma_fast'] = df['close'].rolling(window=self.fast_period).mean()
        df['ma_slow'] = df['close'].rolling(window=self.slow_period).mean()

        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)

        # Momentum (rate of change)
        df['momentum'] = df['close'].pct_change(periods=5)

        # Volume MA (if volume available)
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.

        Args:
            prices: Price series
            period: RSI period

        Returns:
            RSI values
        """
        delta = prices.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)  # Default to neutral RSI

    def _generate_signals_from_indicators(self, df: pd.DataFrame) -> List[Signal]:
        """
        Generate signals from calculated indicators.

        Args:
            df: DataFrame with indicator columns

        Returns:
            List of Signal objects
        """
        signals = []
        symbol = 'ETHUSDT'  # Default symbol

        # Determine symbol from data if available
        if 'symbol' in df.columns:
            symbol = df['symbol'].iloc[0]

        # Track position state
        current_position = 0  # 0: flat, 1: long, -1: short

        for idx, row in df.iterrows():
            # Skip if indicators are NaN
            if pd.isna(row.get('ma_fast')) or pd.isna(row.get('ma_slow')):
                continue

            signal_type = None
            timestamp = row.get('timestamp', datetime.now())

            # MA Crossover signals
            ma_fast = row['ma_fast']
            ma_slow = row['ma_slow']
            rsi = row.get('rsi', 50)

            # Buy signal: Fast MA crosses above Slow MA + RSI not overbought
            if ma_fast > ma_slow and current_position <= 0:
                if rsi < self.rsi_overbought:
                    signal_type = SignalType.BUY
                    current_position = 1

            # Sell signal: Fast MA crosses below Slow MA + RSI not oversold
            elif ma_fast < ma_slow and current_position >= 0:
                if rsi > self.rsi_oversold:
                    if current_position == 1:
                        signal_type = SignalType.CLOSE_LONG
                    else:
                        signal_type = SignalType.SELL
                    current_position = -1 if signal_type == SignalType.SELL else 0

            # Additional RSI-based signals
            elif current_position == 1 and rsi > self.rsi_overbought:
                signal_type = SignalType.CLOSE_LONG
                current_position = 0

            elif current_position == -1 and rsi < self.rsi_oversold:
                signal_type = SignalType.CLOSE_SHORT
                current_position = 0

            if signal_type and signal_type != SignalType.NO_SIGNAL:
                signal = Signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    price=row['close'],
                    timestamp=timestamp if isinstance(timestamp, datetime) else datetime.now(),
                    strategy_id=self.strategy_id,
                    metadata={
                        'ma_fast': ma_fast,
                        'ma_slow': ma_slow,
                        'rsi': rsi
                    }
                )
                signals.append(signal)

        self.logger.info(f"Generated {len(signals)} signals")
        return signals

    def should_close_position(
        self,
        position: Position,
        current_data: pd.Series
    ) -> Optional[Signal]:
        """
        Determine if position should be closed.

        Args:
            position: Current position
            current_data: Current market data

        Returns:
            Close signal if should close, None otherwise
        """
        if pd.isna(current_data.get('rsi')):
            return None

        rsi = current_data['rsi']
        ma_fast = current_data.get('ma_fast', 0)
        ma_slow = current_data.get('ma_slow', 0)

        close_signal = None

        if position.side == 'long':
            # Close long on RSI overbought or MA crossover
            if rsi > self.rsi_overbought or ma_fast < ma_slow:
                close_signal = Signal(
                    symbol=position.symbol,
                    signal_type=SignalType.CLOSE_LONG,
                    price=current_data['close'],
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    metadata={'reason': 'momentum_reversal'}
                )

        elif position.side == 'short':
            # Close short on RSI oversold or MA crossover
            if rsi < self.rsi_oversold or ma_fast > ma_slow:
                close_signal = Signal(
                    symbol=position.symbol,
                    signal_type=SignalType.CLOSE_SHORT,
                    price=current_data['close'],
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    metadata={'reason': 'momentum_reversal'}
                )

        return close_signal

    def get_params(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold
        }

    def set_params(self, **kwargs) -> None:
        """Set strategy parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.logger.info(f"Updated parameters: {kwargs}")
