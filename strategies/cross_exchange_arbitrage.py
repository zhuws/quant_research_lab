"""
Cross-Exchange Arbitrage Strategy for Quant Research Lab.

Implements arbitrage strategies across multiple exchanges:
    - Simple price arbitrage (buy low, sell high across exchanges)
    - Triangular arbitrage
    - Latency arbitrage detection
    - Statistical arbitrage with mean reversion
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.math_utils import safe_divide
from strategies.base_strategy import (
    BaseStrategy, Signal, SignalType, Position, StrategyState, StrategyRegistry
)


class ArbitrageType(Enum):
    """Types of arbitrage opportunities."""
    SIMPLE_SPREAD = 'simple_spread'
    TRIANGULAR = 'triangular'
    STATISTICAL = 'statistical'
    LATENCY = 'latency'
    FUNDING = 'funding'


@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity."""
    arbitrage_type: ArbitrageType
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread: float
    spread_pct: float
    expected_profit: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'arbitrage_type': self.arbitrage_type.value,
            'symbol': self.symbol,
            'buy_exchange': self.buy_exchange,
            'sell_exchange': self.sell_exchange,
            'buy_price': self.buy_price,
            'sell_price': self.sell_price,
            'spread': self.spread,
            'spread_pct': self.spread_pct,
            'expected_profit': self.expected_profit,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class ExchangePrice:
    """Price data from an exchange."""
    exchange: str
    symbol: str
    bid: float
    ask: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid


@StrategyRegistry.register('cross_exchange_arbitrage')
class CrossExchangeArbitrage(BaseStrategy):
    """
    Cross-Exchange Arbitrage Strategy.

    Detects and executes arbitrage opportunities across multiple exchanges.

    Features:
        - Simple spread arbitrage
        - Triangular arbitrage
        - Statistical arbitrage with z-score
        - Real-time opportunity detection
        - Execution cost analysis

    Attributes:
        min_spread_pct: Minimum spread percentage to trigger trade
        transaction_cost: Cost per trade (per exchange)
        max_hold_time: Maximum time to hold arbitrage position
        statistical_window: Window for statistical arbitrage calculations
    """

    def __init__(
        self,
        name: str = 'CrossExchangeArbitrage',
        capital: float = 100000,
        min_spread_pct: float = 0.002,
        transaction_cost: float = 0.001,
        max_hold_time: float = 60.0,
        statistical_window: int = 100,
        z_score_threshold: float = 2.0,
        exchanges: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None
    ):
        """
        Initialize Cross-Exchange Arbitrage Strategy.

        Args:
            name: Strategy name
            capital: Initial capital
            min_spread_pct: Minimum spread to trade (default: 0.2%)
            transaction_cost: Transaction cost per trade per exchange
            max_hold_time: Maximum hold time in seconds
            statistical_window: Window for z-score calculation
            z_score_threshold: Z-score threshold for statistical arbitrage
            exchanges: List of exchanges to monitor
            symbols: List of symbols to trade
        """
        super().__init__(
            name=name,
            capital=capital,
            max_positions=10,
            risk_per_trade=0.05,
            transaction_cost=transaction_cost
        )

        self.min_spread_pct = min_spread_pct
        self.max_hold_time = max_hold_time
        self.statistical_window = statistical_window
        self.z_score_threshold = z_score_threshold
        self.exchanges = exchanges or ['binance', 'bybit']
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']

        # Price data storage
        self._prices: Dict[str, Dict[str, ExchangePrice]] = defaultdict(dict)
        self._price_history: Dict[str, pd.DataFrame] = {}

        # Opportunity tracking
        self._opportunities: List[ArbitrageOpportunity] = []
        self._opportunity_history: List[Dict] = []

        # Triangular paths (cached)
        self._triangular_paths: List[List[str]] = []

        # Statistics for statistical arbitrage
        self._spread_history: Dict[str, List[float]] = defaultdict(list)
        self._spread_mean: Dict[str, float] = {}
        self._spread_std: Dict[str, float] = {}

        self.logger.info(
            f"Initialized CrossExchangeArbitrage with min_spread={min_spread_pct*100:.2f}%"
        )

    def initialize(self, **kwargs) -> None:
        """
        Initialize strategy with configuration.

        Args:
            **kwargs: Configuration including exchanges, symbols, etc.
        """
        super().initialize(**kwargs)

        if 'exchanges' in kwargs:
            self.exchanges = kwargs['exchanges']
        if 'symbols' in kwargs:
            self.symbols = kwargs['symbols']
        if 'min_spread_pct' in kwargs:
            self.min_spread_pct = kwargs['min_spread_pct']

        # Initialize price history storage
        for symbol in self.symbols:
            self._price_history[symbol] = pd.DataFrame()

    def update_price(
        self,
        exchange: str,
        symbol: str,
        bid: float,
        ask: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Update price data from an exchange.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            bid: Best bid price
            ask: Best ask price
            timestamp: Timestamp (optional)
        """
        price = ExchangePrice(
            exchange=exchange,
            symbol=symbol,
            bid=bid,
            ask=ask,
            timestamp=timestamp or datetime.now()
        )

        self._prices[symbol][exchange] = price

        # Update price history
        record = {
            'timestamp': price.timestamp,
            'exchange': exchange,
            'bid': bid,
            'ask': ask,
            'mid': price.mid
        }

        if symbol in self._price_history:
            self._price_history[symbol] = pd.concat([
                self._price_history[symbol],
                pd.DataFrame([record])
            ], ignore_index=True)

        # Update spread statistics
        self._update_spread_stats(symbol)

    def _update_spread_stats(self, symbol: str) -> None:
        """Update spread statistics for statistical arbitrage."""
        if symbol not in self._prices or len(self._prices[symbol]) < 2:
            return

        exchanges = list(self._prices[symbol].keys())
        if len(exchanges) < 2:
            return

        # Calculate spread between exchanges
        p1 = self._prices[symbol][exchanges[0]]
        p2 = self._prices[symbol][exchanges[1]]

        spread_pct = (p1.mid - p2.mid) / p2.mid

        key = f"{symbol}_{exchanges[0]}_{exchanges[1]}"
        self._spread_history[key].append(spread_pct)

        # Keep only last N values
        if len(self._spread_history[key]) > self.statistical_window:
            self._spread_history[key] = self._spread_history[key][-self.statistical_window:]

        # Update mean and std
        if len(self._spread_history[key]) >= 10:
            self._spread_mean[key] = np.mean(self._spread_history[key])
            self._spread_std[key] = np.std(self._spread_history[key])

    def detect_opportunities(self) -> List[ArbitrageOpportunity]:
        """
        Detect all arbitrage opportunities.

        Returns:
            List of ArbitrageOpportunity objects
        """
        opportunities = []

        # Simple spread arbitrage
        simple_opps = self._detect_simple_spread()
        opportunities.extend(simple_opps)

        # Statistical arbitrage
        stat_opps = self._detect_statistical_arbitrage()
        opportunities.extend(stat_opps)

        # Triangular arbitrage (if configured)
        if len(self.symbols) >= 3:
            tri_opps = self._detect_triangular()
            opportunities.extend(tri_opps)

        # Filter by minimum spread
        valid_opps = [
            opp for opp in opportunities
            if opp.spread_pct >= self.min_spread_pct
        ]

        # Store opportunities
        self._opportunities = valid_opps
        self._opportunity_history.extend([opp.to_dict() for opp in valid_opps])

        return valid_opps

    def _detect_simple_spread(self) -> List[ArbitrageOpportunity]:
        """Detect simple spread arbitrage opportunities."""
        opportunities = []

        for symbol in self._prices:
            exchange_prices = self._prices[symbol]

            # Check all exchange pairs
            exchanges = list(exchange_prices.keys())

            for i, ex1 in enumerate(exchanges):
                for ex2 in exchanges[i+1:]:
                    p1 = exchange_prices[ex1]
                    p2 = exchange_prices[ex2]

                    # Check both directions

                    # Direction 1: Buy on ex1, sell on ex2
                    spread1 = p2.bid - p1.ask
                    spread_pct1 = spread1 / p1.ask

                    if spread_pct1 >= self.min_spread_pct:
                        # Account for transaction costs
                        net_profit = spread_pct1 - 2 * self.transaction_cost

                        if net_profit > 0:
                            opp = ArbitrageOpportunity(
                                arbitrage_type=ArbitrageType.SIMPLE_SPREAD,
                                symbol=symbol,
                                buy_exchange=ex1,
                                sell_exchange=ex2,
                                buy_price=p1.ask,
                                sell_price=p2.bid,
                                spread=spread1,
                                spread_pct=spread_pct1,
                                expected_profit=net_profit,
                                metadata={'direction': 'buy_ex1_sell_ex2'}
                            )
                            opportunities.append(opp)

                    # Direction 2: Buy on ex2, sell on ex1
                    spread2 = p1.bid - p2.ask
                    spread_pct2 = spread2 / p2.ask

                    if spread_pct2 >= self.min_spread_pct:
                        net_profit = spread_pct2 - 2 * self.transaction_cost

                        if net_profit > 0:
                            opp = ArbitrageOpportunity(
                                arbitrage_type=ArbitrageType.SIMPLE_SPREAD,
                                symbol=symbol,
                                buy_exchange=ex2,
                                sell_exchange=ex1,
                                buy_price=p2.ask,
                                sell_price=p1.bid,
                                spread=spread2,
                                spread_pct=spread_pct2,
                                expected_profit=net_profit,
                                metadata={'direction': 'buy_ex2_sell_ex1'}
                            )
                            opportunities.append(opp)

        return opportunities

    def _detect_statistical_arbitrage(self) -> List[ArbitrageOpportunity]:
        """Detect statistical arbitrage opportunities using z-score."""
        opportunities = []

        for key, spreads in self._spread_history.items():
            if len(spreads) < self.statistical_window:
                continue

            if key not in self._spread_mean or key not in self._spread_std:
                continue

            mean = self._spread_mean[key]
            std = self._spread_std[key]

            if std == 0:
                continue

            current_spread = spreads[-1]
            z_score = (current_spread - mean) / std

            # Parse key
            parts = key.split('_')
            if len(parts) < 3:
                continue

            symbol = '_'.join(parts[:-2])
            ex1, ex2 = parts[-2], parts[-1]

            # Get current prices
            if symbol not in self._prices:
                continue
            if ex1 not in self._prices[symbol] or ex2 not in self._prices[symbol]:
                continue

            p1 = self._prices[symbol][ex1]
            p2 = self._prices[symbol][ex2]

            # Z-score above threshold: spread is high, expect mean reversion
            # Sell on exchange 1, buy on exchange 2
            if z_score > self.z_score_threshold:
                opp = ArbitrageOpportunity(
                    arbitrage_type=ArbitrageType.STATISTICAL,
                    symbol=symbol,
                    buy_exchange=ex2,
                    sell_exchange=ex1,
                    buy_price=p2.ask,
                    sell_price=p1.bid,
                    spread=abs(p1.bid - p2.ask),
                    spread_pct=abs(current_spread),
                    expected_profit=abs(z_score) * std - 2 * self.transaction_cost,
                    metadata={
                        'z_score': z_score,
                        'mean': mean,
                        'std': std,
                        'direction': 'mean_reversion_down'
                    }
                )
                opportunities.append(opp)

            # Z-score below negative threshold: spread is low, expect mean reversion
            # Buy on exchange 1, sell on exchange 2
            elif z_score < -self.z_score_threshold:
                opp = ArbitrageOpportunity(
                    arbitrage_type=ArbitrageType.STATISTICAL,
                    symbol=symbol,
                    buy_exchange=ex1,
                    sell_exchange=ex2,
                    buy_price=p1.ask,
                    sell_price=p2.bid,
                    spread=abs(p2.bid - p1.ask),
                    spread_pct=abs(current_spread),
                    expected_profit=abs(z_score) * std - 2 * self.transaction_cost,
                    metadata={
                        'z_score': z_score,
                        'mean': mean,
                        'std': std,
                        'direction': 'mean_reversion_up'
                    }
                )
                opportunities.append(opp)

        return opportunities

    def _detect_triangular(self) -> List[ArbitrageOpportunity]:
        """
        Detect triangular arbitrage opportunities.

        Requires at least 3 symbols with proper quote relationships.
        """
        opportunities = []

        # This is a simplified implementation
        # In practice, you need proper currency pair relationships

        if len(self.symbols) < 3:
            return opportunities

        # For triangular arbitrage, we need proper pair relationships
        # e.g., BTC/USDT, ETH/BTC, ETH/USDT

        # This would require additional logic to track cross rates
        # For now, return empty list

        return opportunities

    def generate_signals(
        self,
        data: pd.DataFrame,
        **kwargs
    ) -> List[Signal]:
        """
        Generate trading signals from market data.

        Args:
            data: Market data (can be multi-exchange)
            **kwargs: Additional parameters

        Returns:
            List of Signal objects
        """
        if self.state != StrategyState.RUNNING:
            return []

        signals = []

        # Detect opportunities
        opportunities = self.detect_opportunities()

        for opp in opportunities:
            # Create buy signal for the buy exchange
            buy_signal = Signal(
                symbol=opp.symbol,
                signal_type=SignalType.BUY,
                price=opp.buy_price,
                timestamp=datetime.now(),
                strategy_id=self.strategy_id,
                exchange=opp.buy_exchange,
                confidence=min(1.0, opp.spread_pct / self.min_spread_pct),
                metadata={
                    'arbitrage_type': opp.arbitrage_type.value,
                    'paired_exchange': opp.sell_exchange,
                    'expected_profit': opp.expected_profit,
                    **opp.metadata
                }
            )

            # Create sell signal for the sell exchange
            sell_signal = Signal(
                symbol=opp.symbol,
                signal_type=SignalType.SELL,
                price=opp.sell_price,
                timestamp=datetime.now(),
                strategy_id=self.strategy_id,
                exchange=opp.sell_exchange,
                confidence=min(1.0, opp.spread_pct / self.min_spread_pct),
                metadata={
                    'arbitrage_type': opp.arbitrage_type.value,
                    'paired_exchange': opp.buy_exchange,
                    'expected_profit': opp.expected_profit,
                    **opp.metadata
                }
            )

            signals.extend([buy_signal, sell_signal])

        return signals

    def should_close_position(
        self,
        position: Position,
        current_data: pd.Series
    ) -> Optional[Signal]:
        """
        Determine if position should be closed.

        For arbitrage, close when spread normalizes or max hold time exceeded.

        Args:
            position: Position to evaluate
            current_data: Current market data

        Returns:
            Close signal if should close
        """
        # Check max hold time
        hold_time = (datetime.now() - position.entry_time).total_seconds()
        if hold_time > self.max_hold_time:
            return Signal(
                symbol=position.symbol,
                signal_type=SignalType.CLOSE_LONG if position.side == 'long'
                           else SignalType.CLOSE_SHORT,
                price=current_data.get('close', 0),
                timestamp=datetime.now(),
                strategy_id=self.strategy_id,
                metadata={'reason': 'max_hold_time_exceeded'}
            )

        # For statistical arbitrage, check z-score
        if 'z_score' in position.metadata:
            key = f"{position.symbol}_{position.metadata.get('ex1')}_{position.metadata.get('ex2')}"

            if key in self._spread_mean and key in self._spread_std:
                current_z = position.metadata['z_score']

                # Close if z-score has reverted towards 0
                if abs(current_z) < 0.5:  # Z-score normalized
                    return Signal(
                        symbol=position.symbol,
                        signal_type=SignalType.CLOSE_LONG if position.side == 'long'
                                   else SignalType.CLOSE_SHORT,
                        price=current_data.get('close', 0),
                        timestamp=datetime.now(),
                        strategy_id=self.strategy_id,
                        metadata={'reason': 'mean_reversion_complete'}
                    )

        return None

    def calculate_expected_profit(
        self,
        symbol: str,
        size: float,
        buy_exchange: str,
        sell_exchange: str
    ) -> float:
        """
        Calculate expected profit for an arbitrage trade.

        Args:
            symbol: Trading symbol
            size: Trade size
            buy_exchange: Exchange to buy on
            sell_exchange: Exchange to sell on

        Returns:
            Expected profit in quote currency
        """
        if symbol not in self._prices:
            return 0

        if buy_exchange not in self._prices[symbol] or sell_exchange not in self._prices[symbol]:
            return 0

        buy_price = self._prices[symbol][buy_exchange].ask
        sell_price = self._prices[symbol][sell_exchange].bid

        # Gross profit
        gross_profit = (sell_price - buy_price) * size

        # Transaction costs
        costs = (buy_price + sell_price) * size * self.transaction_cost

        return gross_profit - costs

    def get_best_opportunity(self) -> Optional[ArbitrageOpportunity]:
        """Get the best current arbitrage opportunity."""
        if not self._opportunities:
            return None

        return max(self._opportunities, key=lambda x: x.expected_profit)

    def get_opportunity_summary(self) -> Dict:
        """Get summary of recent opportunities."""
        if not self._opportunity_history:
            return {
                'total_opportunities': 0,
                'by_type': {},
                'avg_spread': 0,
                'avg_profit': 0
            }

        df = pd.DataFrame(self._opportunity_history)

        return {
            'total_opportunities': len(df),
            'by_type': df['arbitrage_type'].value_counts().to_dict() if 'arbitrage_type' in df.columns else {},
            'avg_spread': df['spread_pct'].mean() if 'spread_pct' in df.columns else 0,
            'avg_profit': df['expected_profit'].mean() if 'expected_profit' in df.columns else 0,
            'best_spread': df['spread_pct'].max() if 'spread_pct' in df.columns else 0,
            'symbols': df['symbol'].unique().tolist() if 'symbol' in df.columns else []
        }

    def get_spread_statistics(self) -> Dict:
        """Get spread statistics for all tracked pairs."""
        stats = {}

        for key, spreads in self._spread_history.items():
            if len(spreads) < 10:
                continue

            stats[key] = {
                'mean': self._spread_mean.get(key, 0),
                'std': self._spread_std.get(key, 0),
                'current': spreads[-1],
                'min': min(spreads),
                'max': max(spreads),
                'count': len(spreads)
            }

        return stats


class LatencyArbitrage(BaseStrategy):
    """
    Latency Arbitrage Strategy.

    Exploits price differences due to latency between exchanges.
    Requires very fast execution.
    """

    def __init__(
        self,
        name: str = 'LatencyArbitrage',
        capital: float = 100000,
        latency_threshold_ms: float = 50.0,
        min_spread_pct: float = 0.001,
        **kwargs
    ):
        """
        Initialize Latency Arbitrage Strategy.

        Args:
            name: Strategy name
            capital: Initial capital
            latency_threshold_ms: Maximum acceptable latency
            min_spread_pct: Minimum spread to trade
        """
        super().__init__(name=name, capital=capital, **kwargs)

        self.latency_threshold_ms = latency_threshold_ms
        self.min_spread_pct = min_spread_pct

        self._last_prices: Dict[str, Dict[str, ExchangePrice]] = defaultdict(dict)
        self._latencies: Dict[str, List[float]] = defaultdict(list)

    def record_latency(self, exchange: str, latency_ms: float) -> None:
        """Record exchange latency measurement."""
        self._latencies[exchange].append(latency_ms)
        # Keep last 100 measurements
        if len(self._latencies[exchange]) > 100:
            self._latencies[exchange] = self._latencies[exchange][-100:]

    def get_avg_latency(self, exchange: str) -> float:
        """Get average latency for an exchange."""
        if exchange not in self._latencies or not self._latencies[exchange]:
            return float('inf')
        return np.mean(self._latencies[exchange])

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> List[Signal]:
        """Generate latency arbitrage signals."""
        # This is a placeholder - real latency arbitrage requires
        # direct exchange connections and microsecond-level timing
        return []

    def should_close_position(self, position: Position, current_data: pd.Series) -> Optional[Signal]:
        """Check if position should be closed."""
        # Latency arbitrage positions should be closed very quickly
        hold_time = (datetime.now() - position.entry_time).total_seconds()
        if hold_time > 5:  # 5 seconds max
            return Signal(
                symbol=position.symbol,
                signal_type=SignalType.CLOSE_LONG if position.side == 'long'
                           else SignalType.CLOSE_SHORT,
                timestamp=datetime.now(),
                strategy_id=self.strategy_id,
                metadata={'reason': 'latency_timeout'}
            )
        return None


class TriangularArbitrage(BaseStrategy):
    """
    Triangular Arbitrage Strategy.

    Exploits price inefficiencies in currency triangles.
    Example: BTC/USDT -> ETH/BTC -> ETH/USDT
    """

    def __init__(
        self,
        name: str = 'TriangularArbitrage',
        capital: float = 100000,
        min_profit_pct: float = 0.002,
        **kwargs
    ):
        """
        Initialize Triangular Arbitrage Strategy.

        Args:
            name: Strategy name
            capital: Initial capital
            min_profit_pct: Minimum profit percentage to trade
        """
        super().__init__(name=name, capital=capital, **kwargs)

        self.min_profit_pct = min_profit_pct

        # Define common triangular paths
        self._paths: List[Dict] = []
        self._path_history: Dict[str, List[float]] = defaultdict(list)

    def add_path(
        self,
        pair1: str,
        pair2: str,
        pair3: str,
        direction: str = 'forward'
    ) -> None:
        """
        Add a triangular arbitrage path.

        Args:
            pair1: First trading pair (e.g., 'BTCUSDT')
            pair2: Second trading pair (e.g., 'ETHBTC')
            pair3: Third trading pair (e.g., 'ETHUSDT')
            direction: 'forward' or 'reverse'
        """
        self._paths.append({
            'pair1': pair1,
            'pair2': pair2,
            'pair3': pair3,
            'direction': direction
        })

    def calculate_path_profit(
        self,
        prices: Dict[str, float],
        path: Dict,
        amount: float = 1.0
    ) -> float:
        """
        Calculate profit for a triangular path.

        Args:
            prices: Dictionary of pair -> price
            path: Path definition
            amount: Starting amount

        Returns:
            Profit as fraction of starting amount
        """
        try:
            if path['direction'] == 'forward':
                # Step 1: Convert quote to base1
                p1 = prices.get(path['pair1'], 0)
                if p1 == 0:
                    return 0
                amount1 = amount / p1

                # Step 2: Convert base1 to base2
                p2 = prices.get(path['pair2'], 0)
                if p2 == 0:
                    return 0
                amount2 = amount1 * p2

                # Step 3: Convert base2 back to quote
                p3 = prices.get(path['pair3'], 0)
                if p3 == 0:
                    return 0
                amount3 = amount2 * p3

                return (amount3 - amount) / amount
            else:
                # Reverse direction
                p3 = prices.get(path['pair3'], 0)
                if p3 == 0:
                    return 0
                amount1 = amount * p3

                p2 = prices.get(path['pair2'], 0)
                if p2 == 0:
                    return 0
                amount2 = amount1 / p2

                p1 = prices.get(path['pair1'], 0)
                if p1 == 0:
                    return 0
                amount3 = amount2 * p1

                return (amount3 - amount) / amount

        except Exception:
            return 0

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> List[Signal]:
        """Generate triangular arbitrage signals."""
        signals = []

        for path in self._paths:
            # Get latest prices for all pairs in path
            prices = {}
            for pair in [path['pair1'], path['pair2'], path['pair3']]:
                if pair in data.columns:
                    prices[pair] = data[pair].iloc[-1]

            if len(prices) < 3:
                continue

            # Calculate profit for both directions
            profit_forward = self.calculate_path_profit(prices, path, direction='forward')
            profit_reverse = self.calculate_path_profit(prices, path, direction='reverse')

            # Account for transaction costs
            net_profit_forward = profit_forward - 3 * self.transaction_cost
            net_profit_reverse = profit_reverse - 3 * self.transaction_cost

            if net_profit_forward >= self.min_profit_pct:
                signal = Signal(
                    symbol=path['pair1'],
                    signal_type=SignalType.BUY,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    metadata={
                        'path': path,
                        'expected_profit': net_profit_forward,
                        'direction': 'forward'
                    }
                )
                signals.append(signal)

            elif net_profit_reverse >= self.min_profit_pct:
                signal = Signal(
                    symbol=path['pair1'],
                    signal_type=SignalType.SELL,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    metadata={
                        'path': path,
                        'expected_profit': net_profit_reverse,
                        'direction': 'reverse'
                    }
                )
                signals.append(signal)

        return signals

    def should_close_position(self, position: Position, current_data: pd.Series) -> Optional[Signal]:
        """Check if triangular position should be closed."""
        # Triangular arbitrage should complete almost instantly
        hold_time = (datetime.now() - position.entry_time).total_seconds()
        if hold_time > 10:  # 10 seconds max
            return Signal(
                symbol=position.symbol,
                signal_type=SignalType.CLOSE_LONG if position.side == 'long'
                           else SignalType.CLOSE_SHORT,
                timestamp=datetime.now(),
                strategy_id=self.strategy_id,
                metadata={'reason': 'triangular_timeout'}
            )
        return None


def create_arbitrage_strategy(
    strategy_type: str = 'cross_exchange',
    **kwargs
) -> BaseStrategy:
    """
    Convenience function to create an arbitrage strategy.

    Args:
        strategy_type: Type of arbitrage strategy
        **kwargs: Strategy parameters

    Returns:
        Configured strategy instance
    """
    strategies = {
        'cross_exchange': CrossExchangeArbitrage,
        'latency': LatencyArbitrage,
        'triangular': TriangularArbitrage
    }

    strategy_class = strategies.get(strategy_type)
    if strategy_class:
        return strategy_class(**kwargs)

    raise ValueError(f"Unknown strategy type: {strategy_type}")
