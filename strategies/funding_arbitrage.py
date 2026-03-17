"""
Funding Rate Arbitrage Strategy for Quant Research Lab.

Implements funding rate arbitrage (cash-and-carry) strategies:
    - Perpetual vs spot arbitrage
    - Cross-exchange funding rate arbitrage
    - Funding rate prediction and optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.math_utils import safe_divide
from strategies.base_strategy import (
    BaseStrategy, Signal, SignalType, Position, StrategyState, StrategyRegistry
)


class FundingPositionType(Enum):
    """Types of funding arbitrage positions."""
    LONG_PERP = 'long_perpetual'
    SHORT_PERP = 'short_perpetual'
    CASH_CARRY = 'cash_and_carry'  # Long spot + short perp
    REVERSE_CASH_CARRY = 'reverse_cash_and_carry'  # Short spot + long perp
    CROSS_EXCHANGE = 'cross_exchange'  # Long perp on one exchange, short on another


@dataclass
class FundingRate:
    """Funding rate data structure."""
    exchange: str
    symbol: str
    funding_rate: float
    next_funding_time: datetime
    timestamp: datetime = field(default_factory=datetime.now)
    predicted_rate: Optional[float] = None

    @property
    def annualized_rate(self) -> float:
        """Get annualized funding rate (assuming 8h funding periods)."""
        return self.funding_rate * 3 * 365  # 3 funding periods per day


@dataclass
class FundingOpportunity:
    """Funding rate arbitrage opportunity."""
    symbol: str
    position_type: FundingPositionType
    exchanges: List[str]
    spot_price: float
    perp_price: float
    basis: float
    basis_pct: float
    funding_rate: float
    annualized_return: float
    expected_profit: float
    hold_period: int  # Number of funding periods
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


@StrategyRegistry.register('funding_arbitrage')
class FundingRateArbitrage(BaseStrategy):
    """
    Funding Rate Arbitrage Strategy.

    Exploits differences between perpetual futures funding rates and spot prices.

    Features:
        - Cash-and-carry arbitrage
        - Cross-exchange funding arbitrage
        - Funding rate prediction
        - Automated position management
        - Exit timing optimization

    Attributes:
        min_funding_rate: Minimum funding rate to trigger trade
        min_basis_pct: Minimum basis percentage to trade
        max_hold_periods: Maximum funding periods to hold
        use_prediction: Whether to use predicted funding rates
    """

    def __init__(
        self,
        name: str = 'FundingRateArbitrage',
        capital: float = 100000,
        min_funding_rate: float = 0.0005,  # 0.05% per 8h = ~18% APY
        min_basis_pct: float = 0.002,  # 0.2% minimum basis
        max_hold_periods: int = 21,  # ~7 days max hold
        target_annual_return: float = 0.15,  # 15% target APY
        use_prediction: bool = True,
        close_before_funding: bool = True,
        exchanges: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None
    ):
        """
        Initialize Funding Rate Arbitrage Strategy.

        Args:
            name: Strategy name
            capital: Initial capital
            min_funding_rate: Minimum funding rate to trade (per period)
            min_basis_pct: Minimum basis percentage
            max_hold_periods: Maximum funding periods to hold
            target_annual_return: Target annual return
            use_prediction: Use predicted funding rates
            close_before_funding: Close positions before funding if negative
            exchanges: List of exchanges
            symbols: List of symbols to trade
        """
        super().__init__(
            name=name,
            capital=capital,
            max_positions=5,
            risk_per_trade=0.10,
            transaction_cost=0.0005  # Lower cost for funding arb
        )

        self.min_funding_rate = min_funding_rate
        self.min_basis_pct = min_basis_pct
        self.max_hold_periods = max_hold_periods
        self.target_annual_return = target_annual_return
        self.use_prediction = use_prediction
        self.close_before_funding = close_before_funding
        self.exchanges = exchanges or ['binance', 'bybit']
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']

        # Funding rate storage
        self._funding_rates: Dict[str, Dict[str, FundingRate]] = defaultdict(dict)
        self._funding_history: Dict[str, List[Dict]] = defaultdict(list)

        # Price storage
        self._spot_prices: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._perp_prices: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Basis tracking
        self._basis_history: Dict[str, List[float]] = defaultdict(list)

        # Funding prediction model (simple EMA-based)
        self._predicted_rates: Dict[str, float] = {}

        # Opportunity tracking
        self._opportunities: List[FundingOpportunity] = []

        self.logger.info(
            f"Initialized FundingRateArbitrage with min_rate={min_funding_rate*100:.3f}%"
        )

    def initialize(self, **kwargs) -> None:
        """Initialize strategy with configuration."""
        super().initialize(**kwargs)

        if 'exchanges' in kwargs:
            self.exchanges = kwargs['exchanges']
        if 'symbols' in kwargs:
            self.symbols = kwargs['symbols']
        if 'min_funding_rate' in kwargs:
            self.min_funding_rate = kwargs['min_funding_rate']

    def update_funding_rate(
        self,
        exchange: str,
        symbol: str,
        funding_rate: float,
        next_funding_time: datetime,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Update funding rate data.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            funding_rate: Current funding rate
            next_funding_time: Next funding timestamp
            timestamp: Update timestamp
        """
        rate = FundingRate(
            exchange=exchange,
            symbol=symbol,
            funding_rate=funding_rate,
            next_funding_time=next_funding_time,
            timestamp=timestamp or datetime.now()
        )

        self._funding_rates[symbol][exchange] = rate

        # Store history
        self._funding_history[f"{symbol}_{exchange}"].append({
            'timestamp': rate.timestamp,
            'funding_rate': funding_rate,
            'predicted_rate': rate.predicted_rate
        })

        # Update prediction
        if self.use_prediction:
            self._update_prediction(symbol, exchange)

    def update_spot_price(
        self,
        exchange: str,
        symbol: str,
        price: float
    ) -> None:
        """Update spot price."""
        self._spot_prices[symbol][exchange] = price

    def update_perp_price(
        self,
        exchange: str,
        symbol: str,
        price: float
    ) -> None:
        """Update perpetual futures price."""
        self._perp_prices[symbol][exchange] = price

        # Calculate and store basis
        if exchange in self._spot_prices.get(symbol, {}):
            spot = self._spot_prices[symbol][exchange]
            basis = (price - spot) / spot
            self._basis_history[f"{symbol}_{exchange}"].append(basis)

    def _update_prediction(self, symbol: str, exchange: str) -> None:
        """Update funding rate prediction using EMA."""
        key = f"{symbol}_{exchange}"
        history = self._funding_history.get(key, [])

        if len(history) < 3:
            return

        # Simple EMA-based prediction
        rates = [h['funding_rate'] for h in history[-10:]]
        alpha = 0.3  # EMA smoothing factor

        predicted = rates[0]
        for rate in rates[1:]:
            predicted = alpha * rate + (1 - alpha) * predicted

        self._predicted_rates[key] = predicted

        # Update stored funding rate
        if symbol in self._funding_rates and exchange in self._funding_rates[symbol]:
            self._funding_rates[symbol][exchange].predicted_rate = predicted

    def detect_opportunities(self) -> List[FundingOpportunity]:
        """
        Detect funding rate arbitrage opportunities.

        Returns:
            List of FundingOpportunity objects
        """
        opportunities = []

        for symbol in self.symbols:
            # Single exchange cash-and-carry
            single_ex_opps = self._detect_cash_carry(symbol)
            opportunities.extend(single_ex_opps)

            # Cross-exchange funding arbitrage
            cross_ex_opps = self._detect_cross_exchange_funding(symbol)
            opportunities.extend(cross_ex_opps)

        # Filter by minimum return
        valid_opps = [
            opp for opp in opportunities
            if opp.annualized_return >= self.target_annual_return
        ]

        self._opportunities = valid_opps
        return valid_opps

    def _detect_cash_carry(self, symbol: str) -> List[FundingOpportunity]:
        """
        Detect cash-and-carry arbitrage opportunities.

        Long spot + short perpetual when funding is positive.
        Short spot + long perpetual when funding is negative.
        """
        opportunities = []

        for exchange in self.exchanges:
            # Get prices
            spot_price = self._spot_prices.get(symbol, {}).get(exchange, 0)
            perp_price = self._perp_prices.get(symbol, {}).get(exchange, 0)

            if spot_price == 0 or perp_price == 0:
                continue

            # Get funding rate
            funding_rate = 0
            if symbol in self._funding_rates and exchange in self._funding_rates[symbol]:
                funding_rate = self._funding_rates[symbol][exchange].funding_rate

            # Use predicted rate if available
            if self.use_prediction:
                key = f"{symbol}_{exchange}"
                if key in self._predicted_rates:
                    funding_rate = self._predicted_rates[key]

            # Calculate basis
            basis = perp_price - spot_price
            basis_pct = basis / spot_price

            # Cash-and-carry: funding positive, perp at premium
            if funding_rate > 0 and perp_price > spot_price:
                if funding_rate >= self.min_funding_rate:
                    annualized_return = funding_rate * 3 * 365  # 3 periods per day

                    opp = FundingOpportunity(
                        symbol=symbol,
                        position_type=FundingPositionType.CASH_CARRY,
                        exchanges=[exchange],
                        spot_price=spot_price,
                        perp_price=perp_price,
                        basis=basis,
                        basis_pct=basis_pct,
                        funding_rate=funding_rate,
                        annualized_return=annualized_return,
                        expected_profit=funding_rate * self.capital,
                        hold_period=self.max_hold_periods,
                        metadata={'type': 'cash_and_carry'}
                    )
                    opportunities.append(opp)

            # Reverse cash-and-carry: funding negative, perp at discount
            elif funding_rate < 0 and perp_price < spot_price:
                if abs(funding_rate) >= self.min_funding_rate:
                    annualized_return = abs(funding_rate) * 3 * 365

                    opp = FundingOpportunity(
                        symbol=symbol,
                        position_type=FundingPositionType.REVERSE_CASH_CARRY,
                        exchanges=[exchange],
                        spot_price=spot_price,
                        perp_price=perp_price,
                        basis=basis,
                        basis_pct=basis_pct,
                        funding_rate=funding_rate,
                        annualized_return=annualized_return,
                        expected_profit=abs(funding_rate) * self.capital,
                        hold_period=self.max_hold_periods,
                        metadata={'type': 'reverse_cash_and_carry'}
                    )
                    opportunities.append(opp)

        return opportunities

    def _detect_cross_exchange_funding(
        self,
        symbol: str
    ) -> List[FundingOpportunity]:
        """
        Detect cross-exchange funding rate arbitrage.

        Long perpetual on exchange with lower funding,
        short perpetual on exchange with higher funding.
        """
        opportunities = []

        if len(self.exchanges) < 2:
            return opportunities

        # Get funding rates across exchanges
        rates = {}
        for exchange in self.exchanges:
            if symbol in self._funding_rates and exchange in self._funding_rates[symbol]:
                rate = self._funding_rates[symbol][exchange].funding_rate
                if self.use_prediction:
                    key = f"{symbol}_{exchange}"
                    if key in self._predicted_rates:
                        rate = self._predicted_rates[key]
                rates[exchange] = rate

        if len(rates) < 2:
            return opportunities

        # Find exchange pair with largest spread
        exchanges = list(rates.keys())
        for i, ex1 in enumerate(exchanges):
            for ex2 in exchanges[i+1:]:
                rate_spread = abs(rates[ex1] - rates[ex2])

                if rate_spread >= self.min_funding_rate * 2:
                    # Determine direction
                    if rates[ex1] > rates[ex2]:
                        # Short perp on ex1, long perp on ex2
                        long_exchange = ex2
                        short_exchange = ex1
                    else:
                        long_exchange = ex1
                        short_exchange = ex2

                    # Get prices
                    long_price = self._perp_prices.get(symbol, {}).get(long_exchange, 0)
                    short_price = self._perp_prices.get(symbol, {}).get(short_exchange, 0)

                    if long_price == 0 or short_price == 0:
                        continue

                    annualized_return = rate_spread * 3 * 365

                    opp = FundingOpportunity(
                        symbol=symbol,
                        position_type=FundingPositionType.CROSS_EXCHANGE,
                        exchanges=[long_exchange, short_exchange],
                        spot_price=0,  # Not used
                        perp_price=(long_price + short_price) / 2,
                        basis=0,
                        basis_pct=0,
                        funding_rate=rate_spread,
                        annualized_return=annualized_return,
                        expected_profit=rate_spread * self.capital,
                        hold_period=self.max_hold_periods,
                        metadata={
                            'long_exchange': long_exchange,
                            'short_exchange': short_exchange,
                            'long_rate': rates[long_exchange],
                            'short_rate': rates[short_exchange]
                        }
                    )
                    opportunities.append(opp)

        return opportunities

    def generate_signals(
        self,
        data: pd.DataFrame,
        **kwargs
    ) -> List[Signal]:
        """
        Generate trading signals for funding arbitrage.

        Args:
            data: Market data
            **kwargs: Additional parameters

        Returns:
            List of Signal objects
        """
        if self.state != StrategyState.RUNNING:
            return []

        signals = []
        opportunities = self.detect_opportunities()

        for opp in opportunities:
            if opp.position_type == FundingPositionType.CASH_CARRY:
                # Long spot signal
                spot_signal = Signal(
                    symbol=f"{opp.symbol}_SPOT",
                    signal_type=SignalType.BUY,
                    price=opp.spot_price,
                    size=self.capital / opp.spot_price,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    exchange=opp.exchanges[0],
                    metadata={
                        'position_type': 'cash_carry_leg',
                        'paired_with': f"{opp.symbol}_PERP"
                    }
                )

                # Short perp signal
                perp_signal = Signal(
                    symbol=f"{opp.symbol}_PERP",
                    signal_type=SignalType.SELL,
                    price=opp.perp_price,
                    size=self.capital / opp.perp_price,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    exchange=opp.exchanges[0],
                    metadata={
                        'position_type': 'cash_carry_leg',
                        'paired_with': f"{opp.symbol}_SPOT",
                        'funding_rate': opp.funding_rate
                    }
                )

                signals.extend([spot_signal, perp_signal])

            elif opp.position_type == FundingPositionType.REVERSE_CASH_CARRY:
                # Short spot signal
                spot_signal = Signal(
                    symbol=f"{opp.symbol}_SPOT",
                    signal_type=SignalType.SELL,
                    price=opp.spot_price,
                    size=self.capital / opp.spot_price,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    exchange=opp.exchanges[0],
                    metadata={
                        'position_type': 'reverse_cash_carry_leg',
                        'paired_with': f"{opp.symbol}_PERP"
                    }
                )

                # Long perp signal
                perp_signal = Signal(
                    symbol=f"{opp.symbol}_PERP",
                    signal_type=SignalType.BUY,
                    price=opp.perp_price,
                    size=self.capital / opp.perp_price,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    exchange=opp.exchanges[0],
                    metadata={
                        'position_type': 'reverse_cash_carry_leg',
                        'paired_with': f"{opp.symbol}_SPOT",
                        'funding_rate': opp.funding_rate
                    }
                )

                signals.extend([spot_signal, perp_signal])

            elif opp.position_type == FundingPositionType.CROSS_EXCHANGE:
                long_ex = opp.metadata['long_exchange']
                short_ex = opp.metadata['short_exchange']

                # Long perp on exchange with lower funding
                long_signal = Signal(
                    symbol=f"{opp.symbol}_PERP",
                    signal_type=SignalType.BUY,
                    price=opp.perp_price,
                    size=self.capital / opp.perp_price,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    exchange=long_ex,
                    metadata={
                        'position_type': 'cross_exchange_funding_leg',
                        'paired_with': f"{opp.symbol}_PERP_{short_ex}"
                    }
                )

                # Short perp on exchange with higher funding
                short_signal = Signal(
                    symbol=f"{opp.symbol}_PERP",
                    signal_type=SignalType.SELL,
                    price=opp.perp_price,
                    size=self.capital / opp.perp_price,
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    exchange=short_ex,
                    metadata={
                        'position_type': 'cross_exchange_funding_leg',
                        'paired_with': f"{opp.symbol}_PERP_{long_ex}",
                        'funding_rate': opp.funding_rate
                    }
                )

                signals.extend([long_signal, short_signal])

        return signals

    def should_close_position(
        self,
        position: Position,
        current_data: pd.Series
    ) -> Optional[Signal]:
        """
        Determine if funding position should be closed.

        Close when:
        - Funding rate turns negative (for long perp)
        - Funding rate turns positive (for short perp)
        - Max hold time exceeded
        - Before funding payment if not profitable

        Args:
            position: Position to evaluate
            current_data: Current market data

        Returns:
            Close signal if should close
        """
        # Check hold period
        hold_time = (datetime.now() - position.entry_time).total_seconds()
        hold_periods = hold_time / (8 * 3600)  # 8h periods

        if hold_periods >= self.max_hold_periods:
            return Signal(
                symbol=position.symbol,
                signal_type=SignalType.CLOSE_LONG if position.side == 'long'
                           else SignalType.CLOSE_SHORT,
                price=current_data.get('close', 0),
                timestamp=datetime.now(),
                strategy_id=self.strategy_id,
                metadata={'reason': 'max_hold_periods_exceeded'}
            )

        # Check funding rate
        symbol = position.symbol.replace('_PERP', '').replace('_SPOT', '')
        exchange = position.metadata.get('exchange', self.exchanges[0])

        if symbol in self._funding_rates and exchange in self._funding_rates[symbol]:
            funding_rate = self._funding_rates[symbol][exchange].funding_rate

            # Close if funding turns against us
            if position.side == 'long' and funding_rate < 0:
                return Signal(
                    symbol=position.symbol,
                    signal_type=SignalType.CLOSE_LONG,
                    price=current_data.get('close', 0),
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    metadata={'reason': 'funding_turned_negative'}
                )

            if position.side == 'short' and funding_rate > 0:
                return Signal(
                    symbol=position.symbol,
                    signal_type=SignalType.CLOSE_SHORT,
                    price=current_data.get('close', 0),
                    timestamp=datetime.now(),
                    strategy_id=self.strategy_id,
                    metadata={'reason': 'funding_turned_positive'}
                )

        return None

    def get_funding_summary(self) -> Dict:
        """Get summary of funding rates across all exchanges."""
        summary = {}

        for symbol in self._funding_rates:
            summary[symbol] = {}
            for exchange in self._funding_rates[symbol]:
                rate = self._funding_rates[symbol][exchange]
                summary[symbol][exchange] = {
                    'funding_rate': rate.funding_rate,
                    'annualized': rate.annualized_rate,
                    'predicted': rate.predicted_rate,
                    'next_funding': rate.next_funding_time.isoformat()
                }

        return summary

    def get_best_opportunity(self) -> Optional[FundingOpportunity]:
        """Get the best current funding opportunity."""
        if not self._opportunities:
            return None

        return max(self._opportunities, key=lambda x: x.annualized_return)

    def calculate_expected_return(
        self,
        symbol: str,
        exchange: str,
        hold_periods: int
    ) -> float:
        """
        Calculate expected return for funding arbitrage.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            hold_periods: Number of funding periods to hold

        Returns:
            Expected return as fraction
        """
        if symbol not in self._funding_rates or exchange not in self._funding_rates[symbol]:
            return 0

        funding_rate = self._funding_rates[symbol][exchange].funding_rate

        if self.use_prediction:
            key = f"{symbol}_{exchange}"
            if key in self._predicted_rates:
                funding_rate = self._predicted_rates[key]

        # Expected return over hold period
        expected_return = abs(funding_rate) * hold_periods

        # Subtract transaction costs
        expected_return -= 2 * self.transaction_cost

        return expected_return


class FundingRatePredictor:
    """
    Predicts future funding rates using various models.
    """

    def __init__(
        self,
        lookback_periods: int = 30,
        prediction_horizon: int = 3
    ):
        """
        Initialize Funding Rate Predictor.

        Args:
            lookback_periods: Number of periods to look back
            prediction_horizon: Number of periods to predict ahead
        """
        self.lookback_periods = lookback_periods
        self.prediction_horizon = prediction_horizon
        self._models: Dict[str, Dict] = {}

    def fit(
        self,
        symbol: str,
        funding_history: List[Dict]
    ) -> None:
        """
        Fit prediction model for a symbol.

        Args:
            symbol: Trading symbol
            funding_history: List of funding rate history
        """
        if len(funding_history) < self.lookback_periods:
            return

        rates = [h['funding_rate'] for h in funding_history[-self.lookback_periods:]]

        # Calculate EMA
        alpha = 0.3
        ema = rates[0]
        for rate in rates[1:]:
            ema = alpha * rate + (1 - alpha) * ema

        # Calculate trend
        recent = rates[-5:]
        trend = (recent[-1] - recent[0]) / len(recent) if len(recent) > 1 else 0

        # Calculate volatility
        volatility = np.std(rates[-10:]) if len(rates) >= 10 else 0

        self._models[symbol] = {
            'ema': ema,
            'trend': trend,
            'volatility': volatility
        }

    def predict(
        self,
        symbol: str,
        horizon: Optional[int] = None
    ) -> Optional[float]:
        """
        Predict funding rate for a symbol.

        Args:
            symbol: Trading symbol
            horizon: Prediction horizon (periods ahead)

        Returns:
            Predicted funding rate
        """
        if symbol not in self._models:
            return None

        horizon = horizon or self.prediction_horizon
        model = self._models[symbol]

        # Simple linear extrapolation with EMA baseline
        predicted = model['ema'] + model['trend'] * horizon

        return predicted

    def get_confidence(
        self,
        symbol: str
    ) -> float:
        """
        Get prediction confidence (inverse of volatility).

        Args:
            symbol: Trading symbol

        Returns:
            Confidence score (0-1)
        """
        if symbol not in self._models:
            return 0

        volatility = self._models[symbol]['volatility']

        # Convert volatility to confidence
        # Higher volatility = lower confidence
        confidence = 1 / (1 + volatility * 100)

        return confidence


def create_funding_strategy(
    strategy_type: str = 'funding_arbitrage',
    **kwargs
) -> BaseStrategy:
    """
    Convenience function to create a funding strategy.

    Args:
        strategy_type: Type of funding strategy
        **kwargs: Strategy parameters

    Returns:
        Configured strategy instance
    """
    return FundingRateArbitrage(**kwargs)
