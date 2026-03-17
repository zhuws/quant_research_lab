"""
Execution Module for Quant Research Lab.

Provides live trading execution capabilities:
    - Exchange connectivity (Binance, Bybit)
    - Order management
    - Paper trading
    - Position tracking
    - Real-time execution

Components:
    - ExchangeGateway: Abstract interface for exchanges
    - BinanceGateway: Binance exchange implementation
    - BybitGateway: Bybit exchange implementation
    - PaperTrader: Paper trading for testing
    - OrderManager: Order lifecycle management
"""

from execution.exchange_gateway import ExchangeGateway, ExchangeConfig
from execution.binance_gateway import BinanceGateway
from execution.bybit_gateway import BybitGateway
from execution.paper_trader import PaperTrader, PaperTradingConfig
from execution.order_manager import OrderManager, OrderStatus

__all__ = [
    'ExchangeGateway',
    'ExchangeConfig',
    'BinanceGateway',
    'BybitGateway',
    'PaperTrader',
    'PaperTradingConfig',
    'OrderManager',
    'OrderStatus'
]
