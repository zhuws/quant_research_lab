# Quant Research Lab Specification

This document defines the architecture and implementation requirements for a **professional crypto quantitative trading research platform**.

Claude Code should use this document as the **primary system specification** and generate the full implementation.

Target project size: **12,000 – 15,000 lines of Python code**

Target asset:

ETHUSDT

Primary exchange:

Binance

Secondary exchange:

Bybit

---

# 1 SYSTEM GOALS

The system must support:

1. Market data collection
2. Feature engineering
3. Alpha research
4. Machine learning prediction
5. Reinforcement learning trading
6. Multi-strategy portfolio
7. Cross-exchange arbitrage
8. Professional backtesting
9. Walk-forward validation
10. Live trading
11. Risk management
12. Monitoring dashboard

---

# 2 PROJECT STRUCTURE

Generate the following project structure.

quant_research_lab/

data/
market_data_collector.py
binance_downloader.py
bybit_downloader.py
websocket_stream.py
orderbook_recorder.py
trades_collector.py
funding_rate_fetcher.py
mysql_storage.py

features/
feature_pipeline.py
technical_features.py
orderflow_features.py
volatility_features.py
liquidity_features.py
cross_exchange_features.py

research/
factor_library.py
alpha_discovery.py
factor_evaluator.py
regime_detection.py

alpha_models/
lightgbm_model.py
xgboost_model.py
neural_network_model.py
alpha_trainer.py
feature_selection.py

strategies/
momentum_strategy.py
mean_reversion_strategy.py
volatility_breakout_strategy.py
ai_signal_strategy.py
orderflow_strategy.py
cross_exchange_arbitrage.py

rl/
trading_env.py
ppo_agent.py
train_rl_agent.py

portfolio/
portfolio_optimizer.py
strategy_allocator.py
capital_allocator.py

execution/
exchange_gateway.py
order_manager.py
smart_execution.py
twap_vwap.py

risk/
risk_engine.py
drawdown_control.py
exposure_limits.py
volatility_filter.py

backtest/
vectorized_engine.py
event_engine.py
walk_forward.py
performance_metrics.py

alpha_lab/
genetic_alpha_search.py
feature_importance_analysis.py
factor_decay_analysis.py

monitoring/
metrics_collector.py
trading_dashboard.py
performance_report.py

utils/
logger.py
math_utils.py
data_utils.py
time_utils.py

config/
config.yaml

main.py

---

# 3 DATA INFRASTRUCTURE

The system must collect market data from:

Binance REST API
Binance WebSocket
Bybit REST API
Bybit WebSocket

Data types:

OHLCV
OrderBook snapshots
Trades
Funding Rate

Store data in MySQL.

Required tables:

ohlcv
orderbook
trades
funding_rates
features
signals
orders
positions
pnl

---

# 4 FEATURE ENGINEERING

Build a feature pipeline producing **150+ features**.

Feature categories:

Momentum
Volatility
Volume
OrderBook imbalance
Liquidity pressure
Trade flow
Cross-exchange price spread

Example features:

momentum_20 = close / close.shift(20)

volatility_20 = std(returns_20)

orderbook_imbalance =
(bid_volume - ask_volume) /
(bid_volume + ask_volume)

micro_price =
(ask_price * bid_volume +
bid_price * ask_volume) /
(bid_volume + ask_volume)

cross_exchange_spread =
binance_price - bybit_price

---

# 5 ALPHA RESEARCH

Create an automated alpha research system.

Capabilities:

generate candidate factors
evaluate factors
rank factors

Metrics:

Information Coefficient (IC)
Sharpe ratio
Factor stability
Alpha decay

---

# 6 MACHINE LEARNING MODELS

Implement ML alpha models.

Models required:

LightGBM
XGBoost
Neural Network (PyTorch)

Target label:

future_return =
log(close[t+10] / close[t])

Training pipeline:

dataset builder
feature selection
cross validation
model comparison
model persistence

---

# 7 REINFORCEMENT LEARNING

Use PPO algorithm.

Build a custom trading environment.

State space:

features
alpha prediction
position
balance
volatility
market regime

Actions:

0 hold
1 long
2 short

Reward:

reward =
pnl

* 0.1 * drawdown
* 0.01 * volatility
* 0.001 * turnover

---

# 8 STRATEGY LIBRARY

Implement strategies:

Momentum Strategy
Mean Reversion Strategy
Volatility Breakout Strategy
AI Signal Strategy
OrderFlow Strategy
Cross Exchange Arbitrage

Each strategy outputs trading signals.

---

# 9 PORTFOLIO OPTIMIZATION

Implement portfolio optimization.

Methods:

Mean-variance optimization
Risk parity
Sharpe maximization

Allocate capital across strategies.

---

# 10 BACKTEST ENGINE

Create a professional backtesting engine.

Modes:

Vectorized backtest
Event-driven backtest

Features:

multi-strategy simulation
transaction costs
slippage modeling
order execution simulation

---

# 11 WALK-FORWARD VALIDATION

Implement walk-forward testing.

Example:

train 2019–2022
test 2023

train 2020–2023
test 2024

---

# 12 EXECUTION ENGINE

Execution methods:

market orders
limit orders
TWAP
VWAP

Exchange integration via CCXT.

---

# 13 RISK MANAGEMENT

Implement a full risk engine.

Controls:

max position size
max drawdown
daily loss limit
volatility filter
exchange exposure limits

---

# 14 MONITORING

Implement monitoring modules.

Track:

PnL
positions
drawdown
latency
strategy performance

---

# 15 CONFIGURATION

Create config.yaml.

Example configuration:

symbol: ETHUSDT
exchanges: [binance, bybit]
timeframe: 1m
trade_size: 0.1
max_position: 1
fee: 0.001

---

# 16 MAIN PROGRAM

main.py must support the following commands:

download_data
build_features
run_alpha_discovery
train_models
train_rl
run_backtest
run_walkforward
start_live_trading

---

# 17 CODE QUALITY REQUIREMENTS

All code must include:

type hints
docstrings
logging
exception handling
modular architecture

---

# 18 OUTPUT REQUIREMENT

Claude Code must generate the **entire project implementation** according to this specification.

Total expected project size:

12000–15000 lines of Python code.

The system must be capable of:

alpha research
machine learning training
reinforcement learning trading
multi-strategy portfolio management
live crypto trading

