# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quant Research Lab is a professional crypto quantitative trading research platform supporting:
- Market data collection (Binance/Bybit REST & WebSocket)
- Feature engineering (150+ features)
- Alpha research and factor discovery
- Machine learning prediction (LightGBM, XGBoost, Neural Networks)
- Reinforcement learning trading (PPO)
- Multi-strategy portfolio management
- Cross-exchange arbitrage
- Walk-forward validation and backtesting
- Live/paper trading

Primary symbol: ETHUSDT. Primary exchanges: Binance, Bybit.

## Commands

### Setup & Installation
```bash
./start.sh setup          # Full setup (check deps, install, setup-db)
./start.sh install        # Install Python dependencies
./start.sh setup-db       # Initialize MySQL database tables
```

### Running with Docker
```bash
docker-compose up -d mysql              # Start MySQL
docker-compose up -d quant-lab          # Start trading service
docker-compose up -d data-collector     # Start data collection
docker-compose up -d dashboard          # Start monitoring dashboard (port 8050)
```

### Main CLI Commands
```bash
python main.py download_data --symbol ETHUSDT --start-date 2024-01-01 --end-date 2024-12-31
python main.py build_features --symbol ETHUSDT
python main.py run_alpha_discovery --symbol ETHUSDT
python main.py train_models --symbol ETHUSDT --model-type lightgbm
python main.py train_rl --symbol ETHUSDT
python main.py run_backtest --symbol ETHUSDT --strategy momentum
python main.py run_walkforward --symbol ETHUSDT --strategy momentum
python main.py start_live_trading --symbol ETHUSDT --strategy momentum --paper
```

### Quick Start Script
```bash
./start.sh download      # Download historical data
./start.sh features      # Build features
./start.sh backtest      # Run backtest
./start.sh train-ml      # Train ML model
./start.sh live          # Start paper trading
./start.sh test          # Run basic tests
```

## Architecture

```
quant_research_lab/
├── main.py                 # CLI entry point (QuantResearchLab class)
├── config/config.yaml      # Central configuration
├── data/                   # Data collection & storage
│   ├── mysql_storage.py    # MySQL database manager (MySQLStorage class)
│   ├── market_data_collector.py  # Orchestrates all data collection
│   ├── binance_downloader.py / bybit_downloader.py  # Exchange-specific downloaders
│   ├── websocket_stream.py # Real-time WebSocket streaming
│   └── funding_rate_fetcher.py  # Funding rate data
├── features/               # Feature engineering pipeline
│   ├── feature_pipeline.py # Main orchestrator (FeaturePipeline class)
│   ├── technical_features.py    # ~70 indicators (RSI, MACD, etc.)
│   ├── volatility_features.py   # ~35 volatility indicators
│   ├── orderflow_features.py    # ~30 order flow features
│   ├── liquidity_features.py    # ~25 liquidity features
│   └── cross_exchange_features.py  # ~20 cross-exchange features
├── alpha_models/           # ML model training
│   ├── base_model.py       # Abstract BaseModel class
│   ├── tree_models.py      # LightGBM, XGBoost, CatBoost, RandomForest
│   ├── neural_models.py    # PyTorch neural networks
│   ├── model_trainer.py    # Training pipeline with CV
│   └── feature_selector.py # Feature selection utilities
├── strategies/             # Trading strategies
│   ├── base_strategy.py    # Abstract BaseStrategy with Signal/Position classes
│   ├── momentum_strategy.py
│   └── cross_exchange_arbitrage.py
├── rl_agents/              # Reinforcement learning
│   ├── trading_env.py      # Gymnasium trading environment
│   ├── ppo_agent.py        # PPO implementation
│   └── dqn_agent.py        # DQN implementation
├── backtest/               # Backtesting engine
│   ├── vectorized_engine.py   # Fast vectorized backtest
│   ├── walk_forward.py     # Walk-forward validation
│   └── execution_simulator.py # Order execution simulation
├── execution/              # Order execution
│   ├── exchange_gateway.py # Abstract ExchangeGateway interface
│   ├── binance_gateway.py / bybit_gateway.py  # Exchange implementations
│   ├── paper_trader.py     # Paper trading simulation
│   └── order_manager.py    # Order lifecycle management
├── risk/                   # Risk management
│   ├── risk_engine.py      # Central risk engine
│   ├── drawdown_control.py
│   └── exposure_limits.py
├── portfolio/              # Portfolio management
│   ├── portfolio_optimizer.py  # Mean-variance optimization
│   └── capital_allocator.py
├── research/               # Alpha research
│   ├── alpha_discovery.py  # Automated factor discovery
│   ├── factor_library.py   # Pre-built factors
│   └── factor_evaluator.py # IC, Sharpe, decay analysis
├── monitoring/             # Monitoring & alerting
│   ├── dashboard_server.py # Dash-based web dashboard
│   └── metrics_collector.py # Prometheus metrics
└── utils/                  # Utilities
    ├── logger.py           # Centralized logging
    ├── math_utils.py       # Safe divide, etc.
    └── time_utils.py       # UTC time handling
```

## Key Patterns

### Strategy Pattern
All strategies inherit from `BaseStrategy` and implement:
- `generate_signals(data: pd.DataFrame) -> List[Signal]`
- `should_close_position(position: Position, current_data: pd.Series) -> Optional[Signal]`

Use `StrategyRegistry` for dynamic strategy registration:
```python
@StrategyRegistry.register('my_strategy')
class MyStrategy(BaseStrategy):
    ...
```

### Exchange Gateway Pattern
All exchange implementations inherit from `ExchangeGateway` abstract class.
- Use `ExchangeConfig` dataclass for configuration
- Standardized `Order`, `Position`, `AccountInfo`, `Ticker` dataclasses
- All async methods: `connect()`, `place_order()`, `get_positions()`, etc.

### Model Pattern
All ML models inherit from `BaseModel` and implement:
- `fit(X, y, eval_set)` - Training
- `predict(X)` - Prediction
- `evaluate(X, y)` - Returns `ModelMetrics` (IC, Sharpe, RMSE, etc.)

Use `ModelTrainer` for cross-validation and hyperparameter optimization.

### Feature Pipeline
Use `FeaturePipeline.generate_features(df)` to generate 150+ features.
Call `generate_target_labels(df)` to add ML target columns.

## Database Schema (MySQL)

Tables are defined in `scripts/init.sql` and `data/mysql_storage.py:MySQLStorage.create_tables()`:
- `ohlcv` - OHLCV data (PK: exchange, symbol, timeframe, timestamp)
- `orderbook` - Orderbook snapshots with JSON bids/asks
- `trades` - Individual trades
- `funding_rates` - Funding rate history
- `features` - Computed feature values
- `signals` - Generated trading signals
- `orders` - Order history
- `positions` - Position snapshots
- `pnl` - P&L records

## Configuration

- `config/config.yaml` - Main configuration file
- `.env` - Environment variables (copy from `.env.example`)

**IMPORTANT**: API keys should be set via environment variables, not hardcoded in config files. Use `${VAR_NAME}` syntax in config.yaml to reference env vars.

Key environment variables:
- `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`
- `BINANCE_API_KEY`, `BINANCE_API_SECRET`
- `BYBIT_API_KEY`, `BYBIT_API_SECRET`
- `TRADING_MODE` (paper/live)
- `DEFAULT_SYMBOL`, `START_DATE`, `END_DATE` (for download scripts)

## Dependencies

Core: numpy, pandas, scipy, PyYAML
Database: pymysql, sqlalchemy
ML: scikit-learn, lightgbm, xgboost, torch
RL: gymnasium, stable-baselines3
Technical: ta-lib, pandas-ta
Exchange: ccxt, aiohttp, websockets
Visualization: matplotlib, plotly, dash
Monitoring: prometheus-client

## Testing

No dedicated unit test suite exists. Basic module validation via:
```bash
./start.sh test    # Tests logger, downloader, and websocket modules load correctly
```

## Code Conventions

- Module imports use `sys.path.insert(0, ...)` at file top for project root resolution
- All exchange methods are async (connect, place_order, get_positions, etc.)
- Strategies use `SignalType` enum for signal types (BUY, SELL, CLOSE_LONG, CLOSE_SHORT)
- Feature columns exclude: timestamp, open, high, low, close, volume, and target_* columns
