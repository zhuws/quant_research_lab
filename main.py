"""
Quant Research Lab - Main Entry Point
Professional Crypto Quantitative Trading Research Platform
"""

import argparse
import asyncio
import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import yaml

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logging, get_logger
from utils.time_utils import get_utc_now


class QuantResearchLab:
    """
    Main application class for Quant Research Lab.
    """

    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize Quant Research Lab.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = setup_logging(
            log_level=self.config.get('logging', {}).get('level', 'INFO'),
            log_file=self.config.get('logging', {}).get('file')
        )

        self._storage = None
        self._data_collector = None

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file with environment variable expansion."""
        import os
        import re

        try:
            with open(config_path, 'r') as f:
                content = f.read()

            # Expand environment variables with ${VAR:-default} syntax
            def expand_env(match):
                var_expr = match.group(1)
                if ':-' in var_expr:
                    var_name, default = var_expr.split(':-', 1)
                    return os.getenv(var_name, default)
                else:
                    return os.getenv(var_expr, match.group(0))

            content = re.sub(r'\$\{([^}]+)\}', expand_env, content)
            config = yaml.safe_load(content)
            return config or {}
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            return {}

    @property
    def storage(self):
        """Lazy-load storage."""
        if self._storage is None:
            from data.mysql_storage import MySQLStorage

            db_config = self.config.get('database', {})
            self._storage = MySQLStorage(
                host=db_config.get('host', 'localhost'),
                port=int(db_config.get('port', 3306)),
                user=db_config.get('user', 'quant_user'),
                password=db_config.get('password', 'quant_password'),
                database=db_config.get('database', 'quant_research')
            )
        return self._storage

    def download_data(
        self,
        symbols: Optional[List[str]] = None,
        exchanges: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> None:
        """
        Download historical market data.

        Args:
            symbols: List of trading symbols
            exchanges: List of exchanges
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        from data.market_data_collector import MarketDataCollector

        symbols = symbols or [self.config.get('symbol', 'ETHUSDT')]
        exchanges = exchanges or self.config.get('exchanges', ['binance', 'bybit'])

        start = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc) if start_date else get_utc_now() - timedelta(days=30)
        end = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc) if end_date else get_utc_now()

        self.logger.info(f"Downloading data for {symbols} from {exchanges}")
        self.logger.info(f"Date range: {start} to {end}")

        # Get API keys from environment
        import os
        binance_key = os.getenv('BINANCE_API_KEY')
        binance_secret = os.getenv('BINANCE_API_SECRET')
        bybit_key = os.getenv('BYBIT_API_KEY')
        bybit_secret = os.getenv('BYBIT_API_SECRET')

        collector = MarketDataCollector(
            storage=self.storage,
            symbols=symbols,
            exchanges=exchanges,
            timeframes=self.config.get('data_collection', {}).get('timeframes', ['1m']),
            binance_api_key=binance_key,
            binance_api_secret=binance_secret,
            bybit_api_key=bybit_key,
            bybit_api_secret=bybit_secret
        )

        # Create tables if needed
        self.storage.connect()
        self.storage.create_tables()

        # Download data
        results = collector.download_all_historical(start_date=start, end_date=end)

        # Print summary
        for symbol, exchange_data in results.items():
            for key, df in exchange_data.items():
                if not df.empty:
                    self.logger.info(f"  {symbol} {key}: {len(df)} records")

        collector.close()
        self.storage.disconnect()

    def build_features(
        self,
        symbol: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> None:
        """
        Build features from market data.

        Args:
            symbol: Trading symbol
            output_file: Output file path
        """
        symbol = symbol or self.config.get('symbol', 'ETHUSDT')

        self.logger.info(f"Building features for {symbol}")

        # Import feature modules
        from features.feature_pipeline import FeaturePipeline

        self.storage.connect()

        # Get OHLCV data
        for exchange in self.config.get('exchanges', ['binance']):
            df = self.storage.get_ohlcv(
                exchange=exchange,
                symbol=symbol,
                timeframe='1m'
            )

            if df.empty:
                self.logger.warning(f"No data found for {symbol} on {exchange}")
                continue

            # Build features
            pipeline = FeaturePipeline()
            features_df = pipeline.generate_features(df)

            self.logger.info(f"Generated {len(features_df.columns)} features")

            if output_file:
                features_df.to_csv(output_file, index=False)
                self.logger.info(f"Saved features to {output_file}")

        self.storage.disconnect()

    def run_alpha_discovery(
        self,
        symbol: Optional[str] = None,
        n_factors: int = 100
    ) -> None:
        """
        Run alpha discovery process.

        Args:
            symbol: Trading symbol
            n_factors: Number of factors to generate
        """
        symbol = symbol or self.config.get('symbol', 'ETHUSDT')

        self.logger.info(f"Running alpha discovery for {symbol}")

        from research.alpha_discovery import AlphaDiscovery
        from features.feature_pipeline import FeaturePipeline

        self.storage.connect()

        # Get data
        df = self.storage.get_ohlcv(
            exchange=self.config.get('exchanges', ['binance'])[0],
            symbol=symbol,
            timeframe='1m'
        )

        if df.empty:
            self.logger.error("No data available for alpha discovery")
            return

        # Build features
        pipeline = FeaturePipeline()
        features_df = pipeline.generate_features(df)

        # Run discovery
        discovery = AlphaDiscovery()
        results = discovery.run(features_df, n_factors=n_factors)

        self.logger.info(f"Discovered {len(results['factors'])} alpha factors")

        # Print top factors
        for factor in results['factors'][:10]:
            self.logger.info(
                f"  {factor['name']}: IC={factor['ic']:.4f}, "
                f"Sharpe={factor['sharpe']:.2f}"
            )

        self.storage.disconnect()

    def train_models(
        self,
        symbol: Optional[str] = None,
        model_type: str = 'lightgbm',
        limit: int = 100000
    ) -> None:
        """
        Train ML models.

        Args:
            symbol: Trading symbol
            model_type: Model type (lightgbm, xgboost, neural_network)
            limit: Maximum number of bars to use for training
        """
        symbol = symbol or self.config.get('symbol', 'ETHUSDT')

        self.logger.info(f"Training {model_type} model for {symbol}")

        from features.feature_pipeline import FeaturePipeline
        from alpha_models.model_trainer import ModelTrainer
        from alpha_models.tree_models import LightGBMModel, XGBoostModel

        # Map model type to model class
        model_classes = {
            'lightgbm': LightGBMModel,
            'xgboost': XGBoostModel
        }

        if model_type not in model_classes:
            self.logger.error(f"Unknown model type: {model_type}")
            return

        self.storage.connect()

        # Get data (limited to most recent bars)
        df = self.storage.get_ohlcv(
            exchange=self.config.get('exchanges', ['binance'])[0],
            symbol=symbol,
            timeframe='1m',
            limit=limit
        )

        if df.empty:
            self.logger.error("No data available for training")
            return

        self.logger.info(f"Using {len(df)} bars for training")

        # Build features
        pipeline = FeaturePipeline()
        features_df = pipeline.generate_features(df)
        features_df = pipeline.generate_target_labels(features_df)

        # Train model
        trainer = ModelTrainer(model_class=model_classes[model_type])
        results = trainer.train(features_df)

        self.logger.info(f"Model trained: IC={results.metrics.ic:.4f}, Sharpe={results.metrics.sharpe:.4f}")

        self.storage.disconnect()

    def train_rl(
        self,
        symbol: Optional[str] = None,
        total_timesteps: int = 100000
    ) -> None:
        """
        Train reinforcement learning agent.

        Args:
            symbol: Trading symbol
            total_timesteps: Total training timesteps
        """
        symbol = symbol or self.config.get('symbol', 'ETHUSDT')

        self.logger.info(f"Training RL agent for {symbol}")

        from rl.train_rl_agent import train_rl_agent

        self.storage.connect()

        # Get data
        df = self.storage.get_ohlcv(
            exchange=self.config.get('exchanges', ['binance'])[0],
            symbol=symbol,
            timeframe='1m'
        )

        if df.empty:
            self.logger.error("No data available for RL training")
            return

        # Train agent
        results = train_rl_agent(df, total_timesteps=total_timesteps)

        self.logger.info(f"RL training complete: {results}")

        self.storage.disconnect()

    def run_backtest(
        self,
        symbol: Optional[str] = None,
        strategy: str = 'momentum',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> None:
        """
        Run backtest.

        Args:
            symbol: Trading symbol
            strategy: Strategy name
            start_date: Start date
            end_date: End date
        """
        symbol = symbol or self.config.get('symbol', 'ETHUSDT')

        self.logger.info(f"Running {strategy} backtest for {symbol}")

        from backtest.vectorized_engine import VectorizedBacktestEngine
        from strategies.momentum_strategy import MomentumStrategy

        self.storage.connect()

        # Get data
        df = self.storage.get_ohlcv(
            exchange=self.config.get('exchanges', ['binance'])[0],
            symbol=symbol,
            timeframe='1m'
        )

        if df.empty:
            self.logger.error("No data available for backtest")
            return

        # Filter by date
        if start_date:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            df = df[df['timestamp'] >= start]
        if end_date:
            end = datetime.strptime(end_date, '%Y-%m-%d')
            df = df[df['timestamp'] <= end]

        # Run backtest
        engine = VectorizedBacktestEngine(
            initial_capital=self.config.get('backtest', {}).get('initial_capital', 10000),
            fee=self.config.get('fee', 0.001)
        )

        strategy_obj = MomentumStrategy()
        results = engine.run(df, strategy_obj)

        self.logger.info(f"Backtest results: {results['summary']}")

        self.storage.disconnect()

    def run_walkforward(
        self,
        symbol: Optional[str] = None,
        strategy: str = 'momentum'
    ) -> None:
        """
        Run walk-forward validation.

        Args:
            symbol: Trading symbol
            strategy: Strategy name
        """
        symbol = symbol or self.config.get('symbol', 'ETHUSDT')

        self.logger.info(f"Running walk-forward validation for {symbol}")

        from backtest.walk_forward import WalkForwardValidator
        from strategies.momentum_strategy import MomentumStrategy

        self.storage.connect()

        # Get data
        df = self.storage.get_ohlcv(
            exchange=self.config.get('exchanges', ['binance'])[0],
            symbol=symbol,
            timeframe='1m'
        )

        if df.empty:
            self.logger.error("No data available for walk-forward")
            return

        # Run walk-forward
        validator = WalkForwardValidator()
        strategy_obj = MomentumStrategy()
        results = validator.run(df, strategy_obj)

        self.logger.info(f"Walk-forward results: {results}")

        self.storage.disconnect()

    def start_live_trading(
        self,
        symbol: Optional[str] = None,
        strategy: str = 'momentum',
        paper: bool = True
    ) -> None:
        """
        Start live trading.

        Args:
            symbol: Trading symbol
            strategy: Strategy name
            paper: Use paper trading mode
        """
        symbol = symbol or self.config.get('symbol', 'ETHUSDT')

        mode = "PAPER" if paper else "LIVE"
        self.logger.info(f"Starting {mode} trading for {symbol} with {strategy}")

        from execution.paper_trader import PaperTrader, PaperTradingConfig
        from strategies.momentum_strategy import MomentumStrategy
        from risk.risk_engine import RiskEngine
        from data.websocket_stream import MultiExchangeStreamManager

        # Initialize paper trader for paper mode
        paper_config = PaperTradingConfig(
            initial_capital=self.config.get('backtest', {}).get('initial_capital', 100000),
            commission_rate=self.config.get('fee', 0.001),
            slippage_rate=0.0001
        )
        paper_trader = PaperTrader(paper_config)

        risk_engine = RiskEngine(
            max_position=self.config.get('max_position', 1.0),
            max_drawdown=self.config.get('risk', {}).get('max_drawdown', 0.2)
        )
        strategy_obj = MomentumStrategy()

        # Start WebSocket streams
        async def run_live():
            await paper_trader.connect()

            stream_manager = MultiExchangeStreamManager(
                symbols=[symbol],
                exchanges=self.config.get('exchanges', ['binance'])
            )

            def on_message(msg):
                # Update market price for paper trader
                if hasattr(msg, 'data') and isinstance(msg.data, dict):
                    price = msg.data.get('price') or msg.data.get('close') or msg.data.get('last')
                    if price:
                        paper_trader.update_market_price(symbol, float(price))

                # Process market data
                signal = strategy_obj.generate_signal(msg.data if hasattr(msg, 'data') else {})
                if signal:
                    # Risk check - convert Signal to dict for risk engine
                    signal_dict = signal.to_dict()
                    allowed, reason = risk_engine.check_order(symbol, signal_dict)
                    if allowed:
                        # Place order through paper trader
                        order = paper_trader.place_order(
                            symbol=symbol,
                            side=signal.signal_type.value,
                            order_type='MARKET',
                            quantity=0.001  # Small test quantity
                        )
                        self.logger.info(f"Paper order placed: {order.order_id} - {order.status.value}")

            stream_manager.add_message_handler(on_message)
            await stream_manager.start_all()

        asyncio.run(run_live())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Quant Research Lab - Crypto Quantitative Trading Platform'
    )

    parser.add_argument(
        'command',
        choices=[
            'download_data',
            'build_features',
            'run_alpha_discovery',
            'train_models',
            'train_rl',
            'run_backtest',
            'run_walkforward',
            'start_live_trading'
        ],
        help='Command to execute'
    )

    parser.add_argument(
        '--config', '-c',
        default='config/config.yaml',
        help='Configuration file path'
    )

    parser.add_argument(
        '--symbol', '-s',
        help='Trading symbol'
    )

    parser.add_argument(
        '--exchange', '-e',
        action='append',
        help='Exchange (can be specified multiple times)'
    )

    parser.add_argument(
        '--start-date',
        help='Start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end-date',
        help='End date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--strategy',
        default='momentum',
        help='Strategy name'
    )

    parser.add_argument(
        '--model-type',
        default='lightgbm',
        help='Model type for training'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=100000,
        help='Maximum bars to use for training'
    )

    parser.add_argument(
        '--paper',
        action='store_true',
        help='Use paper trading mode'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output file path'
    )

    args = parser.parse_args()

    # Initialize lab
    lab = QuantResearchLab(config_path=args.config)

    # Execute command
    if args.command == 'download_data':
        lab.download_data(
            symbols=[args.symbol] if args.symbol else None,
            exchanges=args.exchange,
            start_date=args.start_date,
            end_date=args.end_date
        )

    elif args.command == 'build_features':
        lab.build_features(
            symbol=args.symbol,
            output_file=args.output
        )

    elif args.command == 'run_alpha_discovery':
        lab.run_alpha_discovery(symbol=args.symbol)

    elif args.command == 'train_models':
        lab.train_models(
            symbol=args.symbol,
            model_type=args.model_type,
            limit=args.limit
        )

    elif args.command == 'train_rl':
        lab.train_rl(symbol=args.symbol)

    elif args.command == 'run_backtest':
        lab.run_backtest(
            symbol=args.symbol,
            strategy=args.strategy,
            start_date=args.start_date,
            end_date=args.end_date
        )

    elif args.command == 'run_walkforward':
        lab.run_walkforward(
            symbol=args.symbol,
            strategy=args.strategy
        )

    elif args.command == 'start_live_trading':
        lab.start_live_trading(
            symbol=args.symbol,
            strategy=args.strategy,
            paper=args.paper
        )


if __name__ == '__main__':
    main()
