"""
Logging utility module for Quant Research Lab.
Provides centralized logging configuration and management.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional
from datetime import datetime


class QuantLogger:
    """
    Centralized logger for the Quant Research Lab.

    Provides consistent logging across all modules with file rotation
    and configurable output formats.
    """

    _instances: dict = {}

    def __new__(cls, name: str = "quant_lab", **kwargs) -> "QuantLogger":
        """Singleton pattern for logger instances."""
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
            cls._instances[name]._initialized = False
        return cls._instances[name]

    def __init__(
        self,
        name: str = "quant_lab",
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        max_size_mb: int = 100,
        backup_count: int = 5
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name for identification
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            max_size_mb: Maximum log file size in MB before rotation
            backup_count: Number of backup files to keep
        """
        if self._initialized:
            return

        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Clear any existing handlers
        self.logger.handlers = []

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler with rotation
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_size_mb * 1024 * 1024,
                backupCount=backup_count
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self._initialized = True

    def debug(self, msg: str) -> None:
        """Log debug message."""
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        """Log info message."""
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log error message."""
        self.logger.error(msg)

    def critical(self, msg: str) -> None:
        """Log critical message."""
        self.logger.critical(msg)

    def exception(self, msg: str) -> None:
        """Log exception with traceback."""
        self.logger.exception(msg)


def get_logger(
    name: str = "quant_lab",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> QuantLogger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level

    Returns:
        QuantLogger instance
    """
    return QuantLogger(name=name, level=level, log_file=log_file)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_size_mb: int = 100,
    backup_count: int = 5
) -> QuantLogger:
    """
    Setup logging configuration from parameters.

    Args:
        log_level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        max_size_mb: Maximum log file size in MB
        backup_count: Number of backup files to keep

    Returns:
        Configured QuantLogger instance
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }

    level = level_map.get(log_level.upper(), logging.INFO)

    return get_logger(
        name="quant_lab",
        log_file=log_file,
        level=level
    )


class PerformanceLogger:
    """
    Logger for tracking performance metrics.

    Records timing and performance data for analysis.
    """

    def __init__(self, name: str = "performance"):
        """
        Initialize performance logger.

        Args:
            name: Name for the performance log
        """
        self.logger = get_logger(name)
        self._start_times: dict = {}

    def start(self, operation: str) -> None:
        """
        Start timing an operation.

        Args:
            operation: Name of the operation to time
        """
        self._start_times[operation] = datetime.now()
        self.logger.debug(f"Started: {operation}")

    def end(self, operation: str) -> float:
        """
        End timing an operation and log duration.

        Args:
            operation: Name of the operation

        Returns:
            Duration in seconds
        """
        if operation not in self._start_times:
            self.logger.warning(f"No start time for: {operation}")
            return 0.0

        duration = (datetime.now() - self._start_times[operation]).total_seconds()
        del self._start_times[operation]

        self.logger.info(f"Completed: {operation} in {duration:.3f}s")
        return duration

    def log_metric(self, name: str, value: float, unit: str = "") -> None:
        """
        Log a performance metric.

        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
        """
        self.logger.info(f"Metric - {name}: {value:.6f} {unit}")


class TradingLogger:
    """
    Specialized logger for trading operations.

    Provides structured logging for trades, orders, and signals.
    """

    def __init__(self, name: str = "trading"):
        """
        Initialize trading logger.

        Args:
            name: Name for the trading log
        """
        self.logger = get_logger(name)

    def log_signal(
        self,
        strategy: str,
        symbol: str,
        signal: int,
        price: float,
        confidence: float = 1.0
    ) -> None:
        """
        Log a trading signal.

        Args:
            strategy: Strategy name
            symbol: Trading symbol
            signal: Signal value (-1, 0, 1)
            price: Price at signal
            confidence: Signal confidence (0-1)
        """
        self.logger.info(
            f"SIGNAL | {strategy} | {symbol} | {signal} | "
            f"price={price:.4f} | confidence={confidence:.2f}"
        )

    def log_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        status: str
    ) -> None:
        """
        Log an order event.

        Args:
            order_id: Order identifier
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            price: Order price
            status: Order status
        """
        self.logger.info(
            f"ORDER | {order_id} | {symbol} | {side} | "
            f"qty={quantity:.6f} | price={price:.4f} | {status}"
        )

    def log_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        pnl: float = 0.0
    ) -> None:
        """
        Log a completed trade.

        Args:
            trade_id: Trade identifier
            symbol: Trading symbol
            side: Trade side (BUY/SELL)
            quantity: Trade quantity
            price: Execution price
            pnl: Profit/loss from trade
        """
        self.logger.info(
            f"TRADE | {trade_id} | {symbol} | {side} | "
            f"qty={quantity:.6f} | price={price:.4f} | pnl={pnl:.2f}"
        )

    def log_position(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        current_price: float,
        unrealized_pnl: float
    ) -> None:
        """
        Log position update.

        Args:
            symbol: Trading symbol
            quantity: Position quantity
            entry_price: Entry price
            current_price: Current price
            unrealized_pnl: Unrealized profit/loss
        """
        self.logger.info(
            f"POSITION | {symbol} | qty={quantity:.6f} | "
            f"entry={entry_price:.4f} | current={current_price:.4f} | "
            f"upl={unrealized_pnl:.2f}"
        )
