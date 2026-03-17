-- Quant Research Lab Database Initialization
-- Run this script to initialize the database

CREATE DATABASE IF NOT EXISTS quant_research;
USE quant_research;

-- OHLCV data table
CREATE TABLE IF NOT EXISTS ohlcv (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp DATETIME(3) NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(30, 8) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_ohlcv (exchange, symbol, timeframe, timestamp),
    INDEX idx_timestamp (timestamp),
    INDEX idx_symbol_time (symbol, timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Orderbook snapshots
CREATE TABLE IF NOT EXISTS orderbook (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timestamp DATETIME(3) NOT NULL,
    bids JSON NOT NULL,
    asks JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_symbol_time (symbol, timestamp),
    INDEX idx_exchange_time (exchange, timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Trades
CREATE TABLE IF NOT EXISTS trades (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    trade_id VARCHAR(50) NOT NULL,
    timestamp DATETIME(3) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(30, 8) NOT NULL,
    side ENUM('buy', 'sell') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_trade (exchange, trade_id),
    INDEX idx_symbol_time (symbol, timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Funding rates
CREATE TABLE IF NOT EXISTS funding_rates (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timestamp DATETIME(3) NOT NULL,
    funding_rate DECIMAL(10, 8) NOT NULL,
    funding_time DATETIME(3) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_funding (exchange, symbol, timestamp),
    INDEX idx_symbol_time (symbol, timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Trading signals
CREATE TABLE IF NOT EXISTS signals (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    strategy VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timestamp DATETIME(3) NOT NULL,
    `signal` INT NOT NULL,
    confidence DECIMAL(5, 4),
    price DECIMAL(20, 8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_strategy_time (strategy, timestamp),
    INDEX idx_symbol_time (symbol, timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Orders
CREATE TABLE IF NOT EXISTS orders (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    exchange VARCHAR(20) NOT NULL,
    order_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side ENUM('buy', 'sell') NOT NULL,
    order_type ENUM('market', 'limit', 'stop', 'stop_limit') NOT NULL,
    quantity DECIMAL(30, 8) NOT NULL,
    price DECIMAL(20, 8),
    status ENUM('pending', 'open', 'filled', 'cancelled', 'rejected') NOT NULL,
    filled_quantity DECIMAL(30, 8) DEFAULT 0,
    average_price DECIMAL(20, 8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_order (exchange, order_id),
    INDEX idx_symbol_time (symbol, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Positions
CREATE TABLE IF NOT EXISTS positions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side ENUM('long', 'short') NOT NULL,
    quantity DECIMAL(30, 8) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),
    leverage DECIMAL(10, 2) DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_position (exchange, symbol),
    INDEX idx_symbol (symbol)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- PnL records
CREATE TABLE IF NOT EXISTS pnl (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    strategy VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timestamp DATETIME(3) NOT NULL,
    realized_pnl DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),
    total_pnl DECIMAL(20, 8),
    drawdown DECIMAL(10, 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_strategy_time (strategy, timestamp),
    INDEX idx_symbol_time (symbol, timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Grant permissions
GRANT ALL PRIVILEGES ON quant_research.* TO 'bot2'@'%';
FLUSH PRIVILEGES;
