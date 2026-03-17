#!/bin/bash
# Quant Research Lab - Quick Start Script
# Usage: ./start.sh [command]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo -e "${GREEN}[OK] Loaded .env file${NC}"
else
    echo -e "${YELLOW}[WARN] .env file not found, using defaults${NC}"
fi

# Function: Check dependencies
check_dependencies() {
    echo -e "\n${YELLOW}Checking dependencies...${NC}"

    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}[ERROR] Python3 not installed${NC}"
        exit 1
    fi
    echo -e "${GREEN}[OK] Python3: $(python3 --version)${NC}"

    # Check pip
    if ! command -v pip3 &> /dev/null; then
        echo -e "${RED}[ERROR] pip3 not installed${NC}"
        exit 1
    fi
    echo -e "${GREEN}[OK] pip3 available${NC}"

    # Check MySQL
    if command -v mysql &> /dev/null; then
        echo -e "${GREEN}[OK] MySQL client available${NC}"
    else
        echo -e "${YELLOW}[WARN] MySQL client not found${NC}"
    fi
}

# Function: Install dependencies
install() {
    echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
    pip3 install -r requirements.txt
    echo -e "${GREEN}[OK] Dependencies installed${NC}"
}

# Function: Setup database
setup_db() {
    echo -e "\n${YELLOW}Setting up database...${NC}"

    # Create database if not exists
    mysql -u"${MYSQL_USER:-bot2}" -p"${MYSQL_PASSWORD:-bot2_123456}" \
        -e "CREATE DATABASE IF NOT EXISTS ${MYSQL_DATABASE:-quant_research};" 2>/dev/null || {
        echo -e "${YELLOW}[WARN] Could not create database (may already exist or MySQL not running)${NC}"
    }

    # Create tables
    python3 -c "
from data.mysql_storage import MySQLStorage
storage = MySQLStorage(
    host='${MYSQL_HOST:-localhost}',
    port=${MYSQL_PORT:-3306},
    user='${MYSQL_USER:-bot2}',
    password='${MYSQL_PASSWORD:-bot2_123456}',
    database='${MYSQL_DATABASE:-quant_research}'
)
storage.connect()
storage.create_tables()
storage.disconnect()
print('Database tables created successfully!')
" 2>/dev/null && echo -e "${GREEN}[OK] Database setup complete${NC}" || \
    echo -e "${YELLOW}[WARN] Database setup skipped (MySQL may not be running)${NC}"
}

# Function: Download historical data
download_data() {
    echo -e "\n${YELLOW}Downloading historical data...${NC}"
    python3 main.py download_data \
        --symbol "${DEFAULT_SYMBOL:-ETHUSDT}" \
        --start-date "${START_DATE:-2024-01-01}" \
        --end-date "${END_DATE:-2024-12-31}"
    echo -e "${GREEN}[OK] Data download complete${NC}"
}

# Function: Build features
build_features() {
    echo -e "\n${YELLOW}Building features...${NC}"
    python3 main.py build_features --symbol "${DEFAULT_SYMBOL:-ETHUSDT}"
    echo -e "${GREEN}[OK] Feature building complete${NC}"
}

# Function: Run backtest
backtest() {
    echo -e "\n${YELLOW}Running backtest...${NC}"
    python3 main.py run_backtest \
        --symbol "${DEFAULT_SYMBOL:-ETHUSDT}" \
        --strategy momentum
    echo -e "${GREEN}[OK] Backtest complete${NC}"
}

# Function: Train ML model
train_ml() {
    echo -e "\n${YELLOW}Training ML model...${NC}"
    python3 main.py train_models \
        --symbol "${DEFAULT_SYMBOL:-ETHUSDT}" \
        --model-type lightgbm
    echo -e "${GREEN}[OK] ML training complete${NC}"
}

# Function: Start live trading (paper mode)
live_trading() {
    echo -e "\n${YELLOW}Starting live trading (paper mode)...${NC}"
    python3 main.py start_live_trading \
        --symbol "${DEFAULT_SYMBOL:-ETHUSDT}" \
        --strategy momentum \
        --paper
}

# Function: Run all setup steps
setup_all() {
    check_dependencies
    install
    setup_db
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Setup complete! Run './start.sh test' to verify.${NC}"
    echo -e "${GREEN}========================================${NC}"
}

# Function: Run tests
run_tests() {
    echo -e "\n${YELLOW}Running basic tests...${NC}"

    echo "Testing logger..."
    python3 -c "from utils.logger import get_logger; logger = get_logger('test'); logger.info('OK')"

    echo "Testing Binance downloader..."
    python3 -c "from data.binance_downloader import BinanceDownloader; d = BinanceDownloader(); print('Symbols:', d.get_symbols()[:3])"

    echo "Testing WebSocket module..."
    python3 -c "from data.websocket_stream import BinanceWebSocketStream; print('WebSocket module OK')"

    echo -e "${GREEN}[OK] All tests passed${NC}"
}

# Function: Show help
show_help() {
    echo "Quant Research Lab - Quick Start Script"
    echo ""
    echo "Usage: ./start.sh [command]"
    echo ""
    echo "Commands:"
    echo "  check       Check system dependencies"
    echo "  install     Install Python dependencies"
    echo "  setup-db    Setup MySQL database and tables"
    echo "  download    Download historical data"
    echo "  features    Build features from data"
    echo "  backtest    Run backtest"
    echo "  train-ml    Train ML model"
    echo "  live        Start live trading (paper mode)"
    echo "  setup       Run full setup (check, install, setup-db)"
    echo "  test        Run basic tests"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./start.sh setup          # First time setup"
    echo "  ./start.sh download       # Download data"
    echo "  ./start.sh backtest       # Run backtest"
    echo "  ./start.sh live           # Start paper trading"
}

# Main
case "${1:-help}" in
    check)
        check_dependencies
        ;;
    install)
        install
        ;;
    setup-db)
        setup_db
        ;;
    download)
        download_data
        ;;
    features)
        build_features
        ;;
    backtest)
        backtest
        ;;
    train-ml)
        train_ml
        ;;
    live)
        live_trading
        ;;
    setup)
        setup_all
        ;;
    test)
        run_tests
        ;;
    help|*)
        show_help
        ;;
esac
