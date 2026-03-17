#!/bin/bash
# Quick Deploy Script for Quant Research Lab
# Run this script for first-time deployment

set -e

echo "=========================================="
echo "Quant Research Lab - Quick Deploy"
echo "=========================================="

PROJECT_DIR="/home/bot2/claude/quant_research_lab"
cd "$PROJECT_DIR"

# Step 1: Create .env from example if not exists
if [ ! -f .env ]; then
    echo "[1/6] Creating .env file..."
    cp .env.example .env
    echo "Please edit .env file with your API keys!"
    echo "Required: BINANCE_API_KEY, BINANCE_API_SECRET"
else
    echo "[1/6] .env file already exists, skipping..."
fi

# Step 2: Install Python dependencies
echo "[2/6] Installing Python dependencies..."
pip3 install -r requirements.txt --quiet

# Step 3: Create logs directory
echo "[3/6] Creating directories..."
mkdir -p logs data models

# Step 4: Check MySQL
echo "[4/6] Checking MySQL..."
if systemctl is-active --quiet mysql; then
    echo "MySQL is running"
else
    echo "MySQL is not running, attempting to start..."
    sudo systemctl start mysql 2>/dev/null || echo "Please start MySQL manually"
fi

# Step 5: Initialize database
echo "[5/6] Initializing database..."
mysql -u bot2 -pbot2_123456 -e "CREATE DATABASE IF NOT EXISTS quant_research;" 2>/dev/null || true

# Step 6: Create database tables
echo "[6/6] Creating database tables..."
python3 -c "
from data.mysql_storage import MySQLStorage
try:
    storage = MySQLStorage(host='localhost', user='bot2', password='bot2_123456', database='quant_research')
    storage.connect()
    storage.create_tables()
    storage.disconnect()
    print('Database tables created successfully!')
except Exception as e:
    print(f'Database setup skipped: {e}')
" 2>/dev/null || echo "Database setup skipped (MySQL may not be configured)"

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys:"
echo "   nano $PROJECT_DIR/.env"
echo ""
echo "2. Test the system:"
echo "   ./start.sh test"
echo ""
echo "3. Download historical data:"
echo "   ./start.sh download"
echo ""
echo "4. Run backtest:"
echo "   ./start.sh backtest"
echo ""
echo "5. Start paper trading:"
echo "   ./start.sh live"
echo ""
echo "For systemd service (auto-start on boot):"
echo "   sudo cp scripts/quant-lab.service /etc/systemd/system/"
echo "   sudo systemctl enable quant-lab"
echo "   sudo systemctl start quant-lab"
echo ""
echo "For Docker deployment:"
echo "   docker-compose up -d"
echo ""
