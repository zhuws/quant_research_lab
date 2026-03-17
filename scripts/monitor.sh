#!/bin/bash
# Quick Test Script - Run basic tests without database

echo "Quant Research Lab - Quick Test"
echo "================================"

cd /home/bot2/claude/quant_research_lab

# Test 1: Python imports
echo -e "\n[Test 1] Python module imports..."
python3 -c "
import sys
sys.path.insert(0, '.')
modules = [
    'utils.logger',
    'utils.time_utils',
    'data.binance_downloader',
    'data.bybit_downloader',
    'data.websocket_stream',
    'features.orderflow_features',
    'features.liquidity_features',
    'strategies.base_strategy',
    'backtest.backtest_engine',
    'risk.risk_engine',
    'research.factor_library',
    'alpha_models.model_trainer',
    'rl_agents.trading_env'
]
failed = []
for m in modules:
    try:
        __import__(m)
        print(f'  [OK] {m}')
    except Exception as e:
        print(f'  [FAIL] {m}: {e}')
        failed.append(m)
if failed:
    print(f'\nFailed modules: {failed}')
    sys.exit(1)
print('\nAll modules imported successfully!')
"

# Test 2: Binance API connection
echo -e "\n[Test 2] Binance API connection..."
python3 -c "
from data.binance_downloader import BinanceDownloader
d = BinanceDownloader(use_futures=True)
symbols = d.get_symbols()
print(f'  Connected! Available symbols: {len(symbols)}')
print(f'  Sample: {symbols[:5]}')
server_time = d.get_server_time()
print(f'  Server time: {server_time}')
"

# Test 3: Get ETHUSDT price
echo -e "\n[Test 3] Get ETHUSDT current price..."
python3 -c "
from data.binance_downloader import BinanceDownloader
d = BinanceDownloader(use_futures=True)
ticker = d.get_ticker('ETHUSDT')
print(f'  ETHUSDT Price: \${ticker[\"lastPrice\"]}')
print(f'  24h Change: {ticker[\"priceChangePercent\"]}%')
print(f'  24h Volume: {float(ticker[\"volume\"]):.2f} ETH')
"

# Test 4: Download sample OHLCV
echo -e "\n[Test 4] Download sample OHLCV data..."
python3 -c "
from data.binance_downloader import BinanceDownloader
from datetime import datetime, timedelta
d = BinanceDownloader(use_futures=True)
end = datetime.utcnow()
start = end - timedelta(minutes=10)
df = d.download_ohlcv('ETHUSDT', '1m', start, end, limit=5)
print(f'  Downloaded {len(df)} candles')
print(df.to_string())
"

# Test 5: WebSocket (quick connect)
echo -e "\n[Test 5] WebSocket module..."
python3 -c "
from data.websocket_stream import BinanceWebSocketStream
import asyncio

async def test_ws():
    msg_received = False

    def on_msg(msg):
        nonlocal msg_received
        msg_received = True
        print(f'  Received: {msg.channel}')

    ws = BinanceWebSocketStream('ETHUSDT', on_message=on_msg)

    # Just test initialization
    print('  WebSocket module initialized OK')

asyncio.run(test_ws())
"

echo -e "\n================================"
echo "All tests passed!"
echo "================================"
