"""
Dashboard Server for Quant Research Lab.

Provides web-based monitoring dashboard:
    - Real-time metrics display
    - Performance charts
    - Alert management
    - System health status
    - API endpoints

Features:
    - WebSocket for real-time updates
    - REST API for queries
    - Static dashboard UI
    - Prometheus metrics endpoint

Example usage:
    ```python
    from monitoring.dashboard_server import DashboardServer, DashboardConfig
    from monitoring.metrics_collector import MetricsCollector
    from monitoring.performance_monitor import PerformanceMonitor

    config = DashboardConfig(port=8080)
    server = DashboardServer(config)

    server.set_metrics_collector(collector)
    server.set_performance_monitor(monitor)

    await server.start()
    ```
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import threading
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from monitoring.metrics_collector import MetricsCollector
from monitoring.performance_monitor import PerformanceMonitor
from monitoring.alert_manager import AlertManager

# Try to import optional dependencies
try:
    from aiohttp import web
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


@dataclass
class DashboardConfig:
    """
    Dashboard server configuration.

    Attributes:
        host: Server host
        port: Server port
        update_interval: Update interval in seconds
        enable_websocket: Enable WebSocket updates
        enable_prometheus: Enable Prometheus endpoint
        auth_enabled: Enable authentication
        auth_token: Authentication token
    """
    host: str = '0.0.0.0'
    port: int = 8080
    update_interval: float = 1.0
    enable_websocket: bool = True
    enable_prometheus: bool = True
    auth_enabled: bool = False
    auth_token: str = ''


# HTML template for the dashboard
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quant Research Lab - Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; padding: 20px 0; border-bottom: 1px solid #333; margin-bottom: 20px; }
        .header h1 { color: #00d4ff; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: #16213e; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .card h2 { color: #00d4ff; margin-bottom: 15px; font-size: 1.1em; }
        .metric { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #2a2a4a; }
        .metric:last-child { border-bottom: none; }
        .metric-label { color: #888; }
        .metric-value { font-weight: bold; font-size: 1.1em; }
        .positive { color: #00ff88; }
        .negative { color: #ff4757; }
        .warning { color: #ffa502; }
        .status-indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
        .status-good { background: #00ff88; }
        .status-warning { background: #ffa502; }
        .status-error { background: #ff4757; }
        .chart-container { height: 200px; background: #0f0f23; border-radius: 5px; margin-top: 10px; }
        .alerts { max-height: 300px; overflow-y: auto; }
        .alert-item { padding: 10px; margin: 5px 0; border-radius: 5px; }
        .alert-info { background: #1e3a5f; }
        .alert-warning { background: #3d3d00; }
        .alert-error { background: #3d0000; }
        .alert-critical { background: #5c0000; }
        .refresh-info { text-align: center; color: #666; margin-top: 20px; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Quant Research Lab Dashboard</h1>
            <p>Real-time monitoring and performance tracking</p>
        </div>

        <div class="grid">
            <div class="card">
                <h2><span class="status-indicator status-good" id="status-indicator"></span>System Status</h2>
                <div class="metric"><span class="metric-label">Status</span><span class="metric-value" id="system-status">Running</span></div>
                <div class="metric"><span class="metric-label">Uptime</span><span class="metric-value" id="system-uptime">0:00:00</span></div>
                <div class="metric"><span class="metric-label">Last Update</span><span class="metric-value" id="last-update">-</span></div>
            </div>

            <div class="card">
                <h2>Portfolio</h2>
                <div class="metric"><span class="metric-label">Equity</span><span class="metric-value" id="equity">$0</span></div>
                <div class="metric"><span class="metric-label">Total P&L</span><span class="metric-value" id="total-pnl">$0</span></div>
                <div class="metric"><span class="metric-label">Unrealized P&L</span><span class="metric-value" id="unrealized-pnl">$0</span></div>
                <div class="metric"><span class="metric-label">Max Drawdown</span><span class="metric-value" id="max-drawdown">0%</span></div>
            </div>

            <div class="card">
                <h2>Trading</h2>
                <div class="metric"><span class="metric-label">Total Trades</span><span class="metric-value" id="total-trades">0</span></div>
                <div class="metric"><span class="metric-label">Win Rate</span><span class="metric-value" id="win-rate">0%</span></div>
                <div class="metric"><span class="metric-label">Avg Trade P&L</span><span class="metric-value" id="avg-pnl">$0</span></div>
                <div class="metric"><span class="metric-label">Throughput</span><span class="metric-value" id="throughput">0/min</span></div>
            </div>

            <div class="card">
                <h2>Performance</h2>
                <div class="metric"><span class="metric-label">Sharpe Ratio</span><span class="metric-value" id="sharpe">0</span></div>
                <div class="metric"><span class="metric-label">Sortino Ratio</span><span class="metric-value" id="sortino">0</span></div>
                <div class="metric"><span class="metric-label">Daily Return</span><span class="metric-value" id="daily-return">0%</span></div>
                <div class="metric"><span class="metric-label">Monthly Return</span><span class="metric-value" id="monthly-return">0%</span></div>
            </div>

            <div class="card">
                <h2>Execution</h2>
                <div class="metric"><span class="metric-label">Avg Latency</span><span class="metric-value" id="avg-latency">0ms</span></div>
                <div class="metric"><span class="metric-label">P99 Latency</span><span class="metric-value" id="p99-latency">0ms</span></div>
                <div class="metric"><span class="metric-label">Open Positions</span><span class="metric-value" id="open-positions">0</span></div>
            </div>

            <div class="card">
                <h2>Alerts</h2>
                <div class="alerts" id="alerts-list">
                    <p style="color: #666;">No active alerts</p>
                </div>
            </div>
        </div>

        <div class="refresh-info">
            Auto-refreshing every <span id="refresh-interval">1</span>s | WebSocket: <span id="ws-status">Disconnected</span>
        </div>
    </div>

    <script>
        let ws = null;
        let startTime = Date.now();

        function formatCurrency(value) {
            return '$' + value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
        }

        function formatPercent(value) {
            return (value * 100).toFixed(2) + '%';
        }

        function formatDuration(ms) {
            const s = Math.floor(ms / 1000) % 60;
            const m = Math.floor(ms / 60000) % 60;
            const h = Math.floor(ms / 3600000);
            return h + ':' + m.toString().padStart(2, '0') + ':' + s.toString().padStart(2, '0');
        }

        function updateDashboard(data) {
            if (data.snapshot) {
                const s = data.snapshot;
                document.getElementById('equity').textContent = formatCurrency(s.equity);
                document.getElementById('total-pnl').textContent = formatCurrency(s.total_pnl);
                document.getElementById('total-pnl').className = 'metric-value ' + (s.total_pnl >= 0 ? 'positive' : 'negative');
                document.getElementById('unrealized-pnl').textContent = formatCurrency(s.unrealized_pnl);
                document.getElementById('unrealized-pnl').className = 'metric-value ' + (s.unrealized_pnl >= 0 ? 'positive' : 'negative');
                document.getElementById('max-drawdown').textContent = formatPercent(s.max_drawdown);
                document.getElementById('max-drawdown').className = 'metric-value ' + (s.max_drawdown > 0.1 ? 'negative' : '');
                document.getElementById('total-trades').textContent = s.total_trades;
                document.getElementById('win-rate').textContent = s.win_rate.toFixed(1) + '%';
                document.getElementById('win-rate').className = 'metric-value ' + (s.win_rate >= 50 ? 'positive' : 'negative');
                document.getElementById('avg-pnl').textContent = formatCurrency(s.avg_trade_pnl);
                document.getElementById('throughput').textContent = s.throughput.toFixed(1) + '/min';
                document.getElementById('sharpe').textContent = s.sharpe_ratio.toFixed(2);
                document.getElementById('sharpe').className = 'metric-value ' + (s.sharpe_ratio >= 1 ? 'positive' : s.sharpe_ratio < 0 ? 'negative' : '');
                document.getElementById('sortino').textContent = s.sortino_ratio.toFixed(2);
                document.getElementById('daily-return').textContent = formatPercent(s.daily_return);
                document.getElementById('daily-return').className = 'metric-value ' + (s.daily_return >= 0 ? 'positive' : 'negative');
                document.getElementById('monthly-return').textContent = formatPercent(s.monthly_return);
                document.getElementById('monthly-return').className = 'metric-value ' + (s.monthly_return >= 0 ? 'positive' : 'negative');
                document.getElementById('avg-latency').textContent = s.avg_latency_ms.toFixed(1) + 'ms';
                document.getElementById('open-positions').textContent = s.positions;
            }

            if (data.latency_stats) {
                document.getElementById('p99-latency').textContent = data.latency_stats.p99.toFixed(1) + 'ms';
            }

            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
            document.getElementById('system-uptime').textContent = formatDuration(Date.now() - startTime);
        }

        function updateAlerts(alerts) {
            const list = document.getElementById('alerts-list');
            if (!alerts || alerts.length === 0) {
                list.innerHTML = '<p style="color: #666;">No active alerts</p>';
                return;
            }

            list.innerHTML = alerts.map(a =>
                '<div class="alert-item alert-' + a.level + '">' +
                '<strong>' + a.name + '</strong>: ' + a.message +
                '</div>'
            ).join('');
        }

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(protocol + '//' + window.location.host + '/ws');

            ws.onopen = function() {
                document.getElementById('ws-status').textContent = 'Connected';
                document.getElementById('ws-status').style.color = '#00ff88';
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
                if (data.alerts) {
                    updateAlerts(data.alerts);
                }
            };

            ws.onclose = function() {
                document.getElementById('ws-status').textContent = 'Disconnected';
                document.getElementById('ws-status').style.color = '#ff4757';
                setTimeout(connectWebSocket, 5000);
            };
        }

        async function fetchData() {
            try {
                const response = await fetch('/api/dashboard');
                const data = await response.json();
                updateDashboard(data);
                if (data.alerts) {
                    updateAlerts(data.alerts);
                }
            } catch (error) {
                console.error('Fetch error:', error);
            }
        }

        // Initialize
        if (typeof WebSocket !== 'undefined') {
            connectWebSocket();
        } else {
            setInterval(fetchData, 1000);
        }
        fetchData();
        setInterval(() => {
            document.getElementById('system-uptime').textContent = formatDuration(Date.now() - startTime);
        }, 1000);
    </script>
</body>
</html>
'''


class DashboardServer:
    """
    Web-based Monitoring Dashboard.

    Features:
        - Real-time updates via WebSocket
        - REST API for metrics
        - Prometheus endpoint
        - Static HTML dashboard

    Attributes:
        config: Dashboard configuration
    """

    def __init__(self, config: Optional[DashboardConfig] = None):
        """
        Initialize Dashboard Server.

        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        self.logger = get_logger('dashboard_server')

        # Components
        self._metrics_collector: Optional[MetricsCollector] = None
        self._performance_monitor: Optional[PerformanceMonitor] = None
        self._alert_manager: Optional[AlertManager] = None

        # Server
        self._app = None
        self._runner = None
        self._site = None
        self._ws_clients: List = []
        self._update_task = None
        self._start_time: Optional[datetime] = None

    def set_metrics_collector(self, collector: MetricsCollector) -> None:
        """Set metrics collector."""
        self._metrics_collector = collector

    def set_performance_monitor(self, monitor: PerformanceMonitor) -> None:
        """Set performance monitor."""
        self._performance_monitor = monitor

    def set_alert_manager(self, manager: AlertManager) -> None:
        """Set alert manager."""
        self._alert_manager = manager

    async def start(self) -> bool:
        """
        Start the dashboard server.

        Returns:
            True if started successfully
        """
        if not HAS_AIOHTTP:
            self.logger.error("aiohttp not installed. Run: pip install aiohttp")
            return False

        self._app = web.Application()

        # Routes
        self._app.router.add_get('/', self._handle_index)
        self._app.router.add_get('/api/dashboard', self._handle_dashboard)
        self._app.router.add_get('/api/metrics', self._handle_metrics)
        self._app.router.add_get('/api/performance', self._handle_performance)
        self._app.router.add_get('/api/alerts', self._handle_alerts)

        if self.config.enable_prometheus and self._metrics_collector:
            self._app.router.add_get('/metrics', self._handle_prometheus)

        if self.config.enable_websocket:
            self._app.router.add_get('/ws', self._handle_websocket)

        # Start server
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, self.config.host, self.config.port)
        await self._site.start()

        self._start_time = datetime.utcnow()

        # Start update task
        if self.config.enable_websocket:
            self._update_task = asyncio.create_task(self._broadcast_updates())

        self.logger.info(f"Dashboard server started at http://{self.config.host}:{self.config.port}")

        return True

    async def stop(self) -> None:
        """Stop the dashboard server."""
        if self._update_task:
            self._update_task.cancel()

        if self._runner:
            await self._runner.cleanup()

        self.logger.info("Dashboard server stopped")

    async def _handle_index(self, request) -> web.Response:
        """Serve dashboard HTML."""
        return web.Response(text=DASHBOARD_HTML, content_type='text/html')

    async def _handle_dashboard(self, request) -> web.Response:
        """Get dashboard data."""
        data = await self._get_dashboard_data()
        return web.json_response(data)

    async def _handle_metrics(self, request) -> web.Response:
        """Get all metrics."""
        if self._metrics_collector:
            summaries = self._metrics_collector.get_all_summaries()
            data = {name: summary.to_dict() for name, summary in summaries.items()}
        else:
            data = {}
        return web.json_response(data)

    async def _handle_performance(self, request) -> web.Response:
        """Get performance data."""
        if self._performance_monitor:
            snapshot = self._performance_monitor.get_snapshot()
            data = snapshot.to_dict()
        else:
            data = {}
        return web.json_response(data)

    async def _handle_alerts(self, request) -> web.Response:
        """Get active alerts."""
        if self._alert_manager:
            alerts = self._alert_manager.get_active_alerts()
            data = [a.to_dict() for a in alerts]
        else:
            data = []
        return web.json_response(data)

    async def _handle_prometheus(self, request) -> web.Response:
        """Serve Prometheus metrics."""
        if self._metrics_collector:
            text = self._metrics_collector.export_prometheus()
        else:
            text = ''
        return web.Response(text=text, content_type='text/plain')

    async def _handle_websocket(self, request) -> web.WebSocketResponse:
        """Handle WebSocket connection."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self._ws_clients.append(ws)
        self.logger.info(f"WebSocket client connected. Total: {len(self._ws_clients)}")

        # Send initial data
        data = await self._get_dashboard_data()
        await ws.send_json(data)

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    # Handle client messages if needed
                    pass
                elif msg.type == web.WSMsgType.ERROR:
                    self.logger.error(f"WebSocket error: {ws.exception()}")
        finally:
            self._ws_clients.remove(ws)
            self.logger.info(f"WebSocket client disconnected. Total: {len(self._ws_clients)}")

        return ws

    async def _get_dashboard_data(self) -> Dict[str, Any]:
        """Compile dashboard data."""
        data = {
            'timestamp': datetime.utcnow().isoformat(),
            'uptime': str(datetime.utcnow() - self._start_time) if self._start_time else '0:00:00'
        }

        if self._performance_monitor:
            data['snapshot'] = self._performance_monitor.get_snapshot().to_dict()
            data['latency_stats'] = self._performance_monitor.get_latency_stats()

        if self._alert_manager:
            alerts = self._alert_manager.get_active_alerts(limit=10)
            data['alerts'] = [a.to_dict() for a in alerts]

        return data

    async def _broadcast_updates(self) -> None:
        """Broadcast updates to WebSocket clients."""
        while True:
            try:
                await asyncio.sleep(self.config.update_interval)

                if self._ws_clients:
                    data = await self._get_dashboard_data()
                    message = json.dumps(data)

                    for ws in self._ws_clients[:]:
                        try:
                            await ws.send_str(message)
                        except Exception:
                            pass  # Client will be removed in handler

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Broadcast error: {e}")


__all__ = [
    'DashboardConfig',
    'DashboardServer'
]
