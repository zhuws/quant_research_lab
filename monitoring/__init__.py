"""
Monitoring Module for Quant Research Lab.

Provides comprehensive monitoring and dashboard capabilities:
    - Real-time metrics collection
    - Performance monitoring
    - Alert management
    - Dashboard visualization
    - System health tracking

Components:
    - MetricsCollector: Collect and aggregate metrics
    - PerformanceMonitor: Track strategy and system performance
    - AlertManager: Alert generation and notification
    - DashboardServer: Web-based dashboard
"""

from monitoring.metrics_collector import MetricsCollector, MetricType
from monitoring.performance_monitor import PerformanceMonitor, PerformanceSnapshot
from monitoring.alert_manager import AlertManager, Alert, AlertLevel
from monitoring.dashboard_server import DashboardServer, DashboardConfig

__all__ = [
    'MetricsCollector',
    'MetricType',
    'PerformanceMonitor',
    'PerformanceSnapshot',
    'AlertManager',
    'Alert',
    'AlertLevel',
    'DashboardServer',
    'DashboardConfig'
]
