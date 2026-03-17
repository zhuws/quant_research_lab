"""
Metrics Collector for Quant Research Lab.

Provides real-time metrics collection and aggregation:
    - Multi-type metric support
    - Time-series storage
    - Statistical aggregation
    - Export capabilities

Metric Types:
    - Counter: Monotonically increasing values
    - Gauge: Point-in-time values
    - Histogram: Distribution of values
    - Summary: Statistical summary

Example usage:
    ```python
    from monitoring.metrics_collector import MetricsCollector, MetricType

    collector = MetricsCollector()

    # Record metrics
    collector.gauge('equity', 100000)
    collector.counter('trades_executed', 1)
    collector.histogram('order_latency_ms', 45)

    # Get aggregated metrics
    summary = collector.get_metric_summary('equity')
    ```
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading
import time
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger


class MetricType(Enum):
    """Type of metric."""
    COUNTER = 'counter'
    GAUGE = 'gauge'
    HISTOGRAM = 'histogram'
    SUMMARY = 'summary'


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class MetricSummary:
    """Statistical summary of a metric."""
    name: str
    count: int = 0
    sum_value: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    last_value: float = 0.0
    last_timestamp: Optional[datetime] = None

    # Percentiles
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'count': self.count,
            'sum': self.sum_value,
            'min': self.min_value,
            'max': self.max_value,
            'mean': self.mean,
            'std': self.std,
            'last_value': self.last_value,
            'last_timestamp': self.last_timestamp.isoformat() if self.last_timestamp else None,
            'p50': self.p50,
            'p90': self.p90,
            'p95': self.p95,
            'p99': self.p99
        }


class MetricsCollector:
    """
    Real-time Metrics Collection and Aggregation.

    Features:
        - Multiple metric types
        - Time-series storage
        - Statistical analysis
        - Thread-safe operations
        - Export capabilities

    Attributes:
        max_points: Maximum points per metric
        default_window: Default aggregation window
    """

    def __init__(
        self,
        max_points: int = 10000,
        default_window: timedelta = timedelta(hours=24)
    ):
        """
        Initialize Metrics Collector.

        Args:
            max_points: Maximum points per metric
            default_window: Default aggregation window
        """
        self.max_points = max_points
        self.default_window = default_window
        self.logger = get_logger('metrics_collector')

        # Storage
        self._metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)

        # Labels
        self._metric_labels: Dict[str, Dict[str, str]] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Callbacks
        self._callbacks: List[Callable[[MetricPoint], None]] = []

    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> float:
        """
        Increment a counter metric.

        Args:
            name: Metric name
            value: Increment value (default 1)
            labels: Optional labels
            timestamp: Optional timestamp

        Returns:
            New counter value
        """
        with self._lock:
            if name not in self._counters:
                self._counters[name] = 0.0

            self._counters[name] += value

            point = MetricPoint(
                name=name,
                value=self._counters[name],
                timestamp=timestamp or datetime.utcnow(),
                labels=labels or {},
                metric_type=MetricType.COUNTER
            )

            self._record_point(name, point)
            self._notify_callbacks(point)

            return self._counters[name]

    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Set a gauge metric.

        Args:
            name: Metric name
            value: Gauge value
            labels: Optional labels
            timestamp: Optional timestamp
        """
        with self._lock:
            self._gauges[name] = value

            if labels:
                self._metric_labels[name] = labels

            point = MetricPoint(
                name=name,
                value=value,
                timestamp=timestamp or datetime.utcnow(),
                labels=labels or {},
                metric_type=MetricType.GAUGE
            )

            self._record_point(name, point)
            self._notify_callbacks(point)

    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a histogram value.

        Args:
            name: Metric name
            value: Value to record
            labels: Optional labels
            timestamp: Optional timestamp
        """
        with self._lock:
            self._histograms[name].append(value)

            # Keep limited history
            if len(self._histograms[name]) > self.max_points:
                self._histograms[name] = self._histograms[name][-self.max_points:]

            point = MetricPoint(
                name=name,
                value=value,
                timestamp=timestamp or datetime.utcnow(),
                labels=labels or {},
                metric_type=MetricType.HISTOGRAM
            )

            self._record_point(name, point)
            self._notify_callbacks(point)

    def timing(
        self,
        name: str,
        value_ms: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a timing metric (convenience for histogram).

        Args:
            name: Metric name
            value_ms: Value in milliseconds
            labels: Optional labels
        """
        self.histogram(f"{name}_ms", value_ms, labels)

    def get_counter(self, name: str) -> float:
        """Get current counter value."""
        return self._counters.get(name, 0.0)

    def get_gauge(self, name: str) -> float:
        """Get current gauge value."""
        return self._gauges.get(name, 0.0)

    def get_metric_summary(
        self,
        name: str,
        window: Optional[timedelta] = None
    ) -> MetricSummary:
        """
        Get statistical summary of a metric.

        Args:
            name: Metric name
            window: Time window (default: 24h)

        Returns:
            MetricSummary with statistics
        """
        window = window or self.default_window
        cutoff = datetime.utcnow() - window

        with self._lock:
            points = [p for p in self._metrics.get(name, []) if p.timestamp >= cutoff]

        if not points:
            return MetricSummary(name=name)

        values = [p.value for p in points]

        return MetricSummary(
            name=name,
            count=len(values),
            sum_value=sum(values),
            min_value=min(values),
            max_value=max(values),
            mean=np.mean(values),
            std=np.std(values) if len(values) > 1 else 0,
            last_value=values[-1],
            last_timestamp=points[-1].timestamp,
            p50=float(np.percentile(values, 50)) if values else 0,
            p90=float(np.percentile(values, 90)) if values else 0,
            p95=float(np.percentile(values, 95)) if values else 0,
            p99=float(np.percentile(values, 99)) if values else 0
        )

    def get_all_summaries(
        self,
        window: Optional[timedelta] = None
    ) -> Dict[str, MetricSummary]:
        """
        Get summaries for all metrics.

        Args:
            window: Time window

        Returns:
            Dictionary of metric summaries
        """
        with self._lock:
            names = list(self._metrics.keys())

        return {name: self.get_metric_summary(name, window) for name in names}

    def get_metric_series(
        self,
        name: str,
        window: Optional[timedelta] = None
    ) -> pd.DataFrame:
        """
        Get time series for a metric.

        Args:
            name: Metric name
            window: Time window

        Returns:
            DataFrame with timestamp and value columns
        """
        window = window or self.default_window
        cutoff = datetime.utcnow() - window

        with self._lock:
            points = [p for p in self._metrics.get(name, []) if p.timestamp >= cutoff]

        if not points:
            return pd.DataFrame(columns=['timestamp', 'value'])

        df = pd.DataFrame([
            {'timestamp': p.timestamp, 'value': p.value}
            for p in points
        ])
        df.set_index('timestamp', inplace=True)

        return df

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted string
        """
        lines = []

        with self._lock:
            # Export counters
            for name, value in self._counters.items():
                labels = self._format_labels(self._metric_labels.get(name, {}))
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{labels} {value}")

            # Export gauges
            for name, value in self._gauges.items():
                if name in self._counters:
                    continue  # Already exported as counter
                labels = self._format_labels(self._metric_labels.get(name, {}))
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{labels} {value}")

            # Export histograms
            for name, values in self._histograms.items():
                if not values:
                    continue
                summary = self.get_metric_summary(name)

                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {summary.count}")
                lines.append(f"{name}_sum {summary.sum_value}")
                lines.append(f"{name}_min {summary.min_value}")
                lines.append(f"{name}_max {summary.max_value}")
                lines.append(f"{name}_p50 {summary.p50}")
                lines.append(f"{name}_p90 {summary.p90}")
                lines.append(f"{name}_p95 {summary.p95}")
                lines.append(f"{name}_p99 {summary.p99}")

        return '\n'.join(lines)

    def export_json(self) -> str:
        """
        Export metrics as JSON.

        Returns:
            JSON string
        """
        summaries = self.get_all_summaries()
        return json.dumps(
            {name: summary.to_dict() for name, summary in summaries.items()},
            indent=2
        )

    def clear_metric(self, name: str) -> None:
        """Clear a specific metric."""
        with self._lock:
            self._metrics.pop(name, None)
            self._counters.pop(name, None)
            self._gauges.pop(name, None)
            self._histograms.pop(name, None)
            self._metric_labels.pop(name, None)

    def clear_all(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._metric_labels.clear()

    def on_metric(self, callback: Callable[[MetricPoint], None]) -> None:
        """Register callback for new metrics."""
        self._callbacks.append(callback)

    def _record_point(self, name: str, point: MetricPoint) -> None:
        """Record a metric point."""
        self._metrics[name].append(point)

        # Trim to max points
        if len(self._metrics[name]) > self.max_points:
            self._metrics[name] = self._metrics[name][-self.max_points:]

    def _notify_callbacks(self, point: MetricPoint) -> None:
        """Notify registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(point)
            except Exception as e:
                self.logger.error(f"Metric callback error: {e}")

    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus."""
        if not labels:
            return ''
        label_str = ','.join(f'{k}="{v}"' for k, v in labels.items())
        return f'{{{label_str}}}'


class Timer:
    """
    Context manager for timing operations.

    Example:
        ```python
        collector = MetricsCollector()
        with Timer('operation', collector):
            # Do something
            pass
        ```
    """

    def __init__(
        self,
        name: str,
        collector: MetricsCollector,
        labels: Optional[Dict[str, str]] = None
    ):
        self.name = name
        self.collector = collector
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        elapsed_ms = (time.time() - self.start_time) * 1000
        self.collector.timing(self.name, elapsed_ms, self.labels)


__all__ = [
    'MetricType',
    'MetricPoint',
    'MetricSummary',
    'MetricsCollector',
    'Timer'
]
