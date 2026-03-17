"""
Alert Manager for Quant Research Lab.

Provides comprehensive alert management:
    - Multi-level alerts (info, warning, error, critical)
    - Alert rules and conditions
    - Notification channels
    - Alert history and tracking
    - Cooldown and suppression

Alert types:
    - Risk alerts (drawdown, exposure)
    - Performance alerts (latency, error rate)
    - System alerts (connection, resource)
    - Trading alerts (position, fill)

Example usage:
    ```python
    from monitoring.alert_manager import AlertManager, AlertLevel

    manager = AlertManager()

    # Add rule
    manager.add_rule('drawdown_alert', lambda: drawdown > 0.2, AlertLevel.CRITICAL)

    # Check alerts
    alerts = manager.check_all()

    # Or send manual alert
    manager.send_alert('Order rejected', AlertLevel.WARNING, {'symbol': 'BTCUSDT'})
    ```
"""

import asyncio
from typing import Dict, List, Optional, Callable, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger


class AlertLevel(Enum):
    """Alert severity level."""
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'

    def __lt__(self, other):
        levels = [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]
        return levels.index(self) < levels.index(other)

    def __le__(self, other):
        return self == other or self < other


class AlertCategory(Enum):
    """Alert category."""
    RISK = 'risk'
    PERFORMANCE = 'performance'
    SYSTEM = 'system'
    TRADING = 'trading'
    DATA = 'data'


@dataclass
class Alert:
    """
    Alert data structure.

    Attributes:
        id: Unique alert ID
        name: Alert name
        level: Severity level
        category: Alert category
        message: Alert message
        details: Additional details
        timestamp: When alert was generated
        acknowledged: Whether alert was acknowledged
        acknowledged_by: Who acknowledged
        acknowledged_at: When acknowledged
        suppressed: Whether suppressed
        suppress_until: Suppression end time
    """
    id: str
    name: str
    level: AlertLevel
    category: AlertCategory
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    suppressed: bool = False
    suppress_until: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'level': self.level.value,
            'category': self.category.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'suppressed': self.suppressed
        }


@dataclass
class AlertRule:
    """
    Alert rule definition.

    Attributes:
        name: Rule name
        condition: Condition function (returns True to trigger)
        level: Alert level when triggered
        category: Alert category
        message: Alert message template
        cooldown: Minimum time between alerts
        details_func: Function to generate details
        enabled: Whether rule is active
    """
    name: str
    condition: Callable[[], bool]
    level: AlertLevel = AlertLevel.WARNING
    category: AlertCategory = AlertCategory.SYSTEM
    message: str = "Alert condition triggered"
    cooldown: timedelta = timedelta(minutes=5)
    details_func: Optional[Callable[[], Dict[str, Any]]] = None
    enabled: bool = True


class NotificationChannel:
    """Base class for notification channels."""

    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f'notification_{name}')

    async def send(self, alert: Alert) -> bool:
        """Send notification for alert."""
        raise NotImplementedError


class LogChannel(NotificationChannel):
    """Log-based notification channel."""

    def __init__(self):
        super().__init__('log')
        self.logger = get_logger('alerts')

    async def send(self, alert: Alert) -> bool:
        """Log the alert."""
        level_map = {
            AlertLevel.INFO: self.logger.info,
            AlertLevel.WARNING: self.logger.warning,
            AlertLevel.ERROR: self.logger.error,
            AlertLevel.CRITICAL: self.logger.critical
        }
        log_func = level_map.get(alert.level, self.logger.info)
        log_func(f"[{alert.category.value.upper()}] {alert.name}: {alert.message}")
        return True


class WebhookChannel(NotificationChannel):
    """Webhook notification channel."""

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        super().__init__('webhook')
        self.url = url
        self.headers = headers or {}

    async def send(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.url,
                    json=alert.to_dict(),
                    headers=self.headers
                ) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"Webhook failed: {e}")
            return False


class AlertManager:
    """
    Alert Management System.

    Features:
        - Rule-based alerting
        - Multiple notification channels
        - Alert history
        - Cooldown management
        - Acknowledgment tracking

    Attributes:
        default_cooldown: Default cooldown between alerts
        max_history: Maximum alerts to keep in history
    """

    def __init__(
        self,
        default_cooldown: timedelta = timedelta(minutes=5),
        max_history: int = 1000
    ):
        """
        Initialize Alert Manager.

        Args:
            default_cooldown: Default alert cooldown
            max_history: Maximum history size
        """
        self.default_cooldown = default_cooldown
        self.max_history = max_history
        self.logger = get_logger('alert_manager')

        # Rules
        self._rules: Dict[str, AlertRule] = {}
        self._last_triggered: Dict[str, datetime] = {}

        # Alerts
        self._active_alerts: Dict[str, Alert] = {}
        self._history: List[Alert] = []

        # Channels
        self._channels: List[NotificationChannel] = [LogChannel()]

        # Counters
        self._alert_counter = 0

        # Thread safety
        self._lock = threading.RLock()

    def add_rule(
        self,
        name: str,
        condition: Callable[[], bool],
        level: AlertLevel = AlertLevel.WARNING,
        category: AlertCategory = AlertCategory.SYSTEM,
        message: Optional[str] = None,
        cooldown: Optional[timedelta] = None,
        details_func: Optional[Callable[[], Dict[str, Any]]] = None
    ) -> None:
        """
        Add an alert rule.

        Args:
            name: Rule name
            condition: Condition function
            level: Alert level
            category: Alert category
            message: Alert message
            cooldown: Cooldown period
            details_func: Details function
        """
        rule = AlertRule(
            name=name,
            condition=condition,
            level=level,
            category=category,
            message=message or f"{name} condition triggered",
            cooldown=cooldown or self.default_cooldown,
            details_func=details_func
        )

        with self._lock:
            self._rules[name] = rule

        self.logger.info(f"Added alert rule: {name}")

    def remove_rule(self, name: str) -> bool:
        """Remove an alert rule."""
        with self._lock:
            if name in self._rules:
                del self._rules[name]
                return True
            return False

    def enable_rule(self, name: str, enabled: bool = True) -> None:
        """Enable or disable a rule."""
        with self._lock:
            if name in self._rules:
                self._rules[name].enabled = enabled

    def add_channel(self, channel: NotificationChannel) -> None:
        """Add notification channel."""
        self._channels.append(channel)

    def check_rule(self, name: str) -> Optional[Alert]:
        """
        Check a single rule.

        Args:
            name: Rule name

        Returns:
            Alert if triggered, None otherwise
        """
        with self._lock:
            rule = self._rules.get(name)
            if not rule or not rule.enabled:
                return None

            # Check cooldown
            last = self._last_triggered.get(name)
            if last and datetime.utcnow() - last < rule.cooldown:
                return None

            # Check condition
            try:
                if rule.condition():
                    return self._create_alert(rule)
            except Exception as e:
                self.logger.error(f"Error checking rule {name}: {e}")

            return None

    def check_all(self) -> List[Alert]:
        """
        Check all rules.

        Returns:
            List of triggered alerts
        """
        alerts = []

        with self._lock:
            rule_names = list(self._rules.keys())

        for name in rule_names:
            alert = self.check_rule(name)
            if alert:
                alerts.append(alert)

        return alerts

    def send_alert(
        self,
        name: str,
        level: AlertLevel,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        category: AlertCategory = AlertCategory.SYSTEM
    ) -> Alert:
        """
        Manually send an alert.

        Args:
            name: Alert name
            level: Alert level
            message: Alert message
            details: Additional details
            category: Alert category

        Returns:
            Created alert
        """
        with self._lock:
            self._alert_counter += 1
            alert_id = f"alert_{self._alert_counter}"

            alert = Alert(
                id=alert_id,
                name=name,
                level=level,
                category=category,
                message=message,
                details=details or {}
            )

            self._active_alerts[alert_id] = alert
            self._history.append(alert)

            # Trim history
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]

        # Send notifications
        asyncio.create_task(self._send_notifications(alert))

        self.logger.info(f"Alert created: [{level.value}] {name}: {message}")

        return alert

    def acknowledge(
        self,
        alert_id: str,
        acknowledged_by: Optional[str] = None
    ) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID
            acknowledged_by: Who acknowledged

        Returns:
            True if acknowledged
        """
        with self._lock:
            alert = self._active_alerts.get(alert_id)
            if alert and not alert.acknowledged:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.utcnow()
                return True
            return False

    def suppress(
        self,
        name: str,
        duration: timedelta = timedelta(hours=1)
    ) -> None:
        """
        Suppress alerts for a rule.

        Args:
            name: Rule name
            duration: Suppression duration
        """
        with self._lock:
            self._last_triggered[name] = datetime.utcnow() + duration

    def get_active_alerts(
        self,
        level: Optional[AlertLevel] = None,
        category: Optional[AlertCategory] = None
    ) -> List[Alert]:
        """
        Get active (unacknowledged) alerts.

        Args:
            level: Filter by level
            category: Filter by category

        Returns:
            List of active alerts
        """
        with self._lock:
            alerts = [a for a in self._active_alerts.values() if not a.acknowledged]

        if level:
            alerts = [a for a in alerts if a.level == level]
        if category:
            alerts = [a for a in alerts if a.category == category]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_history(
        self,
        limit: int = 100,
        level: Optional[AlertLevel] = None,
        since: Optional[datetime] = None
    ) -> List[Alert]:
        """
        Get alert history.

        Args:
            limit: Maximum alerts
            level: Filter by level
            since: Filter by time

        Returns:
            List of alerts
        """
        with self._lock:
            alerts = self._history.copy()

        if level:
            alerts = [a for a in alerts if a.level == level]
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]

        return alerts[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self._lock:
            total = len(self._history)
            by_level = defaultdict(int)
            by_category = defaultdict(int)

            for alert in self._history:
                by_level[alert.level.value] += 1
                by_category[alert.category.value] += 1

            return {
                'total_alerts': total,
                'active_alerts': len(self.get_active_alerts()),
                'rules_count': len(self._rules),
                'by_level': dict(by_level),
                'by_category': dict(by_category)
            }

    def clear_resolved(self) -> int:
        """Clear acknowledged alerts."""
        count = 0
        with self._lock:
            to_remove = [
                aid for aid, a in self._active_alerts.items()
                if a.acknowledged
            ]
            for aid in to_remove:
                del self._active_alerts[aid]
                count += 1

        return count

    # Private methods

    def _create_alert(self, rule: AlertRule) -> Alert:
        """Create an alert from a rule."""
        with self._lock:
            self._alert_counter += 1
            alert_id = f"alert_{self._alert_counter}"

            details = {}
            if rule.details_func:
                try:
                    details = rule.details_func() or {}
                except Exception as e:
                    self.logger.error(f"Error getting details: {e}")

            alert = Alert(
                id=alert_id,
                name=rule.name,
                level=rule.level,
                category=rule.category,
                message=rule.message,
                details=details
            )

            self._active_alerts[alert_id] = alert
            self._history.append(alert)
            self._last_triggered[rule.name] = datetime.utcnow()

            # Trim history
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]

        # Send notifications
        asyncio.create_task(self._send_notifications(alert))

        return alert

    async def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        for channel in self._channels:
            try:
                await channel.send(alert)
            except Exception as e:
                self.logger.error(f"Notification failed on {channel.name}: {e}")


__all__ = [
    'AlertLevel',
    'AlertCategory',
    'Alert',
    'AlertRule',
    'NotificationChannel',
    'LogChannel',
    'WebhookChannel',
    'AlertManager'
]
