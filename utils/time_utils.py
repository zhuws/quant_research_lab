"""
Time utility functions for Quant Research Lab.
Provides time-related utilities for trading operations.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple
import pytz
from enum import Enum


class MarketSession(Enum):
    """Market session types."""
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    POST_MARKET = "post_market"
    CLOSED = "closed"


def get_utc_now() -> datetime:
    """
    Get current UTC datetime.

    Returns:
        Current UTC datetime
    """
    return datetime.now(timezone.utc)


def to_utc(dt: datetime) -> datetime:
    """
    Convert datetime to UTC.

    Args:
        dt: Datetime to convert

    Returns:
        UTC datetime
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_timestamp(dt: datetime) -> int:
    """
    Convert datetime to Unix timestamp in milliseconds.

    Args:
        dt: Datetime to convert

    Returns:
        Unix timestamp in milliseconds
    """
    return int(dt.timestamp() * 1000)


def from_timestamp(ts: int) -> datetime:
    """
    Convert Unix timestamp to datetime.

    Args:
        ts: Unix timestamp in milliseconds

    Returns:
        Datetime object
    """
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)


def get_timeframe_ms(timeframe: str) -> int:
    """
    Get timeframe duration in milliseconds.

    Args:
        timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d')

    Returns:
        Duration in milliseconds
    """
    units = {
        's': 1000,
        'm': 1000 * 60,
        'h': 1000 * 60 * 60,
        'd': 1000 * 60 * 60 * 24,
        'w': 1000 * 60 * 60 * 24 * 7
    }

    unit = timeframe[-1].lower()
    value = int(timeframe[:-1])

    return value * units.get(unit, 0)


def align_to_timeframe(
    dt: datetime,
    timeframe: str
) -> datetime:
    """
    Align datetime to timeframe boundary.

    Args:
        dt: Datetime to align
        timeframe: Timeframe to align to

    Returns:
        Aligned datetime
    """
    tf_ms = get_timeframe_ms(timeframe)
    ts = to_timestamp(dt)
    aligned_ts = (ts // tf_ms) * tf_ms
    return from_timestamp(aligned_ts)


def generate_time_range(
    start: datetime,
    end: datetime,
    timeframe: str
) -> List[datetime]:
    """
    Generate list of datetimes between start and end.

    Args:
        start: Start datetime
        end: End datetime
        timeframe: Interval timeframe

    Returns:
        List of datetimes
    """
    result = []
    current = align_to_timeframe(start, timeframe)
    tf_ms = get_timeframe_ms(timeframe)
    end_ts = to_timestamp(end)

    while to_timestamp(current) <= end_ts:
        result.append(current)
        current = from_timestamp(to_timestamp(current) + tf_ms)

    return result


def get_next_candle_time(
    timeframe: str,
    offset: int = 1
) -> datetime:
    """
    Get next candle close time.

    Args:
        timeframe: Timeframe string
        offset: Number of candles ahead

    Returns:
        Next candle close time
    """
    now = get_utc_now()
    current_aligned = align_to_timeframe(now, timeframe)
    tf_ms = get_timeframe_ms(timeframe)
    return from_timestamp(to_timestamp(current_aligned) + tf_ms * offset)


def is_candle_close(
    timeframe: str,
    tolerance_ms: int = 100
) -> bool:
    """
    Check if current time is near a candle close.

    Args:
        timeframe: Timeframe string
        tolerance_ms: Tolerance in milliseconds

    Returns:
        True if near candle close
    """
    now = get_utc_now()
    next_candle = get_next_candle_time(timeframe, 1)
    current_aligned = align_to_timeframe(next_candle, timeframe)

    time_to_close = to_timestamp(next_candle) - to_timestamp(now)
    return time_to_close <= tolerance_ms


def get_funding_times() -> List[datetime]:
    """
    Get funding rate times for the current day.
    Funding occurs at 00:00, 08:00, 16:00 UTC.

    Returns:
        List of funding times for current day
    """
    now = get_utc_now()
    times = []

    for hour in [0, 8, 16]:
        funding_time = now.replace(
            hour=hour, minute=0, second=0, microsecond=0
        )
        times.append(funding_time)

    return times


def get_next_funding_time() -> datetime:
    """
    Get next funding rate time.

    Returns:
        Next funding time
    """
    now = get_utc_now()
    funding_hours = [0, 8, 16]

    current_hour = now.hour
    next_hour = None

    for hour in funding_hours:
        if hour > current_hour:
            next_hour = hour
            break

    if next_hour is None:
        # Next funding is tomorrow at 00:00
        next_funding = now.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)
    else:
        next_funding = now.replace(
            hour=next_hour, minute=0, second=0, microsecond=0
        )

    return next_funding


def time_until_funding() -> timedelta:
    """
    Get time until next funding.

    Returns:
        Time delta until next funding
    """
    next_funding = get_next_funding_time()
    return next_funding - get_utc_now()


def get_trading_hours(
    timezone_str: str = 'UTC'
) -> Tuple[datetime, datetime]:
    """
    Get trading day start and end times.

    Args:
        timezone_str: Timezone string

    Returns:
        Tuple of (start_time, end_time)
    """
    tz = pytz.timezone(timezone_str)
    now = datetime.now(tz)

    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    return start, end


def format_duration(td: timedelta) -> str:
    """
    Format timedelta to human-readable string.

    Args:
        td: Timedelta to format

    Returns:
        Formatted string
    """
    total_seconds = int(td.total_seconds())

    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)


def is_weekend(dt: Optional[datetime] = None) -> bool:
    """
    Check if datetime is a weekend.

    Args:
        dt: Datetime to check (default: now)

    Returns:
        True if weekend
    """
    if dt is None:
        dt = get_utc_now()

    return dt.weekday() >= 5


def get_session(dt: Optional[datetime] = None) -> MarketSession:
    """
    Get market session for datetime.

    Args:
        dt: Datetime to check (default: now)

    Returns:
        MarketSession enum value
    """
    if dt is None:
        dt = get_utc_now()

    # Crypto markets are 24/7, always regular session
    return MarketSession.REGULAR
