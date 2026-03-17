"""
Mathematical utility functions for Quant Research Lab.
Provides common mathematical operations used in quantitative analysis.
"""

import numpy as np
from typing import Union, Optional
from decimal import Decimal, getcontext

# Set precision for decimal operations
getcontext().prec = 28


def safe_divide(
    numerator: Union[float, np.ndarray],
    denominator: Union[float, np.ndarray],
    fill_value: float = 0.0
) -> Union[float, np.ndarray]:
    """
    Safely divide two values, handling division by zero.

    Args:
        numerator: Value to divide
        denominator: Value to divide by
        fill_value: Value to return if denominator is zero

    Returns:
        Result of division or fill_value if denominator is zero
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            denominator != 0,
            numerator / denominator,
            fill_value
        )
    return result


def rolling_std(
    data: np.ndarray,
    window: int,
    min_periods: int = 1
) -> np.ndarray:
    """
    Calculate rolling standard deviation.

    Args:
        data: Input data array
        window: Window size
        min_periods: Minimum number of observations required

    Returns:
        Array of rolling standard deviations
    """
    result = np.full_like(data, np.nan, dtype=float)

    for i in range(len(data)):
        start_idx = max(0, i - window + 1)
        window_data = data[start_idx:i + 1]

        if len(window_data) >= min_periods:
            result[i] = np.std(window_data, ddof=1)

    return result


def rolling_mean(
    data: np.ndarray,
    window: int,
    min_periods: int = 1
) -> np.ndarray:
    """
    Calculate rolling mean.

    Args:
        data: Input data array
        window: Window size
        min_periods: Minimum number of observations required

    Returns:
        Array of rolling means
    """
    result = np.full_like(data, np.nan, dtype=float)

    for i in range(len(data)):
        start_idx = max(0, i - window + 1)
        window_data = data[start_idx:i + 1]

        if len(window_data) >= min_periods:
            result[i] = np.mean(window_data)

    return result


def exponential_moving_average(
    data: np.ndarray,
    span: int
) -> np.ndarray:
    """
    Calculate exponential moving average.

    Args:
        data: Input data array
        span: EMA span (periods)

    Returns:
        Array of EMA values
    """
    alpha = 2.0 / (span + 1.0)
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]

    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

    return ema


def log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Calculate logarithmic returns.

    Args:
        prices: Array of prices

    Returns:
        Array of log returns (same length as input, first value is 0)
    """
    result = np.zeros_like(prices, dtype=float)
    result[1:] = np.log(prices[1:] / prices[:-1])
    return result


def simple_returns(prices: np.ndarray) -> np.ndarray:
    """
    Calculate simple returns.

    Args:
        prices: Array of prices

    Returns:
        Array of simple returns (same length as input, first value is 0)
    """
    result = np.zeros_like(prices, dtype=float)
    result[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    return result


def zscore(
    data: np.ndarray,
    window: Optional[int] = None
) -> np.ndarray:
    """
    Calculate z-score.

    Args:
        data: Input data array
        window: Optional rolling window. If None, uses entire dataset.

    Returns:
        Array of z-scores
    """
    if window is None:
        mean = np.nanmean(data)
        std = np.nanstd(data, ddof=1)
        return safe_divide(data - mean, std, 0.0)

    result = np.full_like(data, np.nan, dtype=float)
    for i in range(window - 1, len(data)):
        window_data = data[i - window + 1:i + 1]
        mean = np.nanmean(window_data)
        std = np.nanstd(window_data, ddof=1)
        result[i] = safe_divide(data[i] - mean, std, 0.0)

    return result


def winsorize(
    data: Union[np.ndarray, 'pd.Series'],
    limits: tuple = (0.05, 0.05)
) -> Union[np.ndarray, 'pd.Series']:
    """
    Winsorize data by clipping extreme values.

    Args:
        data: Input data array or pandas Series
        limits: Tuple of (lower, upper) percentile limits

    Returns:
        Winsorized data (same type as input)
    """
    import pandas as pd

    lower_limit, upper_limit = limits

    # Handle pandas Series
    if isinstance(data, pd.Series):
        lower_quantile = data.quantile(lower_limit)
        upper_quantile = data.quantile(1 - upper_limit)
        return data.clip(lower=lower_quantile, upper=upper_quantile)

    # Handle numpy arrays
    lower_quantile = np.nanquantile(data, lower_limit)
    upper_quantile = np.nanquantile(data, 1 - upper_limit)

    return np.clip(data, lower_quantile, upper_quantile)


def rank(data: np.ndarray, method: str = 'average') -> np.ndarray:
    """
    Rank data values.

    Args:
        data: Input data array
        method: Ranking method ('average', 'min', 'max', 'dense', 'ordinal')

    Returns:
        Array of ranks
    """
    valid_mask = ~np.isnan(data)
    ranks = np.full_like(data, np.nan, dtype=float)

    if np.any(valid_mask):
        valid_data = data[valid_mask]
        temp = valid_data.argsort()
        ranks_temp = np.empty_like(temp, dtype=float)

        if method == 'average':
            ranks_temp[temp] = np.arange(1, len(temp) + 1)
        elif method == 'min':
            ranks_temp[temp] = np.arange(1, len(temp) + 1)
        elif method == 'max':
            ranks_temp[temp] = np.arange(len(temp), 0, -1)
        else:
            ranks_temp[temp] = np.arange(1, len(temp) + 1)

        ranks[valid_mask] = ranks_temp

    return ranks


def percentile_rank(
    data: np.ndarray,
    window: Optional[int] = None
) -> np.ndarray:
    """
    Calculate percentile rank.

    Args:
        data: Input data array
        window: Optional rolling window. If None, uses entire dataset.

    Returns:
        Array of percentile ranks (0-1)
    """
    if window is None:
        valid_mask = ~np.isnan(data)
        valid_data = data[valid_mask]
        result = np.full_like(data, np.nan, dtype=float)
        result[valid_mask] = np.searchsorted(
            np.sort(valid_data), valid_data
        ) / len(valid_data)
        return result

    result = np.full_like(data, np.nan, dtype=float)
    for i in range(window - 1, len(data)):
        window_data = data[i - window + 1:i + 1]
        valid_mask = ~np.isnan(window_data)
        if np.any(valid_mask):
            valid_data = window_data[valid_mask]
            sorted_data = np.sort(valid_data)
            result[i] = np.searchsorted(sorted_data, data[i]) / len(valid_data)

    return result


def corr(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[int] = None
) -> Union[float, np.ndarray]:
    """
    Calculate correlation coefficient.

    Args:
        x: First data array
        y: Second data array
        window: Optional rolling window

    Returns:
        Correlation coefficient(s)
    """
    if window is None:
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if np.sum(valid_mask) < 2:
            return np.nan
        return np.corrcoef(x[valid_mask], y[valid_mask])[0, 1]

    result = np.full_like(x, np.nan, dtype=float)
    for i in range(window - 1, len(x)):
        x_window = x[i - window + 1:i + 1]
        y_window = y[i - window + 1:i + 1]
        valid_mask = ~(np.isnan(x_window) | np.isnan(y_window))

        if np.sum(valid_mask) >= 2:
            result[i] = np.corrcoef(
                x_window[valid_mask],
                y_window[valid_mask]
            )[0, 1]

    return result


def information_coefficient(
    predictions: np.ndarray,
    actuals: np.ndarray,
    method: str = 'spearman'
) -> float:
    """
    Calculate Information Coefficient (IC).

    Args:
        predictions: Predicted values
        actuals: Actual values
        method: Correlation method ('spearman' or 'pearson')

    Returns:
        IC value
    """
    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))

    if np.sum(valid_mask) < 2:
        return np.nan

    pred_valid = predictions[valid_mask]
    actual_valid = actuals[valid_mask]

    if method == 'spearman':
        pred_rank = rank(pred_valid)
        actual_rank = rank(actual_valid)
        return np.corrcoef(pred_rank, actual_rank)[0, 1]
    else:
        return np.corrcoef(pred_valid, actual_valid)[0, 1]
