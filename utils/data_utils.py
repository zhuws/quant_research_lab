"""
Data utility functions for Quant Research Lab.
Provides common data manipulation and transformation utilities.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any
from datetime import datetime, timedelta


def ensure_datetime(
    data: Union[pd.DataFrame, pd.Series],
    column: str = 'timestamp'
) -> Union[pd.DataFrame, pd.Series]:
    """
    Ensure a column is in datetime format.

    Args:
        data: DataFrame or Series
        column: Column name to convert

    Returns:
        Data with converted column
    """
    if isinstance(data, pd.DataFrame):
        if column in data.columns:
            data[column] = pd.to_datetime(data[column])
    return data


def resample_ohlcv(
    df: pd.DataFrame,
    target_freq: str = '5min',
    datetime_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Resample OHLCV data to a different frequency.

    Args:
        df: DataFrame with OHLCV data
        target_freq: Target frequency (e.g., '5min', '1h', '1d')
        datetime_col: Name of datetime column

    Returns:
        Resampled DataFrame
    """
    df = df.copy()

    if datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index(datetime_col)

    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    # Only aggregate columns that exist
    available_rules = {k: v for k, v in agg_rules.items() if k in df.columns}

    resampled = df.resample(target_freq).agg(available_rules)
    resampled = resampled.dropna()

    return resampled.reset_index()


def calculate_vwap(
    df: pd.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    volume_col: str = 'volume'
) -> np.ndarray:
    """
    Calculate Volume Weighted Average Price.

    Args:
        df: DataFrame with price and volume data
        high_col: High price column name
        low_col: Low price column name
        close_col: Close price column name
        volume_col: Volume column name

    Returns:
        Array of VWAP values
    """
    typical_price = (
        df[high_col] + df[low_col] + df[close_col]
    ) / 3

    cum_volume = df[volume_col].cumsum()
    cum_tp_volume = (typical_price * df[volume_col]).cumsum()

    return cum_tp_volume / cum_volume


def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close'
) -> np.ndarray:
    """
    Calculate Average True Range.

    Args:
        df: DataFrame with OHLC data
        period: ATR period
        high_col: High price column name
        low_col: Low price column name
        close_col: Close price column name

    Returns:
        Array of ATR values
    """
    high = df[high_col].values
    low = df[low_col].values
    close = df[close_col].values

    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))

    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First value doesn't have previous close

    atr = np.zeros_like(tr)
    atr[:period] = np.nan
    atr[period - 1] = np.mean(tr[:period])

    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


def fill_missing_values(
    df: pd.DataFrame,
    method: str = 'ffill',
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Fill missing values in DataFrame.

    Args:
        df: DataFrame to fill
        method: Fill method ('ffill', 'bfill', 'interpolate', 'mean', 'median')
        limit: Maximum number of consecutive NaN values to fill

    Returns:
        DataFrame with filled values
    """
    df = df.copy()

    if method == 'ffill':
        df = df.ffill(limit=limit)
    elif method == 'bfill':
        df = df.bfill(limit=limit)
    elif method == 'interpolate':
        df = df.interpolate(limit=limit)
    elif method == 'mean':
        df = df.fillna(df.mean())
    elif method == 'median':
        df = df.fillna(df.median())

    return df


def detect_outliers(
    data: np.ndarray,
    method: str = 'zscore',
    threshold: float = 3.0
) -> np.ndarray:
    """
    Detect outliers in data.

    Args:
        data: Input data array
        method: Detection method ('zscore', 'iqr', 'mad')
        threshold: Threshold for outlier detection

    Returns:
        Boolean array indicating outliers
    """
    valid_mask = ~np.isnan(data)
    outliers = np.zeros(len(data), dtype=bool)

    if method == 'zscore':
        mean = np.nanmean(data)
        std = np.nanstd(data, ddof=1)
        z_scores = np.abs((data - mean) / std)
        outliers = z_scores > threshold

    elif method == 'iqr':
        q1 = np.nanquantile(data, 0.25)
        q3 = np.nanquantile(data, 0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = (data < lower_bound) | (data > upper_bound)

    elif method == 'mad':
        median = np.nanmedian(data)
        mad = np.nanmedian(np.abs(data - median))
        modified_z = 0.6745 * (data - median) / mad if mad > 0 else np.zeros_like(data)
        outliers = np.abs(modified_z) > threshold

    return outliers


def align_dataframes(
    dfs: List[pd.DataFrame],
    datetime_col: str = 'timestamp',
    how: str = 'inner'
) -> List[pd.DataFrame]:
    """
    Align multiple DataFrames by datetime index.

    Args:
        dfs: List of DataFrames to align
        datetime_col: Name of datetime column
        how: Join type ('inner', 'outer', 'left', 'right')

    Returns:
        List of aligned DataFrames
    """
    if not dfs:
        return []

    aligned = []
    for df in dfs:
        df = df.copy()
        if datetime_col in df.columns:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df = df.set_index(datetime_col)
        aligned.append(df)

    # Get common index
    if how == 'inner':
        common_idx = aligned[0].index
        for df in aligned[1:]:
            common_idx = common_idx.intersection(df.index)
        aligned = [df.loc[common_idx] for df in aligned]
    elif how == 'outer':
        all_idx = aligned[0].index
        for df in aligned[1:]:
            all_idx = all_idx.union(df.index)
        aligned = [df.reindex(all_idx) for df in aligned]

    return aligned


def create_lagged_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int]
) -> pd.DataFrame:
    """
    Create lagged features for specified columns.

    Args:
        df: Input DataFrame
        columns: Columns to create lags for
        lags: List of lag periods

    Returns:
        DataFrame with added lagged columns
    """
    df = df.copy()

    for col in columns:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    return df


def calculate_returns(
    prices: pd.Series,
    method: str = 'log',
    periods: int = 1
) -> pd.Series:
    """
    Calculate returns from price series.

    Args:
        prices: Price series
        method: Return method ('log', 'simple')
        periods: Number of periods for return calculation

    Returns:
        Series of returns
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(periods))
    else:
        returns = (prices - prices.shift(periods)) / prices.shift(periods)

    return returns


def normalize_dataframe(
    df: pd.DataFrame,
    method: str = 'zscore',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Normalize DataFrame columns.

    Args:
        df: Input DataFrame
        method: Normalization method ('zscore', 'minmax', 'robust')
        columns: Columns to normalize (if None, normalize all numeric)

    Returns:
        Normalized DataFrame
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col in df.columns:
            if method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
            elif method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
            elif method == 'robust':
                median = df[col].median()
                iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                if iqr > 0:
                    df[col] = (df[col] - median) / iqr

    return df
