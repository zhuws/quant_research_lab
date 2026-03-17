"""
Feature Pipeline for Quant Research Lab.
Main orchestrator for generating all trading features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from features.technical_features import TechnicalFeatureGenerator
from features.volatility_features import VolatilityFeatureGenerator
from features.orderflow_features import OrderFlowFeatureGenerator
from features.liquidity_features import LiquidityFeatureGenerator
from features.cross_exchange_features import CrossExchangeFeatureGenerator


class FeaturePipeline:
    """
    Main feature generation pipeline.

    Orchestrates multiple feature generators to produce 150+ features:
        - Technical features: ~70 indicators
        - Volatility features: ~35 indicators
        - Order flow features: ~30 indicators
        - Liquidity features: ~25 indicators
        - Cross-exchange features: ~20 indicators

    Total: ~180 features
    """

    def __init__(
        self,
        include_technical: bool = True,
        include_volatility: bool = True,
        include_orderflow: bool = True,
        include_liquidity: bool = True,
        include_cross_exchange: bool = True,
        custom_config: Optional[Dict] = None
    ):
        """
        Initialize Feature Pipeline.

        Args:
            include_technical: Include technical features
            include_volatility: Include volatility features
            include_orderflow: Include order flow features
            include_liquidity: Include liquidity features
            include_cross_exchange: Include cross-exchange features
            custom_config: Custom configuration for feature generators
        """
        self.logger = get_logger('feature_pipeline')
        self.custom_config = custom_config or {}

        self.include_technical = include_technical
        self.include_volatility = include_volatility
        self.include_orderflow = include_orderflow
        self.include_liquidity = include_liquidity
        self.include_cross_exchange = include_cross_exchange

        # Initialize feature generators
        self._init_generators()

        # Track feature count
        self.feature_count = 0
        self.feature_names = []

    def _init_generators(self):
        """Initialize feature generators with optional custom config."""
        tech_config = self.custom_config.get('technical', {})
        vol_config = self.custom_config.get('volatility', {})
        flow_config = self.custom_config.get('orderflow', {})
        liq_config = self.custom_config.get('liquidity', {})
        cross_config = self.custom_config.get('cross_exchange', {})

        self.technical_generator = TechnicalFeatureGenerator(**tech_config)
        self.volatility_generator = VolatilityFeatureGenerator(**vol_config)
        self.orderflow_generator = OrderFlowFeatureGenerator(**flow_config)
        self.liquidity_generator = LiquidityFeatureGenerator(**liq_config)
        self.cross_exchange_generator = CrossExchangeFeatureGenerator(**cross_config)

    def generate_features(
        self,
        df: pd.DataFrame,
        other_exchange_df: Optional[pd.DataFrame] = None,
        orderbook_df: Optional[pd.DataFrame] = None,
        trades_df: Optional[pd.DataFrame] = None,
        funding_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate all features from market data.

        Args:
            df: DataFrame with OHLCV data
            other_exchange_df: Optional data from another exchange
            orderbook_df: Optional orderbook data
            trades_df: Optional trade data
            funding_df: Optional funding rate data

        Returns:
            DataFrame with all features added
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided")
            return df

        self.logger.info(f"Generating features from {len(df)} bars")

        features = df.copy()

        # Generate features by category
        if self.include_technical:
            self.logger.info("Generating technical features...")
            features = self.technical_generator.generate_features(features)
            self.logger.info(f"Technical features: {len(features.columns) - len(df.columns)}")

        base_count = len(features.columns)

        if self.include_volatility:
            self.logger.info("Generating volatility features...")
            features = self.volatility_generator.generate_features(features)
            self.logger.info(f"Volatility features: {len(features.columns) - base_count}")

        base_count = len(features.columns)

        if self.include_orderflow:
            self.logger.info("Generating order flow features...")
            features = self.orderflow_generator.generate_features(features, trades_df)
            self.logger.info(f"Order flow features: {len(features.columns) - base_count}")

        base_count = len(features.columns)

        if self.include_liquidity:
            self.logger.info("Generating liquidity features...")
            features = self.liquidity_generator.generate_features(features, orderbook_df)
            self.logger.info(f"Liquidity features: {len(features.columns) - base_count}")

        base_count = len(features.columns)

        if self.include_cross_exchange:
            self.logger.info("Generating cross-exchange features...")
            features = self.cross_exchange_generator.generate_features(
                features,
                other_exchange_df
            )
            if funding_df is not None:
                features = self.cross_exchange_generator.generate_funding_arbitrage_features(
                    features, funding_df
                )
            self.logger.info(f"Cross-exchange features: {len(features.columns) - base_count}")

        # Post-processing
        features = self._post_process_features(features)

        # Store feature information
        self.feature_count = len(features.columns) - len(df.columns)
        self.feature_names = [col for col in features.columns if col not in df.columns]

        self.logger.info(f"Total features generated: {self.feature_count}")

        return features

    def _post_process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process features (handle NaN, inf, etc.)."""
        # Replace inf values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Forward fill then backward fill for any remaining NaN
        # Only for feature columns, not the original OHLCV
        feature_cols = [col for col in df.columns if col not in
                       ['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        for col in feature_cols:
            # Forward fill
            df[col] = df[col].ffill()
            # Backward fill
            df[col] = df[col].bfill()
            # Fill any remaining with 0
            df[col] = df[col].fillna(0)

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self.feature_names

    def get_feature_count(self) -> int:
        """Get count of generated features."""
        return self.feature_count

    def generate_target_labels(
        self,
        df: pd.DataFrame,
        horizons: List[int] = None,
        label_type: str = 'return'
    ) -> pd.DataFrame:
        """
        Generate target labels for ML training.

        Args:
            df: DataFrame with OHLCV data
            horizons: Prediction horizons (bars)
            label_type: Type of label ('return', 'direction', 'volatility')

        Returns:
            DataFrame with target labels added
        """
        horizons = horizons or [1, 5, 10, 20]
        df = df.copy()

        for horizon in horizons:
            if label_type == 'return':
                # Future return
                df[f'target_return_{horizon}'] = (
                    np.log(df['close'].shift(-horizon) / df['close'])
                )

                # Future return direction
                df[f'target_direction_{horizon}'] = (
                    df[f'target_return_{horizon}'] > 0
                ).astype(int)

            elif label_type == 'volatility':
                # Future volatility
                df[f'target_volatility_{horizon}'] = (
                    df['close'].pct_change().shift(-horizon).rolling(horizon).std()
                )

            elif label_type == 'max_return':
                # Maximum favorable excursion
                future_high = df['high'].shift(-1).rolling(horizon).max()
                df[f'target_max_return_{horizon}'] = (
                    np.log(future_high / df['close'])
                )

                # Maximum adverse excursion
                future_low = df['low'].shift(-1).rolling(horizon).min()
                df[f'target_min_return_{horizon}'] = (
                    np.log(future_low / df['close'])
                )

        return df

    def select_features(
        self,
        df: pd.DataFrame,
        method: str = 'correlation',
        target_col: str = 'target_return_10',
        top_n: int = 50
    ) -> List[str]:
        """
        Select top features based on specified method.

        Args:
            df: DataFrame with features and target
            method: Selection method ('correlation', 'mutual_info')
            target_col: Target column name
            top_n: Number of features to select

        Returns:
            List of selected feature names
        """
        if target_col not in df.columns:
            self.logger.warning(f"Target column {target_col} not found")
            return []

        feature_cols = [col for col in df.columns if col not in
                       ['timestamp', 'open', 'high', 'low', 'close', 'volume'] and
                       not col.startswith('target')]

        if method == 'correlation':
            # Calculate correlation with target
            correlations = df[feature_cols + [target_col]].corr()[target_col]
            correlations = correlations.drop(target_col).abs()
            top_features = correlations.nlargest(top_n).index.tolist()

        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression

            # Prepare data
            X = df[feature_cols].fillna(0)
            y = df[target_col].fillna(0)

            # Calculate mutual information
            mi = mutual_info_regression(X, y)
            mi_series = pd.Series(mi, index=feature_cols)
            top_features = mi_series.nlargest(top_n).index.tolist()

        else:
            self.logger.warning(f"Unknown method: {method}")
            return feature_cols[:top_n]

        return top_features

    def get_feature_importance(
        self,
        df: pd.DataFrame,
        target_col: str = 'target_return_10'
    ) -> pd.DataFrame:
        """
        Get feature importance ranking.

        Args:
            df: DataFrame with features and target
            target_col: Target column name

        Returns:
            DataFrame with feature importance scores
        """
        feature_cols = [col for col in df.columns if col not in
                       ['timestamp', 'open', 'high', 'low', 'close', 'volume'] and
                       not col.startswith('target')]

        if target_col not in df.columns:
            return pd.DataFrame()

        # Correlation-based importance
        correlations = df[feature_cols + [target_col]].corr()[target_col]
        correlations = correlations.drop(target_col)

        importance_df = pd.DataFrame({
            'feature': correlations.index,
            'correlation': correlations.values,
            'abs_correlation': correlations.abs().values
        }).sort_values('abs_correlation', ascending=False)

        return importance_df

    def normalize_features(
        self,
        df: pd.DataFrame,
        method: str = 'zscore',
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Normalize features.

        Args:
            df: DataFrame with features
            method: Normalization method ('zscore', 'minmax', 'robust')
            feature_cols: Columns to normalize (default: all features)

        Returns:
            DataFrame with normalized features
        """
        df = df.copy()

        if feature_cols is None:
            feature_cols = [col for col in df.columns if col not in
                          ['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        for col in feature_cols:
            if method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                df[col] = (df[col] - mean) / std.replace(0, 1)

            elif method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val).replace(0, 1)

            elif method == 'robust':
                median = df[col].median()
                iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                df[col] = (df[col] - median) / iqr.replace(0, 1)

        return df

    def save_features(
        self,
        df: pd.DataFrame,
        filepath: str,
        format: str = 'parquet'
    ) -> None:
        """
        Save features to file.

        Args:
            df: DataFrame with features
            filepath: Output file path
            format: Output format ('parquet', 'csv', 'hdf')
        """
        if format == 'parquet':
            df.to_parquet(filepath, index=False)
        elif format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'hdf':
            df.to_hdf(filepath, key='features', mode='w')
        else:
            raise ValueError(f"Unknown format: {format}")

        self.logger.info(f"Saved features to {filepath}")

    def load_features(self, filepath: str) -> pd.DataFrame:
        """
        Load features from file.

        Args:
            filepath: Input file path

        Returns:
            DataFrame with features
        """
        if filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.h5') or filepath.endswith('.hdf'):
            df = pd.read_hdf(filepath, key='features')
        else:
            raise ValueError(f"Unknown file format: {filepath}")

        self.logger.info(f"Loaded features from {filepath}")
        return df


def build_features(
    df: pd.DataFrame,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Convenience function to build features.

    Args:
        df: DataFrame with OHLCV data
        config: Optional configuration dict

    Returns:
        DataFrame with features
    """
    pipeline = FeaturePipeline(custom_config=config)
    return pipeline.generate_features(df)
