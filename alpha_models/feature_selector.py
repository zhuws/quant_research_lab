"""
Feature Selector for Quant Research Lab.
Selects optimal features for model training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.feature_selection import (
    mutual_info_regression,
    mutual_info_classif,
    SelectKBest,
    SelectPercentile,
    RFECV
)
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger


class FeatureSelector:
    """
    Feature selection for model training.

    Methods:
        - Correlation-based selection
        - Mutual information selection
        - Recursive feature elimination
        - Lasso-based selection
        - Tree-based importance selection
    """

    def __init__(
        self,
        target_column: str = 'target_return_10',
        max_features: int = 50,
        min_features: int = 10,
        correlation_threshold: float = 0.95,
        random_state: int = 42
    ):
        """
        Initialize Feature Selector.

        Args:
            target_column: Target column name
            max_features: Maximum features to select
            min_features: Minimum features to select
            correlation_threshold: Threshold for removing correlated features
            random_state: Random seed
        """
        self.logger = get_logger('feature_selector')
        self.target_column = target_column
        self.max_features = max_features
        self.min_features = min_features
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state

        self.selected_features_: List[str] = []
        self.feature_scores_: Dict[str, float] = {}

    def select_features(
        self,
        data: pd.DataFrame,
        method: str = 'mutual_info',
        task_type: str = 'regression'
    ) -> List[str]:
        """
        Select features using specified method.

        Args:
            data: DataFrame with features and target
            method: Selection method
            task_type: 'regression' or 'classification'

        Returns:
            List of selected feature names
        """
        self.logger.info(f"Selecting features using {method}")

        # Prepare data
        X = data.drop(columns=[self.target_column], errors='ignore')
        y = data[self.target_column] if self.target_column in data.columns else data.iloc[:, -1]

        # Exclude non-feature columns
        exclude_cols = {'timestamp', 'open', 'high', 'low', 'close', 'volume', self.target_column}
        feature_cols = [col for col in X.columns if col not in exclude_cols and not col.startswith('target')]
        X = X[feature_cols]

        # Remove highly correlated features
        X, removed_corr = self._remove_correlated(X)
        self.logger.info(f"Removed {len(removed_corr)} highly correlated features")

        # Select features
        if method == 'correlation':
            selected = self._correlation_selection(X, y)

        elif method == 'mutual_info':
            selected = self._mutual_info_selection(X, y, task_type)

        elif method == 'tree_importance':
            selected = self._tree_importance_selection(X, y, task_type)

        elif method == 'lasso':
            selected = self._lasso_selection(X, y)

        elif method == 'rfe':
            selected = self._rfe_selection(X, y, task_type)

        elif method == 'combined':
            selected = self._combined_selection(X, y, task_type)

        else:
            selected = self._mutual_info_selection(X, y, task_type)

        self.selected_features_ = selected[:self.max_features]
        self.logger.info(f"Selected {len(self.selected_features_)} features")

        return self.selected_features_

    def _remove_correlated(
        self,
        X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features."""
        corr_matrix = X.corr().abs()

        # Find pairs of highly correlated features
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to drop
        to_drop = [
            column for column in upper.columns
            if any(upper[column] > self.correlation_threshold)
        ]

        X_reduced = X.drop(columns=to_drop)

        return X_reduced, to_drop

    def _correlation_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[str]:
        """Select features based on correlation with target."""
        correlations = X.corrwith(y).abs()
        correlations = correlations.sort_values(ascending=False)

        self.feature_scores_ = correlations.to_dict()

        return correlations.head(self.max_features).index.tolist()

    def _mutual_info_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str
    ) -> List[str]:
        """Select features using mutual information."""
        X_filled = X.fillna(0)

        if task_type == 'regression':
            mi = mutual_info_regression(X_filled, y, random_state=self.random_state)
        else:
            mi = mutual_info_classif(X_filled, y, random_state=self.random_state)

        mi_series = pd.Series(mi, index=X.columns)
        mi_series = mi_series.sort_values(ascending=False)

        self.feature_scores_ = mi_series.to_dict()

        return mi_series.head(self.max_features).index.tolist()

    def _tree_importance_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str
    ) -> List[str]:
        """Select features using tree-based importance."""
        X_filled = X.fillna(0)

        if task_type == 'regression':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=-1
            )

        model.fit(X_filled, y)

        importance = pd.Series(model.feature_importances_, index=X.columns)
        importance = importance.sort_values(ascending=False)

        self.feature_scores_ = importance.to_dict()

        return importance.head(self.max_features).index.tolist()

    def _lasso_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[str]:
        """Select features using Lasso regularization."""
        X_filled = X.fillna(0)

        # Standardize
        X_std = (X_filled - X_filled.mean()) / (X_filled.std() + 1e-8)

        # Fit Lasso with cross-validation
        lasso = LassoCV(cv=5, random_state=self.random_state, n_jobs=-1)
        lasso.fit(X_std, y)

        # Get non-zero coefficients
        coef = pd.Series(np.abs(lasso.coef_), index=X.columns)
        coef = coef[coef > 0].sort_values(ascending=False)

        self.feature_scores_ = coef.to_dict()

        if len(coef) < self.min_features:
            # Fall back to correlation if too few features
            return self._correlation_selection(X, y)

        return coef.head(self.max_features).index.tolist()

    def _rfe_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str
    ) -> List[str]:
        """Select features using recursive feature elimination."""
        from sklearn.linear_model import Ridge

        X_filled = X.fillna(0)

        # Use Ridge as the estimator for speed
        estimator = Ridge()

        rfe = RFECV(
            estimator=estimator,
            step=0.1,
            cv=3,
            min_features_to_select=self.min_features,
            n_jobs=-1
        )

        rfe.fit(X_filled, y)

        selected = X.columns[rfe.support_].tolist()

        # Add ranking as scores
        self.feature_scores_ = {
            col: 1.0 / (rfe.ranking_[i] + 1)
            for i, col in enumerate(X.columns)
        }

        return selected[:self.max_features]

    def _combined_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str
    ) -> List[str]:
        """Combine multiple selection methods."""
        # Get selections from different methods
        corr_features = set(self._correlation_selection(X, y))
        mi_features = set(self._mutual_info_selection(X, y, task_type))
        tree_features = set(self._tree_importance_selection(X, y, task_type))

        # Count occurrences
        feature_counts = {}
        for feature in corr_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

        for feature in mi_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

        for feature in tree_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

        # Sort by count, then by mutual information
        sorted_features = sorted(
            feature_counts.keys(),
            key=lambda x: (feature_counts[x], self.feature_scores_.get(x, 0)),
            reverse=True
        )

        return sorted_features[:self.max_features]

    def get_feature_scores(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns:
            DataFrame with feature scores
        """
        if not self.feature_scores_:
            return pd.DataFrame()

        scores = pd.DataFrame([
            {'feature': k, 'score': v}
            for k, v in self.feature_scores_.items()
        ]).sort_values('score', ascending=False)

        return scores

    def transform(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transform data to selected features only.

        Args:
            data: DataFrame with all features

        Returns:
            DataFrame with selected features only
        """
        if not self.selected_features_:
            raise ValueError("No features selected. Call select_features() first.")

        available_features = [f for f in self.selected_features_ if f in data.columns]

        if len(available_features) < len(self.selected_features_):
            missing = set(self.selected_features_) - set(available_features)
            self.logger.warning(f"Missing features: {missing}")

        return data[available_features]

    def fit_transform(
        self,
        data: pd.DataFrame,
        method: str = 'mutual_info',
        task_type: str = 'regression'
    ) -> pd.DataFrame:
        """
        Fit selector and transform data.

        Args:
            data: DataFrame with features and target
            method: Selection method
            task_type: 'regression' or 'classification'

        Returns:
            DataFrame with selected features
        """
        self.select_features(data, method, task_type)
        return self.transform(data)


def select_features(
    data: pd.DataFrame,
    target_column: str = 'target_return_10',
    max_features: int = 50,
    method: str = 'mutual_info'
) -> List[str]:
    """
    Convenience function for feature selection.

    Args:
        data: DataFrame with features and target
        target_column: Target column name
        max_features: Maximum features to select
        method: Selection method

    Returns:
        List of selected feature names
    """
    selector = FeatureSelector(
        target_column=target_column,
        max_features=max_features
    )

    return selector.select_features(data, method=method)
