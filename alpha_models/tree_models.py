"""
Tree-based Models for Quant Research Lab.
LightGBM and XGBoost implementations for alpha prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from alpha_models.base_model import BaseModel, ModelPrediction, ModelMetrics


class LightGBMModel(BaseModel):
    """
    LightGBM model for alpha prediction.

    Supports both regression and classification tasks.
    Optimized for fast training and low memory usage.
    """

    def __init__(
        self,
        name: str = 'lightgbm',
        task_type: str = 'regression',
        target_column: str = 'target_return_10',
        feature_columns: Optional[List[str]] = None,
        random_state: int = 42,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.01,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        early_stopping_rounds: int = 50,
        **kwargs
    ):
        """
        Initialize LightGBM Model.

        Args:
            name: Model name
            task_type: 'regression' or 'classification'
            target_column: Target column name
            feature_columns: List of feature columns
            random_state: Random seed
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            num_leaves: Number of leaves
            min_child_samples: Minimum samples per leaf
            subsample: Subsample ratio
            colsample_bytree: Column sample ratio
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            early_stopping_rounds: Early stopping rounds
            **kwargs: Additional LightGBM parameters
        """
        super().__init__(
            name=name,
            target_column=target_column,
            feature_columns=feature_columns,
            random_state=random_state
        )

        self.task_type = task_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.extra_params = kwargs

        # Store best iteration
        self.best_iteration_ = None

    def _get_lgbm_params(self) -> Dict[str, Any]:
        """Get LightGBM parameters."""
        params = {
            'objective': 'regression' if self.task_type == 'regression' else 'binary',
            'metric': 'rmse' if self.task_type == 'regression' else 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'min_child_samples': self.min_child_samples,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1,
            **self.extra_params
        }
        return params

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'LightGBMModel':
        """
        Fit the LightGBM model.

        Args:
            X: Features DataFrame
            y: Target Series
            eval_set: Optional evaluation set (X_val, y_val)
            **kwargs: Additional fit parameters

        Returns:
            self
        """
        import lightgbm as lgb

        self.logger.info(f"Fitting LightGBM model with {len(X)} samples")

        # Prepare features and target
        X_array = self._prepare_features(X, fit=True)
        y_array = self._prepare_target(y, self.task_type)

        # Create dataset
        train_data = lgb.Dataset(X_array, label=y_array, feature_name=self.feature_columns)

        # Prepare validation data
        valid_data = None
        if eval_set is not None:
            X_val = self._prepare_features(eval_set[0])
            y_val = self._prepare_target(eval_set[1], self.task_type)
            valid_data = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_columns)

        # Train model
        callbacks = [lgb.log_evaluation(period=100)]

        if valid_data is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=self.early_stopping_rounds))

        self.model = lgb.train(
            self._get_lgbm_params(),
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[valid_data] if valid_data else None,
            callbacks=callbacks
        )

        self.best_iteration_ = self.model.best_iteration
        self.is_fitted = True

        # Get feature importance
        self.feature_importance_ = dict(zip(
            self.feature_columns,
            self.model.feature_importance(importance_type='gain')
        ))

        # Calculate training metrics
        self.training_metrics_ = self.evaluate(X, y)

        self.logger.info(f"Model trained. Best iteration: {self.best_iteration_}")
        self.logger.info(f"Training IC: {self.training_metrics_.ic:.4f}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features DataFrame

        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_array = self._prepare_features(X)
        predictions = self.model.predict(X_array, num_iteration=self.best_iteration_)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.

        Args:
            X: Features DataFrame

        Returns:
            Probability array
        """
        predictions = self.predict(X)

        if self.task_type == 'regression':
            # For regression, return predictions as-is
            return predictions
        else:
            # For binary classification, return probability of positive class
            return predictions

    def _get_model_state(self) -> Any:
        """Get model-specific state."""
        return {
            'model': self.model,
            'best_iteration': self.best_iteration_,
            'params': self._get_lgbm_params()
        }

    def _set_model_state(self, state: Any) -> None:
        """Set model-specific state."""
        if state:
            self.model = state.get('model')
            self.best_iteration_ = state.get('best_iteration')


class XGBoostModel(BaseModel):
    """
    XGBoost model for alpha prediction.

    Supports both regression and classification tasks.
    Known for high performance and robust predictions.
    """

    def __init__(
        self,
        name: str = 'xgboost',
        task_type: str = 'regression',
        target_column: str = 'target_return_10',
        feature_columns: Optional[List[str]] = None,
        random_state: int = 42,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.01,
        min_child_weight: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        gamma: float = 0,
        early_stopping_rounds: int = 50,
        **kwargs
    ):
        """
        Initialize XGBoost Model.

        Args:
            name: Model name
            task_type: 'regression' or 'classification'
            target_column: Target column name
            feature_columns: List of feature columns
            random_state: Random seed
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            min_child_weight: Minimum child weight
            subsample: Subsample ratio
            colsample_bytree: Column sample ratio
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            gamma: Minimum split loss
            early_stopping_rounds: Early stopping rounds
            **kwargs: Additional XGBoost parameters
        """
        super().__init__(
            name=name,
            target_column=target_column,
            feature_columns=feature_columns,
            random_state=random_state
        )

        self.task_type = task_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.early_stopping_rounds = early_stopping_rounds
        self.extra_params = kwargs

    def _get_xgb_params(self) -> Dict[str, Any]:
        """Get XGBoost parameters."""
        params = {
            'objective': 'reg:squarederror' if self.task_type == 'regression' else 'binary:logistic',
            'eval_metric': 'rmse' if self.task_type == 'regression' else 'auc',
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'gamma': self.gamma,
            'random_state': self.random_state,
            'n_jobs': -1,
            **self.extra_params
        }
        return params

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'XGBoostModel':
        """
        Fit the XGBoost model.

        Args:
            X: Features DataFrame
            y: Target Series
            eval_set: Optional evaluation set
            **kwargs: Additional fit parameters

        Returns:
            self
        """
        import xgboost as xgb

        self.logger.info(f"Fitting XGBoost model with {len(X)} samples")

        # Prepare features and target
        X_array = self._prepare_features(X, fit=True)
        y_array = self._prepare_target(y, self.task_type)

        # Prepare eval set
        eval_sets = []
        if eval_set is not None:
            X_val = self._prepare_features(eval_set[0])
            y_val = self._prepare_target(eval_set[1], self.task_type)
            eval_sets = [(X_val, y_val)]

        # Create and train model
        self.model = xgb.XGBRegressor if self.task_type == 'regression' else xgb.XGBClassifier
        self.model = xgb.XGBRegressor(**self._get_xgb_params(), n_estimators=self.n_estimators)

        self.model.fit(
            X_array,
            y_array,
            eval_set=eval_sets if eval_sets else None,
            verbose=100
        )

        self.is_fitted = True

        # Get feature importance
        self.feature_importance_ = dict(zip(
            self.feature_columns,
            self.model.feature_importances_
        ))

        # Calculate training metrics
        self.training_metrics_ = self.evaluate(X, y)

        self.logger.info(f"Model trained. Training IC: {self.training_metrics_.ic:.4f}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features DataFrame

        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_array = self._prepare_features(X)
        return self.model.predict(X_array)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.

        Args:
            X: Features DataFrame

        Returns:
            Probability array
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_array = self._prepare_features(X)

        if self.task_type == 'classification':
            return self.model.predict_proba(X_array)[:, 1]
        else:
            return self.predict(X)

    def _get_model_state(self) -> Any:
        """Get model-specific state."""
        return {
            'model': self.model,
            'params': self._get_xgb_params()
        }

    def _set_model_state(self, state: Any) -> None:
        """Set model-specific state."""
        if state:
            self.model = state.get('model')


class CatBoostModel(BaseModel):
    """
    CatBoost model for alpha prediction.

    Excellent for categorical features and robust to overfitting.
    """

    def __init__(
        self,
        name: str = 'catboost',
        task_type: str = 'regression',
        target_column: str = 'target_return_10',
        feature_columns: Optional[List[str]] = None,
        random_state: int = 42,
        iterations: int = 500,
        depth: int = 8,
        learning_rate: float = 0.01,
        l2_leaf_reg: float = 3.0,
        early_stopping_rounds: int = 50,
        **kwargs
    ):
        """
        Initialize CatBoost Model.

        Args:
            name: Model name
            task_type: 'regression' or 'classification'
            target_column: Target column name
            feature_columns: List of feature columns
            random_state: Random seed
            iterations: Number of trees
            depth: Tree depth
            learning_rate: Learning rate
            l2_leaf_reg: L2 regularization
            early_stopping_rounds: Early stopping rounds
            **kwargs: Additional CatBoost parameters
        """
        super().__init__(
            name=name,
            target_column=target_column,
            feature_columns=feature_columns,
            random_state=random_state
        )

        self.task_type = task_type
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.early_stopping_rounds = early_stopping_rounds
        self.extra_params = kwargs

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'CatBoostModel':
        """
        Fit the CatBoost model.

        Args:
            X: Features DataFrame
            y: Target Series
            eval_set: Optional evaluation set
            **kwargs: Additional fit parameters

        Returns:
            self
        """
        from catboost import CatBoostRegressor, CatBoostClassifier

        self.logger.info(f"Fitting CatBoost model with {len(X)} samples")

        # Prepare features and target
        X_array = self._prepare_features(X, fit=True)
        y_array = self._prepare_target(y, self.task_type)

        # Create model
        if self.task_type == 'regression':
            self.model = CatBoostRegressor(
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                l2_leaf_reg=self.l2_leaf_reg,
                random_state=self.random_state,
                verbose=100,
                **self.extra_params
            )
        else:
            self.model = CatBoostClassifier(
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                l2_leaf_reg=self.l2_leaf_reg,
                random_state=self.random_state,
                verbose=100,
                **self.extra_params
            )

        # Prepare eval set
        eval_sets = None
        if eval_set is not None:
            X_val = self._prepare_features(eval_set[0])
            y_val = self._prepare_target(eval_set[1], self.task_type)
            eval_sets = (X_val, y_val)

        # Train
        self.model.fit(
            X_array,
            y_array,
            eval_set=eval_sets,
            early_stopping_rounds=self.early_stopping_rounds if eval_sets else None
        )

        self.is_fitted = True

        # Get feature importance
        self.feature_importance_ = dict(zip(
            self.feature_columns,
            self.model.feature_importances_
        ))

        # Calculate training metrics
        self.training_metrics_ = self.evaluate(X, y)

        self.logger.info(f"Model trained. Training IC: {self.training_metrics_.ic:.4f}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_array = self._prepare_features(X)
        return self.model.predict(X_array)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_array = self._prepare_features(X)

        if self.task_type == 'classification':
            return self.model.predict_proba(X_array)[:, 1]
        else:
            return self.predict(X)

    def _get_model_state(self) -> Any:
        """Get model-specific state."""
        return {'model': self.model}

    def _set_model_state(self, state: Any) -> None:
        """Set model-specific state."""
        if state:
            self.model = state.get('model')


class RandomForestModel(BaseModel):
    """
    Random Forest model for alpha prediction.

    Simple but effective ensemble model with built-in feature importance.
    """

    def __init__(
        self,
        name: str = 'random_forest',
        task_type: str = 'regression',
        target_column: str = 'target_return_10',
        feature_columns: Optional[List[str]] = None,
        random_state: int = 42,
        n_estimators: int = 200,
        max_depth: int = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = 'sqrt',
        **kwargs
    ):
        """
        Initialize Random Forest Model.

        Args:
            name: Model name
            task_type: 'regression' or 'classification'
            target_column: Target column name
            feature_columns: List of feature columns
            random_state: Random seed
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples per leaf
            max_features: Max features per split
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            target_column=target_column,
            feature_columns=feature_columns,
            random_state=random_state
        )

        self.task_type = task_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.extra_params = kwargs

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'RandomForestModel':
        """
        Fit the Random Forest model.

        Args:
            X: Features DataFrame
            y: Target Series
            eval_set: Not used for Random Forest
            **kwargs: Additional parameters

        Returns:
            self
        """
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        self.logger.info(f"Fitting Random Forest model with {len(X)} samples")

        # Prepare features and target
        X_array = self._prepare_features(X, fit=True)
        y_array = self._prepare_target(y, self.task_type)

        # Create model
        if self.task_type == 'regression':
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                n_jobs=-1,
                **self.extra_params
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                n_jobs=-1,
                **self.extra_params
            )

        # Train
        self.model.fit(X_array, y_array)
        self.is_fitted = True

        # Get feature importance
        self.feature_importance_ = dict(zip(
            self.feature_columns,
            self.model.feature_importances_
        ))

        # Calculate training metrics
        self.training_metrics_ = self.evaluate(X, y)

        self.logger.info(f"Model trained. Training IC: {self.training_metrics_.ic:.4f}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_array = self._prepare_features(X)
        return self.model.predict(X_array)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_array = self._prepare_features(X)

        if self.task_type == 'classification':
            return self.model.predict_proba(X_array)[:, 1]
        else:
            return self.predict(X)

    def _get_model_state(self) -> Any:
        """Get model-specific state."""
        return {'model': self.model}

    def _set_model_state(self, state: Any) -> None:
        """Set model-specific state."""
        if state:
            self.model = state.get('model')
