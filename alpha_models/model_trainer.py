"""
Model Trainer for Quant Research Lab.
Training pipeline with cross-validation and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Type
from dataclasses import dataclass
from datetime import datetime
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from alpha_models.base_model import BaseModel, ModelMetrics


@dataclass
class TrainingResult:
    """
    Result of model training.

    Attributes:
        model: Trained model
        metrics: Training metrics
        cv_scores: Cross-validation scores
        best_params: Best hyperparameters
        feature_importance: Feature importance dict
        training_time: Training time in seconds
    """
    model: BaseModel
    metrics: ModelMetrics
    cv_scores: Optional[List[float]] = None
    best_params: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    training_time: float = 0.0


class ModelTrainer:
    """
    Model training pipeline.

    Features:
        - Walk-forward cross-validation
        - Time-series split validation
        - Hyperparameter optimization
        - Ensemble training
        - Model persistence
    """

    def __init__(
        self,
        model_class: Type[BaseModel],
        target_column: str = 'target_return_10',
        cv_method: str = 'walk_forward',
        n_splits: int = 5,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_state: int = 42
    ):
        """
        Initialize Model Trainer.

        Args:
            model_class: Model class to train
            target_column: Target column name
            cv_method: Cross-validation method ('walk_forward', 'time_series', 'purged')
            n_splits: Number of CV splits
            train_ratio: Ratio for training data
            val_ratio: Ratio for validation data
            random_state: Random seed
        """
        self.logger = get_logger('model_trainer')
        self.model_class = model_class
        self.target_column = target_column
        self.cv_method = cv_method
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_state = random_state

    def train(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None
    ) -> TrainingResult:
        """
        Train a model.

        Args:
            data: DataFrame with features and target
            feature_columns: List of feature columns
            model_params: Model parameters
            eval_set: Optional evaluation set

        Returns:
            TrainingResult with trained model and metrics
        """
        import time
        start_time = time.time()

        model_params = model_params or {}
        model_params['target_column'] = self.target_column
        model_params['random_state'] = self.random_state

        self.logger.info(f"Training {self.model_class.__name__}")

        # Prepare data
        X = data.drop(columns=[self.target_column], errors='ignore')
        y = data[self.target_column] if self.target_column in data.columns else data.iloc[:, -1]

        # Create model
        model = self.model_class(**model_params)

        if feature_columns:
            model.feature_columns = feature_columns

        # Train
        model.fit(X, y, eval_set=eval_set)

        # Calculate metrics
        metrics = model.evaluate(X, y)

        training_time = time.time() - start_time

        self.logger.info(f"Training completed in {training_time:.2f}s")
        self.logger.info(f"IC: {metrics.ic:.4f}, Sharpe: {metrics.sharpe:.4f}")

        return TrainingResult(
            model=model,
            metrics=metrics,
            feature_importance=model.get_feature_importance(),
            training_time=training_time
        )

    def train_with_cv(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        model_params: Optional[Dict[str, Any]] = None
    ) -> TrainingResult:
        """
        Train with cross-validation.

        Args:
            data: DataFrame with features and target
            feature_columns: List of feature columns
            model_params: Model parameters

        Returns:
            TrainingResult with CV scores
        """
        import time
        start_time = time.time()

        model_params = model_params or {}
        model_params['target_column'] = self.target_column
        model_params['random_state'] = self.random_state

        self.logger.info(f"Training {self.model_class.__name__} with {self.cv_method} CV")

        # Prepare data
        X = data.drop(columns=[self.target_column], errors='ignore')
        y = data[self.target_column] if self.target_column in data.columns else data.iloc[:, -1]

        # Get CV splits
        splits = self._get_cv_splits(len(data))

        cv_scores = []
        best_model = None
        best_ic = -np.inf

        for fold, (train_idx, val_idx) in enumerate(splits):
            self.logger.info(f"Training fold {fold + 1}/{len(splits)}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Create and train model
            model = self.model_class(**model_params)
            if feature_columns:
                model.feature_columns = feature_columns

            model.fit(X_train, y_train, eval_set=(X_val, y_val))

            # Evaluate
            metrics = model.evaluate(X_val, y_val)
            cv_scores.append(metrics.ic)

            self.logger.info(f"Fold {fold + 1} IC: {metrics.ic:.4f}")

            # Track best model
            if metrics.ic > best_ic:
                best_ic = metrics.ic
                best_model = model

        training_time = time.time() - start_time

        # Final training metrics on full data
        final_metrics = best_model.evaluate(X, y)

        self.logger.info(f"CV IC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        return TrainingResult(
            model=best_model,
            metrics=final_metrics,
            cv_scores=cv_scores,
            feature_importance=best_model.get_feature_importance(),
            training_time=training_time
        )

    def _get_cv_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get cross-validation split indices."""
        if self.cv_method == 'walk_forward':
            return self._walk_forward_splits(n_samples)
        elif self.cv_method == 'time_series':
            return self._time_series_splits(n_samples)
        elif self.cv_method == 'purged':
            return self._purged_splits(n_samples)
        else:
            return self._time_series_splits(n_samples)

    def _walk_forward_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Walk-forward cross-validation splits."""
        splits = []
        train_size = int(n_samples * self.train_ratio)
        val_size = int(n_samples * self.val_ratio)
        step = (n_samples - train_size - val_size) // (self.n_splits - 1) if self.n_splits > 1 else 1

        for i in range(self.n_splits):
            train_start = i * step
            train_end = train_start + train_size
            val_start = train_end
            val_end = val_start + val_size

            if val_end > n_samples:
                break

            train_idx = np.arange(train_start, train_end)
            val_idx = np.arange(val_start, val_end)

            splits.append((train_idx, val_idx))

        return splits

    def _time_series_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Time-series cross-validation splits."""
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        return list(tscv.split(np.arange(n_samples)))

    def _purged_splits(self, n_samples: int, embargo_pct: float = 0.01) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Purged cross-validation splits with embargo."""
        splits = []
        fold_size = n_samples // self.n_splits
        embargo = int(n_samples * embargo_pct)

        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size

            # Training data before validation with embargo
            train_idx = np.arange(0, val_start - embargo)

            # Training data after validation with embargo
            post_train_start = val_end + embargo
            post_train_idx = np.arange(post_train_start, n_samples)

            train_idx = np.concatenate([train_idx, post_train_idx])
            val_idx = np.arange(val_start, val_end)

            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))

        return splits


class HyperparameterOptimizer:
    """
    Hyperparameter optimization for models.

    Supports:
        - Grid search
        - Random search
        - Bayesian optimization
    """

    def __init__(
        self,
        model_class: Type[BaseModel],
        target_column: str = 'target_return_10',
        optimization_method: str = 'random',
        n_trials: int = 50,
        random_state: int = 42
    ):
        """
        Initialize Hyperparameter Optimizer.

        Args:
            model_class: Model class to optimize
            target_column: Target column name
            optimization_method: 'grid', 'random', or 'bayesian'
            n_trials: Number of trials for random/bayesian
            random_state: Random seed
        """
        self.logger = get_logger('hyperparameter_optimizer')
        self.model_class = model_class
        self.target_column = target_column
        self.optimization_method = optimization_method
        self.n_trials = n_trials
        self.random_state = random_state

    def optimize(
        self,
        data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        cv_splits: int = 3
    ) -> Tuple[Dict[str, Any], float]:
        """
        Optimize hyperparameters.

        Args:
            data: DataFrame with features and target
            param_grid: Parameter grid to search
            cv_splits: Number of CV splits

        Returns:
            Tuple of (best_params, best_score)
        """
        if self.optimization_method == 'grid':
            return self._grid_search(data, param_grid, cv_splits)
        elif self.optimization_method == 'random':
            return self._random_search(data, param_grid, cv_splits)
        elif self.optimization_method == 'bayesian':
            return self._bayesian_optimization(data, param_grid, cv_splits)
        else:
            return self._random_search(data, param_grid, cv_splits)

    def _grid_search(
        self,
        data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        cv_splits: int
    ) -> Tuple[Dict[str, Any], float]:
        """Grid search optimization."""
        from itertools import product

        self.logger.info("Running grid search optimization")

        X = data.drop(columns=[self.target_column], errors='ignore')
        y = data[self.target_column] if self.target_column in data.columns else data.iloc[:, -1]

        best_score = -np.inf
        best_params = {}

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        total = len(combinations)
        self.logger.info(f"Testing {total} parameter combinations")

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            try:
                # Cross-validation score
                cv_score = self._evaluate_params(X, y, params, cv_splits)

                self.logger.info(f"[{i+1}/{total}] {params} -> IC: {cv_score:.4f}")

                if cv_score > best_score:
                    best_score = cv_score
                    best_params = params.copy()

            except Exception as e:
                self.logger.warning(f"Error with params {params}: {e}")

        self.logger.info(f"Best params: {best_params} with IC: {best_score:.4f}")
        return best_params, best_score

    def _random_search(
        self,
        data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        cv_splits: int
    ) -> Tuple[Dict[str, Any], float]:
        """Random search optimization."""
        import random

        self.logger.info(f"Running random search with {self.n_trials} trials")

        X = data.drop(columns=[self.target_column], errors='ignore')
        y = data[self.target_column] if self.target_column in data.columns else data.iloc[:, -1]

        best_score = -np.inf
        best_params = {}

        for i in range(self.n_trials):
            # Sample random parameters
            params = {
                key: random.choice(values)
                for key, values in param_grid.items()
            }

            try:
                cv_score = self._evaluate_params(X, y, params, cv_splits)

                self.logger.info(f"[{i+1}/{self.n_trials}] {params} -> IC: {cv_score:.4f}")

                if cv_score > best_score:
                    best_score = cv_score
                    best_params = params.copy()

            except Exception as e:
                self.logger.warning(f"Error with params {params}: {e}")

        self.logger.info(f"Best params: {best_params} with IC: {best_score:.4f}")
        return best_params, best_score

    def _bayesian_optimization(
        self,
        data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        cv_splits: int
    ) -> Tuple[Dict[str, Any], float]:
        """Bayesian optimization using Optuna."""
        try:
            import optuna
        except ImportError:
            self.logger.warning("Optuna not installed, falling back to random search")
            return self._random_search(data, param_grid, cv_splits)

        self.logger.info("Running Bayesian optimization")

        X = data.drop(columns=[self.target_column], errors='ignore')
        y = data[self.target_column] if self.target_column in data.columns else data.iloc[:, -1]

        def objective(trial):
            params = {}
            for key, values in param_grid.items():
                if isinstance(values[0], int):
                    params[key] = trial.suggest_int(key, min(values), max(values))
                elif isinstance(values[0], float):
                    params[key] = trial.suggest_float(key, min(values), max(values))
                else:
                    params[key] = trial.suggest_categorical(key, values)

            return self._evaluate_params(X, y, params, cv_splits)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        self.logger.info(f"Best params: {best_params} with IC: {best_score:.4f}")
        return best_params, best_score

    def _evaluate_params(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict[str, Any],
        cv_splits: int
    ) -> float:
        """Evaluate parameters using cross-validation."""
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=cv_splits)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = self.model_class(**params)
            model.fit(X_train, y_train)

            metrics = model.evaluate(X_val, y_val)
            scores.append(metrics.ic)

        return np.mean(scores)


class EnsembleTrainer:
    """
    Train ensemble of multiple models.
    """

    def __init__(
        self,
        model_configs: List[Dict[str, Any]],
        target_column: str = 'target_return_10',
        ensemble_method: str = 'average',
        weights: Optional[List[float]] = None
    ):
        """
        Initialize Ensemble Trainer.

        Args:
            model_configs: List of model configurations
            target_column: Target column name
            ensemble_method: 'average', 'weighted', or 'stacking'
            weights: Model weights for weighted ensemble
        """
        self.logger = get_logger('ensemble_trainer')
        self.model_configs = model_configs
        self.target_column = target_column
        self.ensemble_method = ensemble_method
        self.weights = weights

        self.models_: List[BaseModel] = []
        self.meta_model_: Optional[BaseModel] = None

    def train(
        self,
        data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Train ensemble models.

        Args:
            data: Training data
            val_data: Validation data for stacking

        Returns:
            Dictionary with training results
        """
        self.logger.info(f"Training ensemble with {len(self.model_configs)} models")

        X = data.drop(columns=[self.target_column], errors='ignore')
        y = data[self.target_column] if self.target_column in data.columns else data.iloc[:, -1]

        results = {'models': [], 'metrics': []}

        for i, config in enumerate(self.model_configs):
            model_class = config['class']
            params = config.get('params', {})

            self.logger.info(f"Training model {i+1}: {model_class.__name__}")

            model = model_class(**params)
            model.fit(X, y)

            metrics = model.evaluate(X, y)

            self.models_.append(model)
            results['models'].append(model)
            results['metrics'].append(metrics)

            self.logger.info(f"Model {i+1} IC: {metrics.ic:.4f}")

        # Train stacking meta-model if requested
        if self.ensemble_method == 'stacking' and val_data is not None:
            self._train_stacking_model(val_data)

        return results

    def _train_stacking_model(self, val_data: pd.DataFrame) -> None:
        """Train stacking meta-model."""
        from sklearn.linear_model import Ridge

        X_val = val_data.drop(columns=[self.target_column], errors='ignore')
        y_val = val_data[self.target_column] if self.target_column in val_data.columns else val_data.iloc[:, -1]

        # Get predictions from all models
        predictions = np.column_stack([
            model.predict(X_val) for model in self.models_
        ])

        # Train meta-model
        self.meta_model_ = Ridge()
        self.meta_model_.fit(predictions, y_val)

        self.logger.info("Trained stacking meta-model")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            X: Features DataFrame

        Returns:
            Ensemble predictions
        """
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])

        if self.ensemble_method == 'average':
            return np.mean(predictions, axis=1)

        elif self.ensemble_method == 'weighted':
            weights = self.weights or [1.0 / len(self.models_)] * len(self.models_)
            return np.average(predictions, axis=1, weights=weights)

        elif self.ensemble_method == 'stacking' and self.meta_model_ is not None:
            return self.meta_model_.predict(predictions)

        else:
            return np.mean(predictions, axis=1)


def train_model(
    data: pd.DataFrame,
    model_type: str = 'lightgbm',
    target_column: str = 'target_return_10',
    model_params: Optional[Dict[str, Any]] = None
) -> TrainingResult:
    """
    Convenience function to train a model.

    Args:
        data: DataFrame with features and target
        model_type: Model type ('lightgbm', 'xgboost', 'catboost', 'random_forest')
        target_column: Target column name
        model_params: Model parameters

    Returns:
        TrainingResult
    """
    from alpha_models.tree_models import LightGBMModel, XGBoostModel, CatBoostModel, RandomForestModel

    model_classes = {
        'lightgbm': LightGBMModel,
        'xgboost': XGBoostModel,
        'catboost': CatBoostModel,
        'random_forest': RandomForestModel
    }

    model_class = model_classes.get(model_type, LightGBMModel)

    trainer = ModelTrainer(
        model_class=model_class,
        target_column=target_column
    )

    return trainer.train(data, model_params=model_params)
