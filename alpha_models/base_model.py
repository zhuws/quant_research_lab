"""
Base Model for Quant Research Lab.
Abstract base class for all prediction models.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import joblib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger


@dataclass
class ModelPrediction:
    """
    Container for model predictions.

    Attributes:
        predictions: Model predictions
        probabilities: Prediction probabilities (for classification)
        confidence: Prediction confidence scores
        feature_importance: Feature importance dict
    """
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None


@dataclass
class ModelMetrics:
    """
    Container for model evaluation metrics.

    Attributes:
        ic: Information Coefficient
        ic_ir: IC Information Ratio
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        accuracy: Classification accuracy
        sharpe: Sharpe ratio of predictions
        hit_rate: Hit rate (direction accuracy)
    """
    ic: float = 0.0
    ic_ir: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    accuracy: float = 0.0
    sharpe: float = 0.0
    hit_rate: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'ic': self.ic,
            'ic_ir': self.ic_ir,
            'rmse': self.rmse,
            'mae': self.mae,
            'accuracy': self.accuracy,
            'sharpe': self.sharpe,
            'hit_rate': self.hit_rate
        }


class BaseModel(ABC):
    """
    Abstract base class for prediction models.

    All prediction models must inherit from this class
    and implement the required methods.
    """

    def __init__(
        self,
        name: str = 'base_model',
        target_column: str = 'target_return_10',
        feature_columns: Optional[List[str]] = None,
        random_state: int = 42
    ):
        """
        Initialize Base Model.

        Args:
            name: Model name
            target_column: Target column name
            feature_columns: List of feature columns (None = auto-detect)
            random_state: Random seed for reproducibility
        """
        self.logger = get_logger(name)
        self.name = name
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.random_state = random_state

        # Model state
        self.model = None
        self.is_fitted = False
        self.feature_importance_: Optional[Dict[str, float]] = None
        self.training_metrics_: Optional[ModelMetrics] = None

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'BaseModel':
        """
        Fit the model.

        Args:
            X: Features DataFrame
            y: Target Series
            eval_set: Optional evaluation set
            **kwargs: Additional model-specific arguments

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features DataFrame

        Returns:
            Predictions array
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities (for classification).

        Args:
            X: Features DataFrame

        Returns:
            Probability array
        """
        pass

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        return self.feature_importance_ or {}

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> ModelMetrics:
        """
        Evaluate model performance.

        Args:
            X: Features DataFrame
            y: True target values

        Returns:
            ModelMetrics with evaluation results
        """
        predictions = self.predict(X)

        # Calculate metrics
        metrics = ModelMetrics()

        # IC (Information Coefficient)
        if len(predictions) == len(y):
            valid_mask = ~(np.isnan(predictions) | np.isnan(y))
            if valid_mask.sum() > 10:
                from scipy import stats
                ic, _ = stats.spearmanr(predictions[valid_mask], y[valid_mask])
                metrics.ic = ic if not np.isnan(ic) else 0.0

                # IC IR (using rolling IC)
                window = min(252, len(predictions) // 4)
                if window > 20:
                    pred_series = pd.Series(predictions)
                    y_series = pd.Series(y.values)
                    rolling_ic = pred_series.rolling(window).corr(y_series)
                    ic_std = rolling_ic.std()
                    metrics.ic_ir = metrics.ic / ic_std if ic_std > 0 else 0.0

        # RMSE and MAE
        mse = np.mean((predictions - y.values) ** 2)
        metrics.rmse = np.sqrt(mse)
        metrics.mae = np.mean(np.abs(predictions - y.values))

        # Hit rate (direction accuracy)
        correct_direction = np.sign(predictions) == np.sign(y.values)
        metrics.hit_rate = np.mean(correct_direction)

        # Accuracy (for classification-like evaluation)
        metrics.accuracy = metrics.hit_rate

        # Sharpe ratio of predictions
        if len(predictions) > 1:
            pred_returns = predictions * y.values  # Strategy returns
            if np.std(pred_returns) > 0:
                metrics.sharpe = np.mean(pred_returns) / np.std(pred_returns) * np.sqrt(252)

        return metrics

    def save(self, filepath: str) -> None:
        """
        Save model to file.

        Args:
            filepath: Output file path
        """
        model_data = {
            'name': self.name,
            'target_column': self.target_column,
            'feature_columns': self.feature_columns,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted,
            'feature_importance': self.feature_importance_,
            'training_metrics': self.training_metrics_.to_dict() if self.training_metrics_ else None
        }

        # Save model-specific data
        model_data['model'] = self._get_model_state()

        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> 'BaseModel':
        """
        Load model from file.

        Args:
            filepath: Input file path

        Returns:
            self
        """
        model_data = joblib.load(filepath)

        self.name = model_data.get('name', self.name)
        self.target_column = model_data.get('target_column', self.target_column)
        self.feature_columns = model_data.get('feature_columns', self.feature_columns)
        self.random_state = model_data.get('random_state', self.random_state)
        self.is_fitted = model_data.get('is_fitted', False)
        self.feature_importance_ = model_data.get('feature_importance')

        if model_data.get('training_metrics'):
            self.training_metrics_ = ModelMetrics(**model_data['training_metrics'])

        self._set_model_state(model_data.get('model'))

        self.logger.info(f"Model loaded from {filepath}")
        return self

    @abstractmethod
    def _get_model_state(self) -> Any:
        """Get model-specific state for serialization."""
        pass

    @abstractmethod
    def _set_model_state(self, state: Any) -> None:
        """Set model-specific state from serialization."""
        pass

    def _prepare_features(
        self,
        X: pd.DataFrame,
        fit: bool = False
    ) -> np.ndarray:
        """
        Prepare features for model.

        Args:
            X: Features DataFrame
            fit: Whether this is during fitting

        Returns:
            Feature array
        """
        if self.feature_columns is None:
            # Auto-detect feature columns
            exclude_cols = {
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                self.target_column
            }
            self.feature_columns = [
                col for col in X.columns
                if col not in exclude_cols and not col.startswith('target')
            ]
            self.logger.info(f"Auto-detected {len(self.feature_columns)} features")

        X_features = X[self.feature_columns].copy()

        # Handle missing values
        X_features = X_features.fillna(0)

        # Replace inf values
        X_features = X_features.replace([np.inf, -np.inf], 0)

        return X_features.values

    def _prepare_target(
        self,
        y: pd.Series,
        task_type: str = 'regression'
    ) -> np.ndarray:
        """
        Prepare target for model.

        Args:
            y: Target Series
            task_type: 'regression' or 'classification'

        Returns:
            Target array
        """
        if task_type == 'classification':
            # Convert to binary labels
            return (y > 0).astype(int).values
        else:
            return y.fillna(0).values

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"
