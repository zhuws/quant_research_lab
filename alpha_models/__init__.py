"""
Alpha Models for Quant Research Lab.
Machine learning models for alpha prediction.
"""

from alpha_models.base_model import BaseModel, ModelPrediction, ModelMetrics
from alpha_models.tree_models import (
    LightGBMModel,
    XGBoostModel,
    CatBoostModel,
    RandomForestModel
)
from alpha_models.neural_models import MLPModel, LSTMModel, TransformerModel
from alpha_models.model_trainer import (
    ModelTrainer,
    TrainingResult,
    HyperparameterOptimizer,
    EnsembleTrainer,
    train_model
)
from alpha_models.feature_selector import FeatureSelector, select_features

__all__ = [
    # Base
    'BaseModel',
    'ModelPrediction',
    'ModelMetrics',

    # Tree Models
    'LightGBMModel',
    'XGBoostModel',
    'CatBoostModel',
    'RandomForestModel',

    # Neural Models
    'MLPModel',
    'LSTMModel',
    'TransformerModel',

    # Training
    'ModelTrainer',
    'TrainingResult',
    'HyperparameterOptimizer',
    'EnsembleTrainer',
    'train_model',

    # Feature Selection
    'FeatureSelector',
    'select_features'
]
