"""
Neural Network Models for Quant Research Lab.
MLP, LSTM, and Transformer-based models for alpha prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from alpha_models.base_model import BaseModel, ModelMetrics


class MLPModel(BaseModel):
    """
    Multi-Layer Perceptron model for alpha prediction.

    Simple feedforward neural network with customizable architecture.
    """

    def __init__(
        self,
        name: str = 'mlp',
        task_type: str = 'regression',
        target_column: str = 'target_return_10',
        feature_columns: Optional[List[str]] = None,
        random_state: int = 42,
        hidden_layers: List[int] = [256, 128, 64],
        dropout_rate: float = 0.2,
        activation: str = 'relu',
        learning_rate: float = 0.001,
        batch_size: int = 1024,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        l2_reg: float = 0.001,
        **kwargs
    ):
        """
        Initialize MLP Model.

        Args:
            name: Model name
            task_type: 'regression' or 'classification'
            target_column: Target column name
            feature_columns: List of feature columns
            random_state: Random seed
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate
            activation: Activation function
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Number of epochs
            early_stopping_patience: Early stopping patience
            l2_reg: L2 regularization
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            target_column=target_column,
            feature_columns=feature_columns,
            random_state=random_state
        )

        self.task_type = task_type
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.l2_reg = l2_reg
        self.extra_params = kwargs

        # Normalization parameters
        self._mean = None
        self._std = None

    def _build_model(self, input_dim: int) -> Any:
        """Build the neural network model."""
        import tensorflow as tf
        from tensorflow.keras import layers, models, regularizers

        tf.random.set_seed(self.random_state)

        model = models.Sequential()

        # Input layer
        model.add(layers.Input(shape=(input_dim,)))

        # Hidden layers
        for units in self.hidden_layers:
            model.add(layers.Dense(
                units,
                activation=self.activation,
                kernel_regularizer=regularizers.l2(self.l2_reg)
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.dropout_rate))

        # Output layer
        if self.task_type == 'regression':
            model.add(layers.Dense(1, activation='linear'))
            loss = 'mse'
        else:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'

        # Compile
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])

        return model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'MLPModel':
        """
        Fit the MLP model.

        Args:
            X: Features DataFrame
            y: Target Series
            eval_set: Optional evaluation set
            **kwargs: Additional parameters

        Returns:
            self
        """
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        self.logger.info(f"Fitting MLP model with {len(X)} samples")

        # Prepare features and target
        X_array = self._prepare_features(X, fit=True)
        y_array = self._prepare_target(y, self.task_type)

        # Normalize features
        self._mean = np.mean(X_array, axis=0)
        self._std = np.std(X_array, axis=0) + 1e-8
        X_array = (X_array - self._mean) / self._std

        # Build model
        self.model = self._build_model(X_array.shape[1])

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Prepare validation data
        validation_data = None
        if eval_set is not None:
            X_val = self._prepare_features(eval_set[0])
            y_val = self._prepare_target(eval_set[1], self.task_type)
            X_val = (X_val - self._mean) / self._std
            validation_data = (X_val, y_val)

        # Train
        history = self.model.fit(
            X_array,
            y_array,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.is_fitted = True

        # Calculate feature importance using permutation
        self._calculate_feature_importance(X_array, y_array)

        # Calculate training metrics
        self.training_metrics_ = self.evaluate(X, y)

        self.logger.info(f"Model trained. Training IC: {self.training_metrics_.ic:.4f}")

        return self

    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray) -> None:
        """Calculate feature importance using permutation method."""
        from sklearn.inspection import permutation_importance

        def predict_fn(X_2d):
            return self.model.predict(X_2d, verbose=0).flatten()

        try:
            result = permutation_importance(
                predict_fn,
                X,
                y,
                n_repeats=5,
                random_state=self.random_state,
                n_jobs=-1
            )

            self.feature_importance_ = dict(zip(
                self.feature_columns,
                result.importances_mean
            ))
        except Exception as e:
            self.logger.warning(f"Could not calculate feature importance: {e}")
            self.feature_importance_ = {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_array = self._prepare_features(X)
        X_array = (X_array - self._mean) / self._std

        predictions = self.model.predict(X_array, verbose=0)
        return predictions.flatten()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        return self.predict(X)

    def _get_model_state(self) -> Any:
        """Get model-specific state."""
        import tempfile
        import os

        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as f:
            temp_path = f.name

        self.model.save(temp_path)

        with open(temp_path, 'rb') as f:
            model_bytes = f.read()

        os.unlink(temp_path)

        return {
            'model_bytes': model_bytes,
            'mean': self._mean,
            'std': self._std
        }

    def _set_model_state(self, state: Any) -> None:
        """Set model-specific state."""
        import tempfile
        import os

        if state:
            # Load model from bytes
            model_bytes = state.get('model_bytes')

            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as f:
                temp_path = f.name
                f.write(model_bytes)

            from tensorflow.keras.models import load_model
            self.model = load_model(temp_path)

            os.unlink(temp_path)

            self._mean = state.get('mean')
            self._std = state.get('std')


class LSTMModel(BaseModel):
    """
    LSTM model for alpha prediction with sequential data.

    Captures temporal patterns in market data.
    """

    def __init__(
        self,
        name: str = 'lstm',
        task_type: str = 'regression',
        target_column: str = 'target_return_10',
        feature_columns: Optional[List[str]] = None,
        random_state: int = 42,
        sequence_length: int = 20,
        lstm_units: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        **kwargs
    ):
        """
        Initialize LSTM Model.

        Args:
            name: Model name
            task_type: 'regression' or 'classification'
            target_column: Target column name
            feature_columns: List of feature columns
            random_state: Random seed
            sequence_length: Number of time steps
            lstm_units: List of LSTM layer sizes
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Number of epochs
            early_stopping_patience: Early stopping patience
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            target_column=target_column,
            feature_columns=feature_columns,
            random_state=random_state
        )

        self.task_type = task_type
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.extra_params = kwargs

        self._mean = None
        self._std = None

    def _build_model(self, input_shape: Tuple[int, int]) -> Any:
        """Build LSTM model."""
        import tensorflow as tf
        from tensorflow.keras import layers, models

        tf.random.set_seed(self.random_state)

        model = models.Sequential()

        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(layers.LSTM(
                units,
                return_sequences=return_sequences,
                input_shape=input_shape if i == 0 else None
            ))
            model.add(layers.Dropout(self.dropout_rate))

        # Output layer
        if self.task_type == 'regression':
            model.add(layers.Dense(1, activation='linear'))
            loss = 'mse'
        else:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])

        return model

    def _create_sequences(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sequences from feature matrix."""
        n_samples = X.shape[0] - self.sequence_length + 1

        X_seq = np.zeros((n_samples, self.sequence_length, X.shape[1]))

        for i in range(n_samples):
            X_seq[i] = X[i:i + self.sequence_length]

        if y is not None:
            y_seq = y[self.sequence_length - 1:]
            return X_seq, y_seq

        return X_seq, None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'LSTMModel':
        """Fit the LSTM model."""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        self.logger.info(f"Fitting LSTM model with {len(X)} samples")

        # Prepare features and target
        X_array = self._prepare_features(X, fit=True)
        y_array = self._prepare_target(y, self.task_type)

        # Normalize
        self._mean = np.mean(X_array, axis=0)
        self._std = np.std(X_array, axis=0) + 1e-8
        X_array = (X_array - self._mean) / self._std

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_array, y_array)

        # Build model
        self.model = self._build_model((self.sequence_length, X_array.shape[1]))

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Prepare validation data
        validation_data = None
        if eval_set is not None:
            X_val = self._prepare_features(eval_set[0])
            y_val = self._prepare_target(eval_set[1], self.task_type)
            X_val = (X_val - self._mean) / self._std
            X_val_seq, y_val_seq = self._create_sequences(X_val, y_val)
            validation_data = (X_val_seq, y_val_seq)

        # Train
        self.model.fit(
            X_seq,
            y_seq,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.is_fitted = True
        self.feature_importance_ = {}  # LSTM doesn't have direct feature importance

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_array = self._prepare_features(X)
        X_array = (X_array - self._mean) / self._std

        # Pad with zeros if not enough samples
        if len(X_array) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(X_array), X_array.shape[1]))
            X_array = np.vstack([padding, X_array])

        X_seq, _ = self._create_sequences(X_array)

        predictions = self.model.predict(X_seq, verbose=0)
        return predictions.flatten()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        return self.predict(X)

    def _get_model_state(self) -> Any:
        """Get model-specific state."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as f:
            temp_path = f.name

        self.model.save(temp_path)

        with open(temp_path, 'rb') as f:
            model_bytes = f.read()

        os.unlink(temp_path)

        return {
            'model_bytes': model_bytes,
            'mean': self._mean,
            'std': self._std
        }

    def _set_model_state(self, state: Any) -> None:
        """Set model-specific state."""
        import tempfile
        import os

        if state:
            model_bytes = state.get('model_bytes')

            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as f:
                temp_path = f.name
                f.write(model_bytes)

            from tensorflow.keras.models import load_model
            self.model = load_model(temp_path)

            os.unlink(temp_path)

            self._mean = state.get('mean')
            self._std = state.get('std')


class TransformerModel(BaseModel):
    """
    Transformer model for alpha prediction.

    Uses self-attention to capture complex patterns in market data.
    """

    def __init__(
        self,
        name: str = 'transformer',
        task_type: str = 'regression',
        target_column: str = 'target_return_10',
        feature_columns: Optional[List[str]] = None,
        random_state: int = 42,
        sequence_length: int = 20,
        d_model: int = 64,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        d_ff: int = 128,
        dropout_rate: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        **kwargs
    ):
        """
        Initialize Transformer Model.

        Args:
            name: Model name
            task_type: 'regression' or 'classification'
            target_column: Target column name
            feature_columns: List of feature columns
            random_state: Random seed
            sequence_length: Number of time steps
            d_model: Model dimension
            n_heads: Number of attention heads
            n_encoder_layers: Number of encoder layers
            d_ff: Feedforward dimension
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Number of epochs
            early_stopping_patience: Early stopping patience
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            target_column=target_column,
            feature_columns=feature_columns,
            random_state=random_state
        )

        self.task_type = task_type
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.extra_params = kwargs

        self._mean = None
        self._std = None

    def _build_model(self, input_shape: Tuple[int, int]) -> Any:
        """Build Transformer model."""
        import tensorflow as tf
        from tensorflow.keras import layers, models

        tf.random.set_seed(self.random_state)

        # Input
        inputs = layers.Input(shape=input_shape)

        # Feature projection
        x = layers.Dense(self.d_model)(inputs)

        # Positional encoding (simplified)
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        position_embedding = layers.Embedding(
            input_dim=self.sequence_length,
            output_dim=self.d_model
        )(positions)
        x = x + position_embedding

        # Transformer encoder layers
        for _ in range(self.n_encoder_layers):
            # Multi-head attention
            attention = layers.MultiHeadAttention(
                num_heads=self.n_heads,
                key_dim=self.d_model // self.n_heads
            )(x, x)
            attention = layers.Dropout(self.dropout_rate)(attention)
            x = layers.LayerNormalization(epsilon=1e-6)(x + attention)

            # Feedforward
            ff = layers.Dense(self.d_ff, activation='relu')(x)
            ff = layers.Dense(self.d_model)(ff)
            ff = layers.Dropout(self.dropout_rate)(ff)
            x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)

        # Output
        if self.task_type == 'regression':
            outputs = layers.Dense(1, activation='linear')(x)
            loss = 'mse'
        else:
            outputs = layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'

        model = models.Model(inputs=inputs, outputs=outputs)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])

        return model

    def _create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Create sequences from feature matrix."""
        n_samples = X.shape[0] - self.sequence_length + 1
        X_seq = np.zeros((n_samples, self.sequence_length, X.shape[1]))

        for i in range(n_samples):
            X_seq[i] = X[i:i + self.sequence_length]

        if y is not None:
            y_seq = y[self.sequence_length - 1:]
            return X_seq, y_seq

        return X_seq, None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs
    ) -> 'TransformerModel':
        """Fit the Transformer model."""
        from tensorflow.keras.callbacks import EarlyStopping

        self.logger.info(f"Fitting Transformer model with {len(X)} samples")

        X_array = self._prepare_features(X, fit=True)
        y_array = self._prepare_target(y, self.task_type)

        self._mean = np.mean(X_array, axis=0)
        self._std = np.std(X_array, axis=0) + 1e-8
        X_array = (X_array - self._mean) / self._std

        X_seq, y_seq = self._create_sequences(X_array, y_array)

        self.model = self._build_model((self.sequence_length, X_array.shape[1]))

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
        ]

        validation_data = None
        if eval_set is not None:
            X_val = self._prepare_features(eval_set[0])
            y_val = self._prepare_target(eval_set[1], self.task_type)
            X_val = (X_val - self._mean) / self._std
            X_val_seq, y_val_seq = self._create_sequences(X_val, y_val)
            validation_data = (X_val_seq, y_val_seq)

        self.model.fit(
            X_seq,
            y_seq,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_array = self._prepare_features(X)
        X_array = (X_array - self._mean) / self._std

        if len(X_array) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(X_array), X_array.shape[1]))
            X_array = np.vstack([padding, X_array])

        X_seq, _ = self._create_sequences(X_array)

        predictions = self.model.predict(X_seq, verbose=0)
        return predictions.flatten()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        return self.predict(X)

    def _get_model_state(self) -> Any:
        """Get model-specific state."""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as f:
            temp_path = f.name

        self.model.save(temp_path)

        with open(temp_path, 'rb') as f:
            model_bytes = f.read()

        os.unlink(temp_path)

        return {
            'model_bytes': model_bytes,
            'mean': self._mean,
            'std': self._std
        }

    def _set_model_state(self, state: Any) -> None:
        """Set model-specific state."""
        import tempfile
        import os

        if state:
            model_bytes = state.get('model_bytes')

            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as f:
                temp_path = f.name
                f.write(model_bytes)

            from tensorflow.keras.models import load_model
            self.model = load_model(temp_path)

            os.unlink(temp_path)

            self._mean = state.get('mean')
            self._std = state.get('std')
