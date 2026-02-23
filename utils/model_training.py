"""
Model Training Utilities for Human Activity Recognition
Provides machine learning models optimized for edge deployment
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import joblib
import os
import json
import logging

from utils.pytorch_models import (
    is_pytorch_available, PyTorchTrainer, HARMLP, HARCNN
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EdgeMLModel:
    """Base class for edge-optimized machine learning models."""

    def __init__(self, model_type: str, **kwargs):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.performance_metrics = {}
        self.model_params = kwargs

    def _initialize_model(self):
        """Initialize the ML model based on type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.model_params.get('n_estimators', 50),
                max_depth=self.model_params.get('max_depth', 10),
                min_samples_split=self.model_params.get(
                    'min_samples_split', 5),
                min_samples_leaf=self.model_params.get('min_samples_leaf', 2),
                class_weight=self.model_params.get(
                    'class_weight', 'balanced'),
                random_state=42
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                C=self.model_params.get('C', 1.0),
                kernel=self.model_params.get('kernel', 'rbf'),
                gamma=self.model_params.get('gamma', 'scale'),
                class_weight=self.model_params.get(
                    'class_weight', 'balanced'),
                random_state=42,
                probability=True
            )
        elif self.model_type == 'neural_network':
            hidden_layers = self.model_params.get(
                'hidden_layer_sizes', (100, 50))
            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation=self.model_params.get('activation', 'relu'),
                solver=self.model_params.get('solver', 'adam'),
                alpha=self.model_params.get('alpha', 0.0001),
                learning_rate_init=self.model_params.get(
                    'learning_rate', 0.001),
                max_iter=self.model_params.get('max_iter', 500),
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        elif self.model_type == 'pytorch_mlp':
            if not is_pytorch_available():
                raise RuntimeError("PyTorch is required for pytorch_mlp. "
                                   "Install via: pip install torch")
            # Actual model creation is deferred to train() because we need
            # input_size and num_classes which are only known after
            # preprocessing.  Store config for now.
            self.model = None  # placeholder
            self._pytorch_config = {
                'hidden_sizes': self.model_params.get(
                    'hidden_layer_sizes', (128, 64)),
                'dropout': self.model_params.get('dropout', 0.3),
                'epochs': self.model_params.get('max_iter', 200),
                'batch_size': self.model_params.get('batch_size', 32),
                'lr': self.model_params.get('learning_rate', 1e-3),
                'patience': self.model_params.get('patience', 15),
            }
        elif self.model_type == 'pytorch_cnn':
            if not is_pytorch_available():
                raise RuntimeError("PyTorch is required for pytorch_cnn. "
                                   "Install via: pip install torch")
            self.model = None  # placeholder
            self._pytorch_config = {
                'dropout': self.model_params.get('dropout', 0.3),
                'epochs': self.model_params.get('max_iter', 200),
                'batch_size': self.model_params.get('batch_size', 32),
                'lr': self.model_params.get('learning_rate', 1e-3),
                'patience': self.model_params.get('patience', 15),
                'window_size': self.model_params.get('window_size', 150),
                'n_channels': self.model_params.get('n_channels', 6),
            }
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def preprocess_data(self, X: pd.DataFrame, y: pd.Series = None,
                        scaler_type: str = 'standard') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess features and labels for training."""
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = np.asarray(X) if not isinstance(X, np.ndarray) else X

        # CNN uses raw 3D windows — skip scaling entirely
        if self.model_type == 'pytorch_cnn':
            X_scaled = X_array
        elif self.scaler is None:
            if scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unsupported scaler type: {scaler_type}")

            X_scaled = self.scaler.fit_transform(X_array)
        else:
            X_scaled = self.scaler.transform(X_array)

        # Handle labels if provided
        y_encoded = None
        if y is not None:
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)

        return X_scaled, y_encoded

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              scaler_type: str = 'standard', use_cross_validation: bool = True,
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train the model with the provided data.

        Args:
            X_train: Training features
            y_train: Training labels
            scaler_type: Type of feature scaling ('standard' or 'minmax')
            use_cross_validation: Whether to use CV for evaluation
            X_val: Optional validation features for early stopping
            y_val: Optional validation labels for early stopping

        Returns:
            Dictionary containing performance metrics and training history
        """
        logger.info(f"Training {self.model_type} model...")

        # Initialize model
        self._initialize_model()

        # Preprocess data
        X_scaled, y_encoded = self.preprocess_data(
            X_train, y_train, scaler_type)

        # Preprocess validation data if provided
        X_val_scaled, y_val_encoded = None, None
        if X_val is not None and y_val is not None:
            X_val_scaled, y_val_encoded = self.preprocess_data(X_val, y_val)
            logger.info(f"Using validation set: {len(X_val)} samples")

        # Train model with validation-based early stopping if available
        if self.model_type == 'neural_network' and X_val_scaled is not None:
            # Neural networks support partial_fit for monitoring per-epoch progress
            from sklearn.neural_network import MLPClassifier

            # Track validation accuracy per epoch
            val_accuracies = []
            train_accuracies = []
            best_val_accuracy = 0
            best_model_params = None
            patience_counter = 0
            patience = 10  # Stop if no improvement for 10 epochs

            logger.info("Training with validation-based early stopping...")

            # Train epoch by epoch
            for epoch in range(self.model.max_iter):
                # Fit one epoch (using warm_start to continue from previous state)
                self.model.max_iter = epoch + 1
                self.model.warm_start = True
                self.model.fit(X_scaled, y_encoded)

                # Evaluate on validation set
                val_pred = self.model.predict(X_val_scaled)
                val_acc = accuracy_score(y_val_encoded, val_pred)
                val_accuracies.append(val_acc)

                # Track training accuracy
                train_pred = self.model.predict(X_scaled)
                train_acc = accuracy_score(y_encoded, train_pred)
                train_accuracies.append(train_acc)

                # Early stopping check
                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                    best_model_params = self.model.get_params()
                    patience_counter = 0
                    logger.info(
                        f"Epoch {epoch+1}: Val Acc={val_acc:.4f} (improved) - Train Acc={train_acc:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(
                            f"Early stopping at epoch {epoch+1}: No improvement for {patience} epochs")
                        break

            # Store training history
            self.performance_metrics['val_accuracies'] = val_accuracies
            self.performance_metrics['train_accuracies'] = train_accuracies
            self.performance_metrics['best_val_accuracy'] = best_val_accuracy
            self.performance_metrics['stopped_epoch'] = epoch + 1
            self.performance_metrics['early_stopped'] = (
                patience_counter >= patience)

        elif self.model_type in ('pytorch_mlp', 'pytorch_cnn'):
            # ---- PyTorch training path ----
            cfg = self._pytorch_config
            num_classes = len(set(y_encoded))
            input_size = X_scaled.shape[1]

            if self.model_type == 'pytorch_mlp':
                import torch.nn as _nn
                net = HARMLP(
                    input_size=input_size,
                    hidden_sizes=cfg['hidden_sizes'],
                    num_classes=num_classes,
                    dropout=cfg['dropout'],
                )
            else:  # pytorch_cnn
                # CNN expects (batch, window_size, channels) not flat features
                # For CNN, X_scaled is already raw windows, not extracted features
                net = HARCNN(
                    window_size=cfg['window_size'],
                    n_channels=cfg['n_channels'],
                    num_classes=num_classes,
                    dropout=cfg['dropout'],
                )

            trainer = PyTorchTrainer(net)
            pt_metrics = trainer.train(
                X_scaled, y_encoded,
                X_val=X_val_scaled, y_val=y_val_encoded,
                epochs=cfg['epochs'],
                batch_size=cfg['batch_size'],
                lr=cfg['lr'],
                patience=cfg['patience'],
            )

            # Store the trainer so evaluate/predict/save can use it
            self._pytorch_trainer = trainer
            # Make self.model point to the PyTorch net so save_model can
            # detect that it's a PyTorch model
            self.model = net

            self.performance_metrics.update(pt_metrics)

        else:
            # Standard training for other models or when no validation set
            self.model.fit(X_scaled, y_encoded)

        # Cross-validation for model evaluation (if requested and no validation used)
        if use_cross_validation and X_val is None and self.model_type not in (
                'pytorch_mlp', 'pytorch_cnn'):
            cv_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=5)
            self.performance_metrics['cv_mean_accuracy'] = cv_scores.mean()
            self.performance_metrics['cv_std_accuracy'] = cv_scores.std()
            logger.info(
                f"Cross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # Training accuracy — predict() returns decoded strings, compare
        # against original string labels for consistency
        train_predictions = self.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        self.performance_metrics['train_accuracy'] = train_accuracy

        logger.info(
            f"Training completed. Training accuracy: {train_accuracy:.4f}")

        return self.performance_metrics

    def predict(self, X) -> np.ndarray:
        """Predict class labels. Returns decoded string labels.

        Works for sklearn, PyTorch MLP, and PyTorch CNN models.
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X) if not isinstance(X, np.ndarray) else X

        # Scale (skip for CNN which uses raw 3D data)
        if self.scaler is not None and self.model_type != 'pytorch_cnn':
            X_scaled = self.scaler.transform(X_array)
        else:
            X_scaled = X_array

        # Get encoded predictions
        if self.model_type in ('pytorch_mlp', 'pytorch_cnn'):
            encoded_preds = self._pytorch_trainer.predict(X_scaled)
        else:
            encoded_preds = self.model.predict(X_scaled)

        # Decode to original string labels
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(encoded_preds)
        return encoded_preds

    def predict_proba(self, X) -> Optional[np.ndarray]:
        """Return class probabilities.

        Works for sklearn, PyTorch MLP, and PyTorch CNN models.
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X) if not isinstance(X, np.ndarray) else X

        # Scale (skip for CNN which uses raw 3D data)
        if self.scaler is not None and self.model_type != 'pytorch_cnn':
            X_scaled = self.scaler.transform(X_array)
        else:
            X_scaled = X_array

        if self.model_type in ('pytorch_mlp', 'pytorch_cnn'):
            return self._pytorch_trainer.predict_proba(X_scaled)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        return None

    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        """Evaluate the trained model on test data."""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")

        # Preprocess test data (handles CNN 3D data via preprocess_data)
        X_scaled, y_encoded = self.preprocess_data(X_test, y_test)

        # Make predictions — get encoded integer labels
        if self.model_type in ('pytorch_mlp', 'pytorch_cnn'):
            y_pred = self._pytorch_trainer.predict(X_scaled)
            y_pred_proba = self._pytorch_trainer.predict_proba(X_scaled)
        else:
            y_pred = self.model.predict(X_scaled)
            y_pred_proba = self.model.predict_proba(X_scaled) if hasattr(
                self.model, 'predict_proba') else None

        # Calculate metrics with original label names
        accuracy = accuracy_score(y_encoded, y_pred)
        conf_matrix = confusion_matrix(y_encoded, y_pred)

        # Get target names for classification report
        if self.label_encoder is not None:
            target_names = self.label_encoder.classes_.tolist()
            class_report = classification_report(
                y_encoded, y_pred, target_names=target_names, output_dict=True)
        else:
            class_report = classification_report(
                y_encoded, y_pred, output_dict=True)

        # Store evaluation results
        evaluation_results = {
            'test_accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'predictions': y_pred.tolist(),
            'actual_labels': y_encoded.tolist(),
            'label_names': target_names if self.label_encoder is not None else None
        }

        if y_pred_proba is not None:
            evaluation_results['prediction_probabilities'] = y_pred_proba.tolist()

        # Update performance metrics
        self.performance_metrics.update(evaluation_results)

        logger.info(
            f"Model evaluation completed. Test accuracy: {accuracy:.4f}")

        return evaluation_results

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores if available."""
        if self.model is None:
            return None

        importance_dict = None
        if hasattr(self.model, 'feature_importances_'):
            # For tree-based models
            importances = self.model.feature_importances_
            if self.feature_names:
                importance_dict = dict(zip(self.feature_names, importances))
            else:
                importance_dict = {f"feature_{i}": imp for i,
                                   imp in enumerate(importances)}
        elif hasattr(self.model, 'coef_'):
            # For linear models
            if len(self.model.coef_.shape) > 1:
                # Multi-class case - take mean absolute coefficients
                importances = np.mean(np.abs(self.model.coef_), axis=0)
            else:
                importances = np.abs(self.model.coef_[0])

            if self.feature_names:
                importance_dict = dict(zip(self.feature_names, importances))
            else:
                importance_dict = {f"feature_{i}": imp for i,
                                   imp in enumerate(importances)}

        return importance_dict

    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 param_grid: Dict[str, List] = None,
                                 X_val: Optional[pd.DataFrame] = None,
                                 y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Optimize hyperparameters using validation set or grid search.

        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Dictionary of hyperparameters to search
            X_val: Optional validation features for evaluation
            y_val: Optional validation labels for evaluation

        Returns:
            Dictionary with best parameters and optimization results
        """
        if param_grid is None:
            param_grid = self._get_default_param_grid()

        # PyTorch models don't support sklearn-style hyperparameter search
        if self.model_type in ('pytorch_mlp', 'pytorch_cnn'):
            logger.warning("Hyperparameter optimization is not supported for "
                           f"{self.model_type}. Adjust parameters manually.")
            return {'best_params': {}, 'best_score': 0.0,
                    'method': 'not_supported'}

        # Preprocess data
        X_scaled, y_encoded = self.preprocess_data(X_train, y_train)

        # Preprocess validation data if provided
        X_val_scaled, y_val_encoded = None, None
        if X_val is not None and y_val is not None:
            X_val_scaled, y_val_encoded = self.preprocess_data(X_val, y_val)
            logger.info(
                f"Using validation-based hyperparameter tuning with {len(X_val)} samples")
        else:
            logger.info("Using cross-validation for hyperparameter tuning")

        # Use validation set if available, otherwise fall back to CV
        if X_val_scaled is not None:
            # Manual grid search using validation set
            best_score = 0
            best_params = None
            best_model = None

            # Generate all parameter combinations
            from itertools import product
            keys = param_grid.keys()
            values = param_grid.values()
            param_combinations = [dict(zip(keys, v)) for v in product(*values)]

            logger.info(
                f"Testing {len(param_combinations)} parameter combinations...")

            for i, params in enumerate(param_combinations, 1):
                # Initialize model with these parameters
                self.model_params.update(params)
                self._initialize_model()

                # Train on training set
                self.model.fit(X_scaled, y_encoded)

                # Evaluate on validation set
                val_pred = self.model.predict(X_val_scaled)
                val_score = accuracy_score(y_val_encoded, val_pred)

                if val_score > best_score:
                    best_score = val_score
                    best_params = params.copy()
                    best_model = self.model
                    logger.info(
                        f"  [{i}/{len(param_combinations)}] New best: {val_score:.4f} with {params}")

            # Update model with best parameters
            self.model = best_model
            self.model_params.update(best_params)

            optimization_results = {
                'best_params': best_params,
                'best_score': best_score,
                'method': 'validation_set',
                'n_combinations_tested': len(param_combinations)
            }

        else:
            # Initialize model
            self._initialize_model()

            # Grid search with cross-validation
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_scaled, y_encoded)

            # Update model with best parameters
            self.model = grid_search.best_estimator_
            self.model_params.update(grid_search.best_params_)

            optimization_results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'method': 'cross_validation',
                'cv_results': grid_search.cv_results_
            }

        logger.info(
            f"Hyperparameter optimization completed. Best score: {optimization_results['best_score']:.4f}")
        logger.info(f"Best parameters: {optimization_results['best_params']}")

        return optimization_results

    def _get_default_param_grid(self) -> Dict[str, List]:
        """Get default hyperparameter grid for optimization."""
        if self.model_type == 'random_forest':
            return {
                'n_estimators': [25, 50, 100],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                # Include class balancing options
                'class_weight': ['balanced', 'balanced_subsample']
            }
        elif self.model_type == 'svm':
            return {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                # Include class balancing option
                'class_weight': ['balanced', None]
            }
        elif self.model_type == 'neural_network':
            return {
                'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1]
                # Note: MLPClassifier doesn't support class_weight parameter
            }
        elif self.model_type in ('pytorch_mlp', 'pytorch_cnn'):
            # PyTorch models don't support sklearn GridSearchCV
            return {}
        else:
            return {}

    def save_model(self, filepath: str):
        """Save the trained model and preprocessing components."""
        if self.model is None:
            raise ValueError("No trained model to save")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'performance_metrics': self.performance_metrics
        }

        # For PyTorch models, also save the exported weights and trainer
        if self.model_type in ('pytorch_mlp', 'pytorch_cnn'):
            import torch
            trainer = getattr(self, '_pytorch_trainer', None)
            if trainer is not None:
                if self.model_type == 'pytorch_mlp':
                    model_data['pytorch_weights'] = trainer.export_mlp_weights()
                else:
                    model_data['pytorch_weights'] = trainer.export_cnn_weights()
                # Save PyTorch state_dict separately for exact reloading
                model_data['pytorch_state_dict'] = self.model.state_dict()
                model_data['pytorch_config'] = getattr(self, '_pytorch_config', {})

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'EdgeMLModel':
        """Load a trained model from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        # Reconstruct the EdgeMLModel instance
        instance = cls(
            model_type=model_data['model_type'],
            **model_data['model_params']
        )

        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.label_encoder = model_data['label_encoder']
        instance.feature_names = model_data['feature_names']
        instance.performance_metrics = model_data['performance_metrics']

        # Restore PyTorch trainer if available
        if model_data['model_type'] in ('pytorch_mlp', 'pytorch_cnn'):
            pytorch_config = model_data.get('pytorch_config', {})
            state_dict = model_data.get('pytorch_state_dict')
            if state_dict is not None and is_pytorch_available():
                instance._pytorch_config = pytorch_config
                # The model object was already restored via joblib
                # but we need to rebuild the trainer wrapper
                trainer = PyTorchTrainer(instance.model)
                instance._pytorch_trainer = trainer

        logger.info(f"Model loaded from {filepath}")
        return instance

    # predict() and predict_proba() defined above (unified for all model types)


def extract_orientation_invariant_features(df: pd.DataFrame, sensor_cols: List[str] = None) -> pd.DataFrame:
    """Extract orientation-invariant features using magnitude vectors.

    These features are robust to device orientation changes, making the model
    work regardless of how the sensor is mounted (left/right wrist, rotated, etc.).

    Args:
        df: DataFrame with sensor data
        sensor_cols: Sensor column names (default: ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ'])

    Returns:
        DataFrame with orientation-invariant magnitude-based features
    """
    if sensor_cols is None:
        sensor_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']

    features = {}

    # Calculate magnitude vectors (MOST IMPORTANT for orientation invariance)
    acc_mag = np.sqrt(df['aX']**2 + df['aY']**2 + df['aZ']**2)
    gyro_mag = np.sqrt(df['gX']**2 + df['gY']**2 + df['gZ']**2)

    # Statistical features on acceleration magnitude
    for name, mag_data in [('acc_mag', acc_mag), ('gyro_mag', gyro_mag)]:
        data = mag_data.values

        # Basic statistical features
        features[f'{name}_mean'] = np.mean(data)
        features[f'{name}_std'] = np.std(data)
        features[f'{name}_min'] = np.min(data)
        features[f'{name}_max'] = np.max(data)
        features[f'{name}_range'] = np.max(data) - np.min(data)
        features[f'{name}_median'] = np.median(data)
        features[f'{name}_q25'] = np.percentile(data, 25)
        features[f'{name}_q75'] = np.percentile(data, 75)
        features[f'{name}_iqr'] = np.percentile(
            data, 75) - np.percentile(data, 25)

        # Advanced statistical features
        features[f'{name}_skewness'] = pd.Series(data).skew()
        features[f'{name}_kurtosis'] = pd.Series(data).kurtosis()
        features[f'{name}_rms'] = np.sqrt(np.mean(data**2))
        features[f'{name}_energy'] = np.sum(data**2)

        # Signal characteristics - sign-product method (consistent with C++ deployment)
        mean_val = np.mean(data)
        # Count strict sign changes: data[i-1] * data[i] < 0
        features[f'{name}_zero_crossings'] = int(
            np.sum(data[:-1] * data[1:] < 0))
        centered = data - mean_val
        mean_crossings = int(np.sum(centered[:-1] * centered[1:] < 0))
        features[f'{name}_mean_crossing_rate'] = mean_crossings / len(data)  # Normalize to rate

    # Jerk magnitude (rate of change of acceleration) - also orientation invariant
    acc_jerk_mag = np.sqrt(
        np.diff(df['aX'])**2 + np.diff(df['aY'])**2 + np.diff(df['aZ'])**2)
    features['acc_jerk_mag_mean'] = np.mean(acc_jerk_mag)
    features['acc_jerk_mag_std'] = np.std(acc_jerk_mag)
    features['acc_jerk_mag_max'] = np.max(acc_jerk_mag)

    return pd.DataFrame([features])


def extract_frequency_magnitude_features(df: pd.DataFrame, sensor_cols: List[str] = None,
                                         sampling_rate: float = 100) -> pd.DataFrame:
    """Extract FFT features from magnitude vectors (orientation invariant).

    Frequency components of magnitude vectors are independent of device orientation.
    Walking has the same step frequency regardless of which wrist or rotation.

    Args:
        df: DataFrame with sensor data
        sensor_cols: Sensor column names
        sampling_rate: Sampling rate in Hz

    Returns:
        DataFrame with frequency domain magnitude features
    """
    if sensor_cols is None:
        sensor_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']

    features = {}

    # Calculate magnitude vectors
    acc_mag = np.sqrt(df['aX']**2 + df['aY']**2 + df['aZ']**2)
    gyro_mag = np.sqrt(df['gX']**2 + df['gY']**2 + df['gZ']**2)

    for name, mag_data in [('acc_mag', acc_mag), ('gyro_mag', gyro_mag)]:
        data = mag_data.values

        # Compute FFT
        fft_vals = np.fft.fft(data)
        fft_magnitude = np.abs(fft_vals)
        fft_freq = np.fft.fftfreq(len(data), 1/sampling_rate)

        # Only use positive frequencies
        pos_mask = fft_freq > 0
        fft_magnitude_pos = fft_magnitude[pos_mask]
        fft_freq_pos = fft_freq[pos_mask]

        # Dominant frequency (most important for activity classification)
        dominant_freq_idx = np.argmax(fft_magnitude_pos)
        features[f'{name}_dominant_frequency'] = fft_freq_pos[dominant_freq_idx]
        features[f'{name}_dominant_frequency_magnitude'] = fft_magnitude_pos[dominant_freq_idx]

        # Spectral centroid (center of mass of spectrum)
        features[f'{name}_spectral_centroid'] = np.sum(
            fft_freq_pos * fft_magnitude_pos) / np.sum(fft_magnitude_pos)

        # Energy in frequency bands
        low_freq_mask = (fft_freq_pos >= 0) & (
            fft_freq_pos < 2)  # 0-2 Hz (walking/slow)
        mid_freq_mask = (fft_freq_pos >= 2) & (
            fft_freq_pos < 5)  # 2-5 Hz (running/fast)
        high_freq_mask = (fft_freq_pos >= 5) & (
            fft_freq_pos < sampling_rate/2)  # >5 Hz

        features[f'{name}_energy_low_freq'] = np.sum(
            fft_magnitude_pos[low_freq_mask]**2)
        features[f'{name}_energy_mid_freq'] = np.sum(
            fft_magnitude_pos[mid_freq_mask]**2)
        features[f'{name}_energy_high_freq'] = np.sum(
            fft_magnitude_pos[high_freq_mask]**2)

        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumsum = np.cumsum(fft_magnitude_pos)
        if cumsum[-1] > 0:
            rolloff_threshold = 0.85 * cumsum[-1]
            rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
            if len(rolloff_idx) > 0:
                features[f'{name}_spectral_rolloff'] = fft_freq_pos[rolloff_idx[0]]
            else:
                features[f'{name}_spectral_rolloff'] = fft_freq_pos[-1]
        else:
            features[f'{name}_spectral_rolloff'] = 0.0

    return pd.DataFrame([features])


def extract_time_domain_features(df: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
    """Extract time-domain statistical features from sensor data."""
    features = {}

    for col in sensor_cols:
        if col not in df.columns:
            continue

        data = df[col].values

        # Basic statistical features
        features[f'{col}_mean'] = np.mean(data)
        features[f'{col}_std'] = np.std(data)
        features[f'{col}_min'] = np.min(data)
        features[f'{col}_max'] = np.max(data)
        features[f'{col}_range'] = np.max(data) - np.min(data)
        features[f'{col}_median'] = np.median(data)
        features[f'{col}_q25'] = np.percentile(data, 25)
        features[f'{col}_q75'] = np.percentile(data, 75)
        features[f'{col}_iqr'] = np.percentile(
            data, 75) - np.percentile(data, 25)

        # Advanced statistical features
        features[f'{col}_skewness'] = pd.Series(data).skew()
        features[f'{col}_kurtosis'] = pd.Series(data).kurtosis()
        features[f'{col}_rms'] = np.sqrt(np.mean(data**2))
        features[f'{col}_energy'] = np.sum(data**2)

        # Signal characteristics - sign-product method (consistent with C++ deployment)
        # Count strict sign changes: data[i-1] * data[i] < 0
        features[f'{col}_zero_crossings'] = int(
            np.sum(data[:-1] * data[1:] < 0))
        centered = data - np.mean(data)
        mean_crossings = int(np.sum(centered[:-1] * centered[1:] < 0))
        features[f'{col}_mean_crossing_rate'] = mean_crossings / len(data)  # Normalize to rate

    return pd.DataFrame([features])


def extract_frequency_domain_features(df: pd.DataFrame, sensor_cols: List[str],
                                      sampling_rate: float = 100) -> pd.DataFrame:
    """Extract frequency-domain features using FFT."""
    features = {}

    for col in sensor_cols:
        if col not in df.columns:
            continue

        data = df[col].values

        # Compute FFT
        fft_vals = np.fft.fft(data)
        fft_magnitude = np.abs(fft_vals)
        fft_freq = np.fft.fftfreq(len(data), 1/sampling_rate)

        # Only use positive frequencies
        pos_mask = fft_freq > 0
        fft_magnitude_pos = fft_magnitude[pos_mask]
        fft_freq_pos = fft_freq[pos_mask]

        # Frequency domain features
        features[f'{col}_spectral_centroid'] = np.sum(
            fft_freq_pos * fft_magnitude_pos) / np.sum(fft_magnitude_pos)
        features[f'{col}_spectral_rolloff'] = fft_freq_pos[np.where(
            np.cumsum(fft_magnitude_pos) >= 0.85 * np.sum(fft_magnitude_pos))[0][0]]
        features[f'{col}_spectral_bandwidth'] = np.sqrt(np.sum(
            ((fft_freq_pos - features[f'{col}_spectral_centroid'])**2) * fft_magnitude_pos) / np.sum(fft_magnitude_pos))

        # Dominant frequency
        dominant_freq_idx = np.argmax(fft_magnitude_pos)
        features[f'{col}_dominant_frequency'] = fft_freq_pos[dominant_freq_idx]
        features[f'{col}_dominant_frequency_magnitude'] = fft_magnitude_pos[dominant_freq_idx]

        # Energy in frequency bands
        low_freq_mask = (fft_freq_pos >= 0) & (fft_freq_pos < 5)
        mid_freq_mask = (fft_freq_pos >= 5) & (fft_freq_pos < 15)
        high_freq_mask = (fft_freq_pos >= 15) & (
            fft_freq_pos < sampling_rate/2)

        features[f'{col}_energy_low_freq'] = np.sum(
            fft_magnitude_pos[low_freq_mask]**2)
        features[f'{col}_energy_mid_freq'] = np.sum(
            fft_magnitude_pos[mid_freq_mask]**2)
        features[f'{col}_energy_high_freq'] = np.sum(
            fft_magnitude_pos[high_freq_mask]**2)

    return pd.DataFrame([features])


def create_feature_vector(df: pd.DataFrame, sensor_cols: List[str] = None,
                          sampling_rate: float = 100, include_frequency: bool = True,
                          orientation_robust: bool = True, include_per_axis: bool = False) -> pd.DataFrame:
    """Create comprehensive feature vector from raw sensor data.

    Args:
        df: Window DataFrame with sensor data columns
        sensor_cols: Sensor column names (default: ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ'])
        sampling_rate: Sampling rate in Hz (default: 100)
        include_frequency: Include FFT features (default: True)
        orientation_robust: Use magnitude-based features (RECOMMENDED - default: True)
        include_per_axis: Include per-axis features (less robust, default: False)

    Returns:
        DataFrame with extracted features

    Feature Counts:
        - Magnitude only (robust): ~33 features (no FFT) or ~49 features (with FFT)
        - Per-axis only: ~90 features (no FFT) or ~138 features (with FFT)
        - Both: ~123 features (no FFT) or ~187 features (with FFT)
    """
    if sensor_cols is None:
        sensor_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']

    features_list = []

    # ORIENTATION-ROBUST FEATURES (magnitude-based) - RECOMMENDED
    if orientation_robust:
        # Time-domain magnitude features (30 features)
        mag_features = extract_orientation_invariant_features(df, sensor_cols)
        features_list.append(mag_features)

        # Frequency-domain magnitude features (16 features) - if enabled
        if include_frequency:
            freq_mag_features = extract_frequency_magnitude_features(
                df, sensor_cols, sampling_rate)
            features_list.append(freq_mag_features)

    # PER-AXIS FEATURES (orientation-dependent) - OPTIONAL
    if include_per_axis:
        # Time-domain per-axis features (90 features)
        time_features = extract_time_domain_features(df, sensor_cols)
        features_list.append(time_features)

        # Frequency-domain per-axis features (48 features) - if enabled
        if include_frequency:
            freq_features = extract_frequency_domain_features(
                df, sensor_cols, sampling_rate)
            features_list.append(freq_features)

    # If neither enabled, fall back to per-axis time features
    if not orientation_robust and not include_per_axis:
        logger.warning(
            "No features enabled! Using per-axis time features as fallback.")
        time_features = extract_time_domain_features(df, sensor_cols)
        features_list.append(time_features)

    # Combine all feature sets
    combined_features = pd.concat(features_list, axis=1)

    return combined_features


def prepare_training_data(window_files: List[str], labels: List[str],
                          sensor_cols: List[str] = None, sampling_rate: float = 100,
                          include_frequency: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare training data from window files."""
    if len(window_files) != len(labels):
        raise ValueError("Number of window files must match number of labels")

    if sensor_cols is None:
        sensor_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']

    all_features = []
    all_labels = []

    for file_path, label in zip(window_files, labels):
        if not os.path.exists(file_path):
            logger.warning(f"Window file not found: {file_path}")
            continue

        # Load window data
        window_df = pd.read_csv(file_path)

        # Extract features
        features = create_feature_vector(
            window_df, sensor_cols, sampling_rate, include_frequency)

        # Add to collections
        all_features.append(features)
        all_labels.append(label)

    # Combine all features
    if all_features:
        X = pd.concat(all_features, ignore_index=True)
        y = pd.Series(all_labels)
        return X, y
    else:
        raise ValueError("No valid window files found")


# Model factory function
def create_model(model_type: str, **kwargs) -> EdgeMLModel:
    """Factory function to create edge ML models."""
    supported_models = ['random_forest', 'svm', 'neural_network',
                        'pytorch_mlp', 'pytorch_cnn']

    if model_type not in supported_models:
        raise ValueError(
            f"Model type '{model_type}' not supported. Choose from: {supported_models}")

    return EdgeMLModel(model_type, **kwargs)
