"""
Edge ML Model for Human Activity Recognition

Provides the EdgeMLModel class — a unified interface for training,
evaluating, saving, and loading machine learning models optimized for
edge deployment. Supports sklearn models (Random Forest, SVM, MLP) and
PyTorch models (MLP, 1D-CNN, 2D-CNN).
"""

import numpy as np
import pandas as pd
import copy
from typing import Tuple, Dict, Any, Optional, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import joblib
import os
import logging

from utils.pytorch_models import (
    is_pytorch_available, PyTorchTrainer, HARMLP, HARCNN, HARCNN2D
)

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
            self.model = None  # placeholder — created in train()
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
        elif self.model_type == 'pytorch_cnn2d':
            if not is_pytorch_available():
                raise RuntimeError("PyTorch is required for pytorch_cnn2d. "
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
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = np.asarray(X) if not isinstance(X, np.ndarray) else X

        # CNN uses raw 3D windows — skip scaling entirely.
        # Batch normalization layers within the CNN handle internal normalization,
        # so applying StandardScaler would distort the raw signal characteristics
        # that convolutional layers need to learn spatial/temporal patterns.
        if self.model_type in ('pytorch_cnn', 'pytorch_cnn2d'):
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
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
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

        self._initialize_model()

        X_scaled, y_encoded = self.preprocess_data(
            X_train, y_train, scaler_type)

        X_val_scaled, y_val_encoded = None, None
        if X_val is not None and y_val is not None:
            X_val_scaled, y_val_encoded = self.preprocess_data(X_val, y_val)
            logger.info(f"Using validation set: {len(X_val)} samples")

        if self.model_type == 'neural_network' and X_val_scaled is not None:
            from sklearn.neural_network import MLPClassifier

            val_accuracies = []
            train_accuracies = []
            best_val_accuracy = 0
            best_model_state = None
            patience_counter = 0
            patience = 10

            logger.info("Training with validation-based early stopping...")

            for epoch in range(self.model.max_iter):
                self.model.max_iter = epoch + 1
                self.model.warm_start = True
                self.model.fit(X_scaled, y_encoded)

                val_pred = self.model.predict(X_val_scaled)
                val_acc = accuracy_score(y_val_encoded, val_pred)
                val_accuracies.append(val_acc)

                train_pred = self.model.predict(X_scaled)
                train_acc = accuracy_score(y_encoded, train_pred)
                train_accuracies.append(train_acc)

                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                    best_model_state = copy.deepcopy(self.model)
                    patience_counter = 0
                    logger.info(
                        f"Epoch {epoch+1}: Val Acc={val_acc:.4f} (improved) - Train Acc={train_acc:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(
                            f"Early stopping at epoch {epoch+1}: No improvement for {patience} epochs")
                        break

            # Restore best model weights (not just hyperparams)
            if best_model_state is not None:
                self.model = best_model_state
                logger.info(f"Restored best model from epoch with val_acc={best_val_accuracy:.4f}")

            self.performance_metrics['val_accuracies'] = val_accuracies
            self.performance_metrics['train_accuracies'] = train_accuracies
            self.performance_metrics['best_val_accuracy'] = best_val_accuracy
            self.performance_metrics['stopped_epoch'] = epoch + 1
            self.performance_metrics['early_stopped'] = (
                patience_counter >= patience)

        elif self.model_type in ('pytorch_mlp', 'pytorch_cnn', 'pytorch_cnn2d'):
            cfg = self._pytorch_config
            num_classes = len(set(y_encoded))
            input_size = X_scaled.shape[1]

            if self.model_type in ('pytorch_cnn', 'pytorch_cnn2d'):
                if np.asarray(X_scaled).ndim != 3:
                    raise ValueError(
                        f"{self.model_type} expects 3D input (n_samples, window_size, n_channels), "
                        f"got shape {np.asarray(X_scaled).shape}"
                    )
                cfg['window_size'] = int(np.asarray(X_scaled).shape[1])
                cfg['n_channels'] = int(np.asarray(X_scaled).shape[2])
                self.model_params['window_size'] = cfg['window_size']
                self.model_params['n_channels'] = cfg['n_channels']

            if self.model_type == 'pytorch_mlp':
                net = HARMLP(
                    input_size=input_size,
                    hidden_sizes=cfg['hidden_sizes'],
                    num_classes=num_classes,
                    dropout=cfg['dropout'],
                )
            elif self.model_type == 'pytorch_cnn2d':
                net = HARCNN2D(
                    window_size=cfg['window_size'],
                    n_channels=cfg['n_channels'],
                    num_classes=num_classes,
                    dropout=cfg['dropout'],
                )
            else:
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

            self._pytorch_trainer = trainer
            self.model = net
            self.performance_metrics.update(pt_metrics)

        else:
            self.model.fit(X_scaled, y_encoded)

        if use_cross_validation and X_val is None and self.model_type not in (
                'pytorch_mlp', 'pytorch_cnn', 'pytorch_cnn2d'):
            cv_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=5)
            self.performance_metrics['cv_mean_accuracy'] = cv_scores.mean()
            self.performance_metrics['cv_std_accuracy'] = cv_scores.std()
            logger.info(
                f"Cross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        train_predictions = self.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        self.performance_metrics['train_accuracy'] = train_accuracy

        logger.info(
            f"Training completed. Training accuracy: {train_accuracy:.4f}")

        return self.performance_metrics

    def predict(self, X) -> np.ndarray:
        """Predict class labels. Returns decoded string labels."""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X) if not isinstance(X, np.ndarray) else X

        if self.scaler is not None and self.model_type not in ('pytorch_cnn', 'pytorch_cnn2d'):
            X_scaled = self.scaler.transform(X_array)
        else:
            X_scaled = X_array

        if self.model_type in ('pytorch_mlp', 'pytorch_cnn', 'pytorch_cnn2d'):
            encoded_preds = self._pytorch_trainer.predict(X_scaled)
        else:
            encoded_preds = self.model.predict(X_scaled)

        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(encoded_preds)
        return encoded_preds

    def predict_proba(self, X) -> Optional[np.ndarray]:
        """Return class probabilities."""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X) if not isinstance(X, np.ndarray) else X

        if self.scaler is not None and self.model_type not in ('pytorch_cnn', 'pytorch_cnn2d'):
            X_scaled = self.scaler.transform(X_array)
        else:
            X_scaled = X_array

        if self.model_type in ('pytorch_mlp', 'pytorch_cnn', 'pytorch_cnn2d'):
            return self._pytorch_trainer.predict_proba(X_scaled)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        return None

    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        """Evaluate the trained model on test data."""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")

        X_scaled, y_encoded = self.preprocess_data(X_test, y_test)

        if self.model_type in ('pytorch_mlp', 'pytorch_cnn', 'pytorch_cnn2d'):
            y_pred = self._pytorch_trainer.predict(X_scaled)
            y_pred_proba = self._pytorch_trainer.predict_proba(X_scaled)
        else:
            y_pred = self.model.predict(X_scaled)
            y_pred_proba = self.model.predict_proba(X_scaled) if hasattr(
                self.model, 'predict_proba') else None

        accuracy = accuracy_score(y_encoded, y_pred)
        conf_matrix = confusion_matrix(y_encoded, y_pred)

        if self.label_encoder is not None:
            target_names = self.label_encoder.classes_.tolist()
            class_report = classification_report(
                y_encoded, y_pred, target_names=target_names, output_dict=True)
        else:
            target_names = None
            class_report = classification_report(
                y_encoded, y_pred, output_dict=True)

        evaluation_results = {
            'test_accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'predictions': y_pred.tolist(),
            'actual_labels': y_encoded.tolist(),
            'label_names': target_names
        }

        if y_pred_proba is not None:
            evaluation_results['prediction_probabilities'] = y_pred_proba.tolist()

        self.performance_metrics.update(evaluation_results)

        logger.info(
            f"Model evaluation completed. Test accuracy: {accuracy:.4f}")

        return evaluation_results

    def evaluate_with_confidence_threshold(
        self, X_test, y_test, confidence_threshold: float = 0.6,
        smoothing_window: int = 1,
    ) -> Dict[str, Any]:
        """Evaluate model simulating on-device confidence rejection and smoothing.

        This mirrors the C++ har_predict() behavior: predictions with
        max probability < confidence_threshold are rejected as "unknown".
        When smoothing_window > 1, applies majority voting over consecutive
        predictions (matching the C++ temporal smoothing logic).

        Args:
            X_test: Test features
            y_test: True labels
            confidence_threshold: Min confidence to accept (default 0.6,
                matches C++ CONFIDENCE_THRESHOLD)
            smoothing_window: Number of consecutive predictions for majority
                vote (default 1 = no smoothing, matches C++ SMOOTHING_WINDOW)

        Returns:
            Dict with aggregate metrics:
            standard_accuracy, deployment_accuracy, accepted_accuracy,
            rejection_rate, sample counts, confidence summary, and
            final_predictions (-1 means unknown).
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")

        X_scaled, y_encoded = self.preprocess_data(X_test, y_test)

        if self.model_type in ('pytorch_mlp', 'pytorch_cnn', 'pytorch_cnn2d'):
            y_pred = self._pytorch_trainer.predict(X_scaled)
            y_pred_proba = self._pytorch_trainer.predict_proba(X_scaled)
        else:
            y_pred = self.model.predict(X_scaled)
            y_pred_proba = (self.model.predict_proba(X_scaled)
                            if hasattr(self.model, 'predict_proba') else None)

        n_total = len(y_encoded)
        y_pred_raw = y_pred.copy()  # Save before smoothing modifies it

        if y_pred_proba is not None:
            max_confidences = np.max(y_pred_proba, axis=1)
            accepted_mask = max_confidences >= confidence_threshold
        else:
            # No probabilities available — accept all (no threshold filtering)
            accepted_mask = np.ones(n_total, dtype=bool)
            max_confidences = np.ones(n_total)

        n_accepted = int(np.sum(accepted_mask))
        n_rejected = n_total - n_accepted
        rejection_rate = n_rejected / n_total if n_total > 0 else 0.0
        thresholded_preds = y_pred.copy()
        thresholded_preds[~accepted_mask] = -1

        # Standard accuracy (no threshold, no smoothing) for comparison
        standard_accuracy = float(accuracy_score(y_encoded, y_pred_raw))

        # Apply majority-vote smoothing (mirrors C++ temporal smoothing)
        smoothing_window = max(1, min(9, smoothing_window))
        if smoothing_window > 1 and n_total > 0:
            # Build per-window predictions: rejected → -1 (unknown)
            raw_preds = thresholded_preds.copy()

            # Majority vote over sliding window (C++ parity):
            # unknown votes are counted and unknown wins ties
            smoothed_preds = np.full(n_total, -1)
            for i in range(n_total):
                start = max(0, i - smoothing_window + 1)
                window_preds = raw_preds[start:i + 1]
                unknown_votes = int(np.sum(window_preds == -1))
                best_class = -1
                best_votes = unknown_votes

                for cls in np.unique(window_preds):
                    if cls < 0:
                        continue
                    cls_votes = int(np.sum(window_preds == cls))
                    if cls_votes > best_votes:
                        best_votes = cls_votes
                        best_class = int(cls)

                smoothed_preds[i] = best_class  # remains -1 if unknown wins/ties

            # Recalculate metrics with smoothed predictions
            smoothed_accepted = smoothed_preds != -1
            n_accepted = int(np.sum(smoothed_accepted))
            n_rejected = n_total - n_accepted
            rejection_rate = n_rejected / n_total if n_total > 0 else 0.0
            y_pred = smoothed_preds
        else:
            y_pred = thresholded_preds

        # Accuracy on accepted predictions only
        if n_accepted > 0:
            accepted_accuracy = float(accuracy_score(
                y_encoded[accepted_mask if smoothing_window <= 1 else smoothed_accepted],
                y_pred[accepted_mask if smoothing_window <= 1 else smoothed_accepted]))
        else:
            accepted_accuracy = 0.0

        # Deployment accuracy: rejected = wrong (device says "unknown")
        deployment_accuracy = n_accepted * accepted_accuracy / n_total if n_total > 0 else 0.0

        return {
            'standard_accuracy': standard_accuracy,
            'deployment_accuracy': deployment_accuracy,
            'accepted_accuracy': accepted_accuracy,
            'rejection_rate': rejection_rate,
            'n_total': n_total,
            'n_accepted': n_accepted,
            'n_rejected': n_rejected,
            'confidence_threshold': confidence_threshold,
            'smoothing_window': smoothing_window,
            'mean_confidence': float(np.mean(max_confidences)),
            'min_confidence': float(np.min(max_confidences)),
            'final_predictions': y_pred.tolist(),
        }

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores if available."""
        if self.model is None:
            return None

        importance_dict = None
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if self.feature_names:
                importance_dict = dict(zip(self.feature_names, importances))
            else:
                importance_dict = {f"feature_{i}": imp for i,
                                   imp in enumerate(importances)}
        elif hasattr(self.model, 'coef_'):
            if len(self.model.coef_.shape) > 1:
                importances = np.mean(np.abs(self.model.coef_), axis=0)
            else:
                importances = np.abs(self.model.coef_[0])

            if self.feature_names:
                importance_dict = dict(zip(self.feature_names, importances))
            else:
                importance_dict = {f"feature_{i}": imp for i,
                                   imp in enumerate(importances)}

        return importance_dict

    def optimize_hyperparameters(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        param_grid: Dict[str, List] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using validation set or grid search."""
        if param_grid is None:
            param_grid = self._get_default_param_grid()

        if self.model_type in ('pytorch_mlp', 'pytorch_cnn', 'pytorch_cnn2d'):
            logger.warning("Hyperparameter optimization is not supported for "
                           f"{self.model_type}. Adjust parameters manually.")
            return {'best_params': {}, 'best_score': 0.0,
                    'method': 'not_supported'}

        X_scaled, y_encoded = self.preprocess_data(X_train, y_train)

        X_val_scaled, y_val_encoded = None, None
        if X_val is not None and y_val is not None:
            X_val_scaled, y_val_encoded = self.preprocess_data(X_val, y_val)
            logger.info(
                f"Using validation-based hyperparameter tuning with {len(X_val)} samples")
        else:
            logger.info("Using cross-validation for hyperparameter tuning")

        if X_val_scaled is not None:
            best_score = 0
            best_params = None
            best_model = None

            from itertools import product
            keys = param_grid.keys()
            values = param_grid.values()
            param_combinations = [dict(zip(keys, v)) for v in product(*values)]

            logger.info(
                f"Testing {len(param_combinations)} parameter combinations...")

            for i, params in enumerate(param_combinations, 1):
                self.model_params.update(params)
                self._initialize_model()
                self.model.fit(X_scaled, y_encoded)

                val_pred = self.model.predict(X_val_scaled)
                val_score = accuracy_score(y_val_encoded, val_pred)

                if val_score > best_score:
                    best_score = val_score
                    best_params = params.copy()
                    best_model = self.model
                    logger.info(
                        f"  [{i}/{len(param_combinations)}] New best: {val_score:.4f} with {params}")

            self.model = best_model
            self.model_params.update(best_params)

            optimization_results = {
                'best_params': best_params,
                'best_score': best_score,
                'method': 'validation_set',
                'n_combinations_tested': len(param_combinations)
            }
        else:
            self._initialize_model()
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_scaled, y_encoded)

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
                'class_weight': ['balanced', 'balanced_subsample']
            }
        elif self.model_type == 'svm':
            return {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'class_weight': ['balanced', None]
            }
        elif self.model_type == 'neural_network':
            return {
                'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
        elif self.model_type in ('pytorch_mlp', 'pytorch_cnn', 'pytorch_cnn2d'):
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

        if self.model_type in ('pytorch_mlp', 'pytorch_cnn', 'pytorch_cnn2d'):
            import torch
            trainer = getattr(self, '_pytorch_trainer', None)
            if trainer is not None:
                if self.model_type == 'pytorch_mlp':
                    model_data['pytorch_weights'] = trainer.export_mlp_weights()
                else:
                    model_data['pytorch_weights'] = trainer.export_cnn_weights()
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

        instance = cls(
            model_type=model_data['model_type'],
            **model_data['model_params']
        )

        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.label_encoder = model_data['label_encoder']
        instance.feature_names = model_data['feature_names']
        instance.performance_metrics = model_data['performance_metrics']

        if model_data['model_type'] in ('pytorch_mlp', 'pytorch_cnn', 'pytorch_cnn2d'):
            pytorch_config = model_data.get('pytorch_config', {})
            state_dict = model_data.get('pytorch_state_dict')
            if state_dict is not None and is_pytorch_available():
                instance._pytorch_config = pytorch_config
                trainer = PyTorchTrainer(instance.model)
                instance._pytorch_trainer = trainer

        logger.info(f"Model loaded from {filepath}")
        return instance
