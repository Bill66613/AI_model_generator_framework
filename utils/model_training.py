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
                min_samples_split=self.model_params.get('min_samples_split', 5),
                min_samples_leaf=self.model_params.get('min_samples_leaf', 2),
                random_state=42
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                C=self.model_params.get('C', 1.0),
                kernel=self.model_params.get('kernel', 'rbf'),
                gamma=self.model_params.get('gamma', 'scale'),
                random_state=42,
                probability=True  # Enable probability estimates
            )
        elif self.model_type == 'neural_network':
            hidden_layers = self.model_params.get('hidden_layer_sizes', (100, 50))
            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation=self.model_params.get('activation', 'relu'),
                solver=self.model_params.get('solver', 'adam'),
                alpha=self.model_params.get('alpha', 0.0001),
                learning_rate_init=self.model_params.get('learning_rate', 0.001),
                max_iter=self.model_params.get('max_iter', 500),
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
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
            X_array = X
            
        # Initialize and fit scaler if not exists
        if self.scaler is None:
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
              scaler_type: str = 'standard', use_cross_validation: bool = True) -> Dict[str, Any]:
        """Train the model with the provided data."""
        logger.info(f"Training {self.model_type} model...")
        
        # Initialize model
        self._initialize_model()
        
        # Preprocess data
        X_scaled, y_encoded = self.preprocess_data(X_train, y_train, scaler_type)
        
        # Train model
        self.model.fit(X_scaled, y_encoded)
        
        # Cross-validation for model evaluation
        if use_cross_validation:
            cv_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=5)
            self.performance_metrics['cv_mean_accuracy'] = cv_scores.mean()
            self.performance_metrics['cv_std_accuracy'] = cv_scores.std()
            logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Training accuracy
        train_predictions = self.model.predict(X_scaled)
        train_accuracy = accuracy_score(y_encoded, train_predictions)
        self.performance_metrics['train_accuracy'] = train_accuracy
        
        logger.info(f"Training completed. Training accuracy: {train_accuracy:.4f}")
        
        return self.performance_metrics
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate the trained model on test data."""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Preprocess test data
        X_scaled, y_encoded = self.preprocess_data(X_test, y_test)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled) if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_encoded, y_pred)
        conf_matrix = confusion_matrix(y_encoded, y_pred)
        class_report = classification_report(y_encoded, y_pred, output_dict=True)
        
        # Store evaluation results
        evaluation_results = {
            'test_accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'predictions': y_pred.tolist(),
            'actual_labels': y_encoded.tolist()
        }
        
        if y_pred_proba is not None:
            evaluation_results['prediction_probabilities'] = y_pred_proba.tolist()
        
        # Update performance metrics
        self.performance_metrics.update(evaluation_results)
        
        logger.info(f"Model evaluation completed. Test accuracy: {accuracy:.4f}")
        
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
                importance_dict = {f"feature_{i}": imp for i, imp in enumerate(importances)}
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
                importance_dict = {f"feature_{i}": imp for i, imp in enumerate(importances)}
        
        return importance_dict
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                param_grid: Dict[str, List] = None) -> Dict[str, Any]:
        """Optimize hyperparameters using grid search."""
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        # Initialize model
        self._initialize_model()
        
        # Preprocess data
        X_scaled, y_encoded = self.preprocess_data(X_train, y_train)
        
        # Grid search
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
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Hyperparameter optimization completed. Best score: {grid_search.best_score_:.4f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return optimization_results
    
    def _get_default_param_grid(self) -> Dict[str, List]:
        """Get default hyperparameter grid for optimization."""
        if self.model_type == 'random_forest':
            return {
                'n_estimators': [25, 50, 100],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'svm':
            return {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
        elif self.model_type == 'neural_network':
            return {
                'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
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
        
        logger.info(f"Model loaded from {filepath}")
        return instance
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled, _ = self.preprocess_data(X)
        predictions = self.model.predict(X_scaled)
        
        # Convert back to original labels if label encoder exists
        if self.label_encoder:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Get prediction probabilities if available."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            return None
        
        X_scaled, _ = self.preprocess_data(X)
        return self.model.predict_proba(X_scaled)


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
        features[f'{col}_iqr'] = np.percentile(data, 75) - np.percentile(data, 25)
        
        # Advanced statistical features
        features[f'{col}_skewness'] = pd.Series(data).skew()
        features[f'{col}_kurtosis'] = pd.Series(data).kurtosis()
        features[f'{col}_rms'] = np.sqrt(np.mean(data**2))
        features[f'{col}_energy'] = np.sum(data**2)
        
        # Signal characteristics
        features[f'{col}_zero_crossings'] = len(np.where(np.diff(np.sign(data)))[0])
        features[f'{col}_mean_crossing_rate'] = len(np.where(np.diff(np.sign(data - np.mean(data))))[0])
    
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
        features[f'{col}_spectral_centroid'] = np.sum(fft_freq_pos * fft_magnitude_pos) / np.sum(fft_magnitude_pos)
        features[f'{col}_spectral_rolloff'] = fft_freq_pos[np.where(np.cumsum(fft_magnitude_pos) >= 0.85 * np.sum(fft_magnitude_pos))[0][0]]
        features[f'{col}_spectral_bandwidth'] = np.sqrt(np.sum(((fft_freq_pos - features[f'{col}_spectral_centroid'])**2) * fft_magnitude_pos) / np.sum(fft_magnitude_pos))
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(fft_magnitude_pos)
        features[f'{col}_dominant_frequency'] = fft_freq_pos[dominant_freq_idx]
        features[f'{col}_dominant_frequency_magnitude'] = fft_magnitude_pos[dominant_freq_idx]
        
        # Energy in frequency bands
        low_freq_mask = (fft_freq_pos >= 0) & (fft_freq_pos < 5)
        mid_freq_mask = (fft_freq_pos >= 5) & (fft_freq_pos < 15)
        high_freq_mask = (fft_freq_pos >= 15) & (fft_freq_pos < sampling_rate/2)
        
        features[f'{col}_energy_low_freq'] = np.sum(fft_magnitude_pos[low_freq_mask]**2)
        features[f'{col}_energy_mid_freq'] = np.sum(fft_magnitude_pos[mid_freq_mask]**2)
        features[f'{col}_energy_high_freq'] = np.sum(fft_magnitude_pos[high_freq_mask]**2)
    
    return pd.DataFrame([features])


def create_feature_vector(df: pd.DataFrame, sensor_cols: List[str] = None, 
                         sampling_rate: float = 100, include_frequency: bool = True) -> pd.DataFrame:
    """Create comprehensive feature vector from raw sensor data."""
    if sensor_cols is None:
        sensor_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
    
    # Extract time-domain features
    time_features = extract_time_domain_features(df, sensor_cols)
    
    # Extract frequency-domain features
    if include_frequency:
        freq_features = extract_frequency_domain_features(df, sensor_cols, sampling_rate)
        # Combine features
        combined_features = pd.concat([time_features, freq_features], axis=1)
    else:
        combined_features = time_features
    
    return combined_features


def prepare_training_data(window_files: List[str], labels: List[str], 
                         sensor_cols: List[str] = None, sampling_rate: float = 100) -> Tuple[pd.DataFrame, pd.Series]:
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
        features = create_feature_vector(window_df, sensor_cols, sampling_rate)
        
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
    supported_models = ['random_forest', 'svm', 'neural_network']
    
    if model_type not in supported_models:
        raise ValueError(f"Model type '{model_type}' not supported. Choose from: {supported_models}")
    
    return EdgeMLModel(model_type, **kwargs)
