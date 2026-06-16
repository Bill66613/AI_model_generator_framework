"""
Test suite for Human Activity Recognition Framework
Tests core functionality of data processing, model training, and deployment
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import json
from unittest.mock import Mock, patch

# Import modules to test
from utils.data_processing import clean_data, low_pass_filter, parse_csv
from utils.model_training import EdgeMLModel, extract_time_domain_features, extract_frequency_domain_features
from config.config import PERSISTENT_DIR, METADATA_FILE


class TestDataProcessing:
    """Test data processing utilities."""
    
    def test_clean_data_remove_missing(self):
        """Test data cleaning with missing value removal."""
        # Create test data with NaN values
        data = {
            'aX': [1.0, 2.0, np.nan, 4.0, 5.0],
            'aY': [1.1, 2.1, 3.1, np.nan, 5.1],
            'aZ': [1.2, 2.2, 3.2, 4.2, 5.2],
            'gX': [0.1, 0.2, 0.3, 0.4, 0.5],
            'gY': [0.11, 0.21, 0.31, 0.41, 0.51],
            'gZ': [0.12, 0.22, 0.32, 0.42, 0.52]
        }
        df = pd.DataFrame(data)
        
        cleaned_df = clean_data(df, 'remove_missing')
        
        # Should remove rows with NaN values
        assert len(cleaned_df) == 3  # Original 5 rows - 2 with NaN
        assert not cleaned_df.isnull().any().any()
    
    def test_clean_data_filter_outliers(self):
        """Test data cleaning with outlier removal."""
        # Create test data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        outlier_data = np.concatenate([normal_data, [10, -10, 15]])  # Add outliers
        
        data = {
            'aX': outlier_data,
            'aY': outlier_data,
            'aZ': outlier_data,
            'gX': outlier_data,
            'gY': outlier_data,
            'gZ': outlier_data
        }
        df = pd.DataFrame(data)
        
        cleaned_df = clean_data(df, 'filter_outliers')
        
        # Should remove outliers (values beyond 3 standard deviations)
        assert len(cleaned_df) < len(df)
    
    def test_low_pass_filter(self):
        """Test low-pass filtering functionality."""
        # Create test signal
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)  # 5Hz + 20Hz
        
        data = pd.DataFrame({
            'aX': signal,
            'aY': signal,
            'aZ': signal
        })
        
        filtered_data = low_pass_filter(data, cutoff=10, fs=100, order=2)
        
        # Check that high frequency component is reduced
        assert isinstance(filtered_data, pd.DataFrame)
        assert filtered_data.shape == data.shape
        assert np.std(filtered_data['aX']) < np.std(data['aX'])  # Reduced variation


class TestModelTraining:
    """Test machine learning model training utilities."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample sensor data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic IMU data for different activities
        walking_ax = np.random.normal(0.5, 0.2, n_samples // 2)
        standing_ax = np.random.normal(0.1, 0.05, n_samples // 2)
        
        data = pd.DataFrame({
            'aX': np.concatenate([walking_ax, standing_ax]),
            'aY': np.random.normal(0, 0.1, n_samples),
            'aZ': np.random.normal(9.8, 0.2, n_samples),
            'gX': np.random.normal(0, 0.05, n_samples),
            'gY': np.random.normal(0, 0.05, n_samples),
            'gZ': np.random.normal(0, 0.05, n_samples)
        })
        
        labels = ['walking'] * (n_samples // 2) + ['standing'] * (n_samples // 2)
        
        return data, labels
    
    def test_extract_time_domain_features(self, sample_data):
        """Test time-domain feature extraction."""
        data, _ = sample_data
        sensor_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
        
        features = extract_time_domain_features(data, sensor_cols)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 1  # Should return one row of features
        
        # Check for expected feature columns
        expected_features = ['mean', 'std', 'min', 'max', 'range', 'median']
        for sensor in sensor_cols:
            for feature in expected_features:
                assert f'{sensor}_{feature}' in features.columns
    
    def test_extract_frequency_domain_features(self, sample_data):
        """Test frequency-domain feature extraction."""
        data, _ = sample_data
        sensor_cols = ['aX', 'aY', 'aZ']
        
        features = extract_frequency_domain_features(data, sensor_cols, sampling_rate=100)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 1
        
        # Check for expected frequency features
        expected_features = ['spectral_centroid', 'spectral_rolloff', 'dominant_frequency']
        for sensor in sensor_cols:
            for feature in expected_features:
                assert f'{sensor}_{feature}' in features.columns
    
    def test_edge_ml_model_random_forest(self, sample_data):
        """Test EdgeMLModel with Random Forest."""
        data, labels = sample_data
        
        # Create features
        sensor_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
        features = extract_time_domain_features(data, sensor_cols)
        
        # Duplicate features to simulate multiple samples
        X = pd.concat([features] * len(labels), ignore_index=True)
        y = pd.Series(labels)
        
        # Train model
        model = EdgeMLModel('random_forest', n_estimators=10, max_depth=5)
        training_results = model.train(X, y, use_cross_validation=False)
        
        assert model.model is not None
        assert 'train_accuracy' in training_results
        assert training_results['train_accuracy'] > 0
        
        # Test prediction
        predictions = model.predict(X.head(5))
        assert len(predictions) == 5

    def test_evaluate_with_confidence_threshold_metrics(self):
        """Test confidence-threshold evaluation metrics on known probabilities."""
        model = EdgeMLModel('random_forest')
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 1, 1, 0])
        mock_model.predict_proba.return_value = np.array([
            [0.80, 0.20],  # accepted, correct
            [0.55, 0.45],  # rejected
            [0.20, 0.80],  # accepted, correct
            [0.45, 0.55],  # rejected
            [0.70, 0.30],  # accepted, wrong
        ])
        model.model = mock_model

        X_test = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'f2': [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        y_test = pd.Series([0, 0, 1, 1, 1])

        results = model.evaluate_with_confidence_threshold(
            X_test, y_test, confidence_threshold=0.6)

        assert results['n_total'] == 5
        assert results['n_accepted'] == 3
        assert results['n_rejected'] == 2
        assert results['rejection_rate'] == pytest.approx(0.4)
        assert results['accepted_accuracy'] == pytest.approx(2 / 3)
        assert results['deployment_accuracy'] == pytest.approx(0.4)
        assert results['final_predictions'] == [0, -1, 1, -1, 0]

    def test_evaluate_with_confidence_threshold_smoothing_unknown_tie(self):
        """Smoothing should count unknown votes and let unknown win ties (C++ parity)."""
        model = EdgeMLModel('random_forest')
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 1, 0, 1])
        mock_model.predict_proba.return_value = np.array([
            [0.90, 0.10],  # accepted -> 0
            [0.55, 0.45],  # rejected -> -1
            [0.10, 0.90],  # accepted -> 1
            [0.52, 0.48],  # rejected -> -1
            [0.05, 0.95],  # accepted -> 1
        ])
        model.model = mock_model

        X_test = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'f2': [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        y_test = pd.Series([0, 1, 1, 0, 1])

        results = model.evaluate_with_confidence_threshold(
            X_test, y_test, confidence_threshold=0.6, smoothing_window=3)

        # raw accepted/rejected => [0, -1, 1, -1, 1]
        # smoothing (unknown wins ties) => [0, -1, -1, -1, 1]
        assert results['final_predictions'] == [0, -1, -1, -1, 1]
        assert results['n_total'] == 5
        assert results['n_accepted'] == 2
        assert results['n_rejected'] == 3
        assert results['rejection_rate'] == pytest.approx(0.6)
        assert results['accepted_accuracy'] == pytest.approx(1.0)
        assert results['deployment_accuracy'] == pytest.approx(0.4)

    def test_model_save_load(self, sample_data, tmp_path):
        """Test model saving and loading functionality."""
        data, labels = sample_data
        
        # Create and train a simple model
        sensor_cols = ['aX', 'aY', 'aZ']
        features = extract_time_domain_features(data, sensor_cols)
        X = pd.concat([features] * len(labels), ignore_index=True)
        y = pd.Series(labels)
        
        model = EdgeMLModel('random_forest', n_estimators=5)
        model.train(X, y, use_cross_validation=False)
        
        # Save model
        model_path = tmp_path / "test_model.joblib"
        model.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        loaded_model = EdgeMLModel.load_model(str(model_path))
        
        assert loaded_model.model_type == model.model_type
        assert loaded_model.feature_names == model.feature_names
        
        # Test that loaded model can make predictions
        predictions = loaded_model.predict(X.head(3))
        assert len(predictions) == 3


class TestTrainingCallbacks:
    """Test training-callback helper behavior."""

    def test_get_fe_train_files_uses_latest_fe_run(self, tmp_path):
        """Only the latest FE dataset should be used for training."""
        from callbacks.training_callbacks import _get_fe_train_files

        training_dir = tmp_path / "training"
        training_dir.mkdir()

        old_name = "running_still_walking_downstairs_walking_upstairs"
        new_name = "running_still_walking"

        # Old FE run
        (training_dir / f"{old_name}_fe_metadata.json").write_text("{}")
        pd.DataFrame({"f1": [1.0], "label": ["running"]}).to_csv(
            training_dir / f"{old_name}_train.csv", index=False
        )

        # New FE run
        (training_dir / f"{new_name}_fe_metadata.json").write_text("{}")
        pd.DataFrame({"f1": [2.0], "label": ["walking"]}).to_csv(
            training_dir / f"{new_name}_train.csv", index=False
        )

        # Ensure deterministic "latest" ordering by mtime
        old_meta = training_dir / f"{old_name}_fe_metadata.json"
        new_meta = training_dir / f"{new_name}_fe_metadata.json"
        os.utime(old_meta, (1, 1))
        os.utime(new_meta, (2, 2))

        train_files = _get_fe_train_files(str(training_dir))

        assert train_files == [str(training_dir / f"{new_name}_train.csv")]


class TestCallbacks:
    """Test Dash callback functions."""
    
    @patch('callbacks.data_callbacks.os.path.exists')
    @patch('callbacks.data_callbacks.open')
    def test_metadata_loading(self, mock_open, mock_exists):
        """Test metadata loading in callbacks."""
        # Mock file operations
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = '{"test_dataset.csv": {"path": "/test/path", "label": "walking"}}'
        
        # This would require more complex mocking of Dash callbacks
        # For now, just test that imports work
        from callbacks import data_callbacks, preprocessing_callbacks, training_callbacks
        
        assert data_callbacks is not None
        assert preprocessing_callbacks is not None
        assert training_callbacks is not None


class TestConfiguration:
    """Test configuration and setup."""
    
    def test_config_imports(self):
        """Test that configuration imports work correctly."""
        from config.config import ROOT_DIR, PERSISTENT_DIR, METADATA_FILE
        
        assert ROOT_DIR is not None
        assert PERSISTENT_DIR is not None
        assert METADATA_FILE is not None

    def test_relative_working_dir_resolution(self):
        """Relative working dirs are resolved from the application root."""
        from config.config import ROOT_DIR, resolve_working_dir

        assert resolve_working_dir("portable_data") == os.path.abspath(
            os.path.join(ROOT_DIR, "portable_data"))

    def test_relative_metadata_path_resolution(self):
        """Relative metadata paths are resolved from the active working dir."""
        from config.config import ROOT_DIR, resolve_metadata_path

        resolved = resolve_metadata_path(
            os.path.join("datasets", "sample.csv"), "portable_data")
        expected = os.path.abspath(
            os.path.join(ROOT_DIR, "portable_data", "datasets", "sample.csv"))
        assert resolved == expected
    
    def test_directory_structure(self):
        """Test that required directories exist or can be created."""
        # This would be more complex in a real test environment
        # For now, just check that paths are defined
        metadata_dir = os.path.dirname(METADATA_FILE)
        assert os.path.isdir(metadata_dir) or not metadata_dir, (
            f"METADATA_FILE directory does not exist: {metadata_dir}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
