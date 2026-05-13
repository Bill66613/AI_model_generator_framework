"""
Training Pipeline Utilities for Human Activity Recognition

Provides orchestration functions for preparing training data from window
files and creating model instances.
"""

import os
import pandas as pd
from typing import Tuple, List
import logging

from config.config import SENSOR_COLUMNS
from utils.feature_extraction import create_feature_vector
from utils.edge_ml_model import EdgeMLModel

logger = logging.getLogger(__name__)


def prepare_training_data(
    window_files: List[str],
    labels: List[str],
    sensor_cols: List[str] = None,
    sampling_rate: float = 100,
    include_frequency: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare training data from window files.

    Loads each window CSV, extracts feature vectors, and combines into
    a feature matrix (X) and label series (y).

    Args:
        window_files: List of file paths to window CSV files
        labels: List of activity labels corresponding to each window file
        sensor_cols: Sensor column names (default: SENSOR_COLUMNS from config)
        sampling_rate: Sampling rate in Hz (default: 100)
        include_frequency: Include FFT features (default: True)

    Returns:
        Tuple of (X: DataFrame of features, y: Series of labels)

    Raises:
        ValueError: If window_files and labels have different lengths or
                    no valid files are found
    """
    if len(window_files) != len(labels):
        raise ValueError("Number of window files must match number of labels")

    if not window_files:
        raise ValueError("window_files and labels cannot be empty")

    if sensor_cols is None:
        sensor_cols = list(SENSOR_COLUMNS)

    all_features = []
    all_labels = []

    for file_path, label in zip(window_files, labels):
        if not os.path.exists(file_path):
            logger.warning(f"Window file not found: {file_path}")
            continue

        window_df = pd.read_csv(file_path)
        features = create_feature_vector(
            window_df, sensor_cols, sampling_rate, include_frequency)

        all_features.append(features)
        all_labels.append(label)

    if all_features:
        X = pd.concat(all_features, ignore_index=True)
        y = pd.Series(all_labels)
        return X, y
    else:
        raise ValueError("No valid window files found")


def create_model(model_type: str, **kwargs) -> EdgeMLModel:
    """Factory function to create edge ML models.

    Args:
        model_type: One of 'random_forest', 'svm', 'neural_network',
                    'pytorch_mlp', 'pytorch_cnn', 'pytorch_cnn2d'
        **kwargs: Model-specific hyperparameters

    Returns:
        EdgeMLModel instance

    Raises:
        ValueError: If model_type is not supported
    """
    supported_models = [
        'random_forest', 'svm', 'neural_network',
        'pytorch_mlp', 'pytorch_cnn', 'pytorch_cnn2d',
    ]

    if model_type not in supported_models:
        raise ValueError(
            f"Model type '{model_type}' not supported. Choose from: {supported_models}")

    return EdgeMLModel(model_type, **kwargs)
