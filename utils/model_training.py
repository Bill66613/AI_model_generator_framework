"""
Model Training Utilities for Human Activity Recognition - Backward-Compatibility Shim

This module re-exports all public symbols from the refactored sub-modules
so that existing ``from utils.model_training import ...`` statements
continue to work without modification.

Canonical locations:
    - EdgeMLModel              -> utils.edge_ml_model
    - Feature extraction funcs -> utils.feature_extraction
    - prepare_training_data    -> utils.training_pipeline
    - create_model             -> utils.training_pipeline
"""

# Re-export EdgeMLModel
from utils.edge_ml_model import EdgeMLModel  # noqa: F401

# Re-export feature extraction functions
from utils.feature_extraction import (  # noqa: F401
    extract_orientation_invariant_features,
    extract_frequency_magnitude_features,
    extract_time_domain_features,
    extract_frequency_domain_features,
    create_feature_vector,
)

# Re-export pipeline utilities
from utils.training_pipeline import (  # noqa: F401
    prepare_training_data,
    create_model,
)

