import os
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve(strict=True).parent.parent
CONF_PATH = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Sensor configuration
# ---------------------------------------------------------------------------
# Default sensor axis groups.  The framework is **not** limited to 6 axes —
# any subset or superset of these can be used.  Datasets that contain extra
# columns (e.g. magnetometer mX/mY/mZ, barometer, temperature) will be
# handled automatically as long as they are listed here.
#
# To customise: edit the lists below *or* override at runtime via the
# ``SENSOR_COLUMNS`` environment variable (comma-separated, e.g.
# ``SENSOR_COLUMNS=aX,aY,aZ``).

DEFAULT_ACCEL_COLUMNS = ['aX', 'aY', 'aZ']
DEFAULT_GYRO_COLUMNS  = ['gX', 'gY', 'gZ']
DEFAULT_SENSOR_COLUMNS = DEFAULT_ACCEL_COLUMNS + DEFAULT_GYRO_COLUMNS

# Allow runtime override via environment variable
_env_cols = os.environ.get('SENSOR_COLUMNS')
SENSOR_COLUMNS = [c.strip() for c in _env_cols.split(',')] if _env_cols else DEFAULT_SENSOR_COLUMNS
ACCEL_COLUMNS  = [c for c in SENSOR_COLUMNS if c.startswith('a')]
GYRO_COLUMNS   = [c for c in SENSOR_COLUMNS if c.startswith('g')]
# Any columns that are neither accel nor gyro (e.g. magnetometer, barometer)
EXTRA_COLUMNS  = [c for c in SENSOR_COLUMNS if c not in ACCEL_COLUMNS + GYRO_COLUMNS]

# Default sampling parameters
DEFAULT_SAMPLING_RATE = 100  # Hz — used as fallback when metadata is absent


def get_sensor_columns_from_metadata(metadata: dict, dataset_name: str) -> list:
    """Return sensor columns for a specific dataset, with fallback to global default.

    If the dataset's metadata contains a 'sensor_columns' key, use that.
    Otherwise fall back to the global ``SENSOR_COLUMNS`` constant.
    """
    if dataset_name in metadata:
        return metadata[dataset_name].get('sensor_columns', SENSOR_COLUMNS)
    return SENSOR_COLUMNS


def get_sampling_rate_from_metadata(metadata: dict, dataset_name: str) -> float:
    """Return sampling rate for a specific dataset, with fallback to default.

    Looks up ``metadata[dataset_name]['sampling_rate']``; if missing returns
    ``DEFAULT_SAMPLING_RATE``.
    """
    if dataset_name in metadata:
        return metadata[dataset_name].get('sampling_rate', DEFAULT_SAMPLING_RATE)
    return DEFAULT_SAMPLING_RATE


# Base persistent data directory
PERSISTENT_DIR = os.path.join(ROOT_DIR, "persistent_data")
if not os.path.exists(PERSISTENT_DIR):
    os.makedirs(PERSISTENT_DIR)

# New organized subdirectories
DATASETS_DIR = os.path.join(PERSISTENT_DIR, "datasets")
WINDOWS_DIR = os.path.join(PERSISTENT_DIR, "windows")
TRAINING_DIR = os.path.join(PERSISTENT_DIR, "training")
MODELS_DIR = os.path.join(PERSISTENT_DIR, "models")

# Create subdirectories if they don't exist
for directory in [DATASETS_DIR, WINDOWS_DIR, TRAINING_DIR, MODELS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Metadata file remains at root of persistent_data for backward compatibility
METADATA_FILE = os.path.join(PERSISTENT_DIR, "metadata.json")
if not os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, 'w') as f:
        json.dump({}, f)

# Helper functions for path management


def get_dataset_path(dataset_name, is_cleaned=False):
    """Get path for raw or cleaned dataset file.

    Args:
        dataset_name: Name of the dataset file (e.g., 'walking_1.csv')
        is_cleaned: If True, returns path for cleaned version

    Returns:
        Full path to the dataset file
    """
    if is_cleaned:
        # Cleaned files use 'cleaned_smoothed_' prefix
        filename = f"cleaned_smoothed_{dataset_name}" if not dataset_name.startswith(
            'cleaned_') else dataset_name
        return os.path.join(DATASETS_DIR, filename)
    else:
        # Raw files - check both old location (persistent_data root) and new location (datasets/)
        new_path = os.path.join(DATASETS_DIR, dataset_name)
        old_path = os.path.join(PERSISTENT_DIR, dataset_name)

        # If file exists in old location but not new, return old path for backward compatibility
        if os.path.exists(old_path) and not os.path.exists(new_path):
            return old_path
        return new_path


def get_window_path(window_id, dataset_name):
    """Get path for a dragged window file.

    Args:
        window_id: ID of the window (e.g., 0, 1, 2, ...)
        dataset_name: Name of the parent dataset (e.g., 'walking_1.csv')

    Returns:
        Full path to the window file
    """
    filename = f"dragged_window_{window_id}_{dataset_name}"
    new_path = os.path.join(WINDOWS_DIR, filename)
    old_path = os.path.join(PERSISTENT_DIR, filename)

    # Backward compatibility: check old location first
    if os.path.exists(old_path) and not os.path.exists(new_path):
        return old_path
    return new_path


def get_window_pattern(dataset_name):
    """Get glob pattern for finding all windows of a dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'walking_1.csv')

    Returns:
        Glob pattern string
    """
    return os.path.join(WINDOWS_DIR, f"dragged_window_*_{dataset_name}")


def get_training_data_path(dataset_name, split_type, base_dir=None):
    """Get path for training/validation/test split file.

    Args:
        dataset_name: Name of the dataset (e.g., 'walking_1.csv')
        split_type: One of 'train', 'val', 'test', or 'metadata'
        base_dir: Optional base directory (defaults to PERSISTENT_DIR)

    Returns:
        Full path to the split file
    """
    if not base_dir:
        base_dir = PERSISTENT_DIR

    if split_type == 'metadata':
        filename = f"{dataset_name}_metadata.json"
    else:
        filename = f"{dataset_name}_{split_type}.csv"

    training_dir = os.path.join(base_dir, 'training')
    new_path = os.path.join(training_dir, filename)
    old_path = os.path.join(PERSISTENT_DIR, 'training_data', filename)

    # Backward compatibility: check old location first
    if os.path.exists(old_path) and not os.path.exists(new_path):
        return old_path
    return new_path


def get_model_path(model_filename, base_dir=None):
    """Get path for a trained model file.

    Args:
        model_filename: Name of the model file (e.g., 'neural_network_har_model_20241212.joblib')
        base_dir: Optional base directory (defaults to PERSISTENT_DIR)

    Returns:
        Full path to the model file
    """
    if not base_dir:
        base_dir = PERSISTENT_DIR

    models_dir = os.path.join(base_dir, 'models')
    new_path = os.path.join(models_dir, model_filename)
    old_path = os.path.join(PERSISTENT_DIR, model_filename)

    # Backward compatibility: check old location first
    if os.path.exists(old_path) and not os.path.exists(new_path):
        return old_path
    return new_path


def get_models_metadata_path(base_dir=None):
    """Get path for the trained models metadata JSON file.

    Args:
        base_dir: Optional base directory (defaults to PERSISTENT_DIR)

    Returns:
        Full path to the models metadata file
    """
    if not base_dir:
        base_dir = PERSISTENT_DIR

    models_dir = os.path.join(base_dir, 'models')
    new_path = os.path.join(models_dir, "trained_models.json")
    old_path = os.path.join(PERSISTENT_DIR, "trained_models.json")

    # Backward compatibility: check old location first
    if os.path.exists(old_path) and not os.path.exists(new_path):
        return old_path
    return new_path
