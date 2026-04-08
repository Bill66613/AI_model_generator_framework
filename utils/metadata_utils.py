"""
Metadata loading utilities for the HAR Edge Deployment Framework.

Provides shared functions for loading and saving metadata.json,
avoiding code duplication across callback modules.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

from config.config import PERSISTENT_DIR

logger = logging.getLogger(__name__)


def get_metadata_path(base_dir: Optional[str] = None) -> str:
    """Return the path to metadata.json for the given base directory.

    Args:
        base_dir: Base persistent-data directory. Defaults to PERSISTENT_DIR.

    Returns:
        Absolute path to metadata.json.
    """
    if not base_dir:
        base_dir = PERSISTENT_DIR
    return os.path.join(base_dir, 'metadata.json')


def load_metadata(base_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load metadata.json and return its contents as a dict.

    Returns an empty dict if the file does not exist or is invalid JSON.

    Args:
        base_dir: Base persistent-data directory. Defaults to PERSISTENT_DIR.

    Returns:
        Parsed metadata dictionary.
    """
    metadata_file = get_metadata_path(base_dir)
    if not os.path.exists(metadata_file):
        return {}
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
        return {}


def save_metadata(metadata: Dict[str, Any], base_dir: Optional[str] = None) -> None:
    """Save metadata dict to metadata.json.

    Args:
        metadata: The metadata dictionary to persist.
        base_dir: Base persistent-data directory. Defaults to PERSISTENT_DIR.
    """
    metadata_file = get_metadata_path(base_dir)
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
