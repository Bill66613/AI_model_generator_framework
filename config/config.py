import os
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve(strict=True).parent.parent
CONF_PATH = os.path.dirname(os.path.abspath(__file__))

PERSISTENT_DIR = os.path.join(ROOT_DIR, "persistent_data")
if not os.path.exists(PERSISTENT_DIR):
    os.makedirs(PERSISTENT_DIR)
    
METADATA_FILE = os.path.join(PERSISTENT_DIR, "metadata.json")
if not os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, 'w') as f:
        json.dump({}, f)