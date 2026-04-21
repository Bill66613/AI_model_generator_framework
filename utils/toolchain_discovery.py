"""
Toolchain discovery for Arduino CLI and PlatformIO.

Auto-detects tool paths even when not on system PATH by searching
common installation locations. Supports environment variable overrides.
"""

import glob
import json
import os
import platform
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# TTL cache (60 s) to avoid repeated subprocess calls within a session
# ---------------------------------------------------------------------------
_cache: Dict[str, Tuple[float, object]] = {}
_CACHE_TTL = 60  # seconds


def _cache_get(key: str):
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < _CACHE_TTL:
        return entry[1]
    return None


def _cache_set(key: str, value):
    _cache[key] = (time.time(), value)


def clear_cache():
    """Clear the discovery cache (e.g. after installing a core)."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Tool path finders
# ---------------------------------------------------------------------------

def find_arduino_cli() -> Optional[str]:
    """Find the Arduino CLI executable, searching common install locations."""
    # 1. Environment variable override
    env_path = os.environ.get('ARDUINO_CLI_PATH')
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Already on PATH
    which = shutil.which('arduino-cli')
    if which:
        return which

    # 3. Common Windows install locations
    if platform.system() == 'Windows':
        home = os.path.expanduser('~')
        search_patterns = [
            os.path.join(home, 'OneDrive', 'Documents', 'arduino-cli_*', 'arduino-cli.exe'),
            os.path.join(home, 'Documents', 'arduino-cli_*', 'arduino-cli.exe'),
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'arduino-cli', 'arduino-cli.exe'),
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Arduino15', 'arduino-cli.exe'),
            os.path.join('C:\\', 'Program Files', 'Arduino CLI', 'arduino-cli.exe'),
        ]
        for pattern in search_patterns:
            matches = glob.glob(pattern)
            if matches:
                # Return the most recent version (sorted descending)
                matches.sort(reverse=True)
                return matches[0]
    else:
        # Linux/macOS
        candidates = [
            os.path.expanduser('~/bin/arduino-cli'),
            '/usr/local/bin/arduino-cli',
            '/usr/bin/arduino-cli',
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

    return None


def find_platformio_cli() -> Optional[str]:
    """Find the PlatformIO CLI executable, searching common install locations."""
    # 1. Environment variable override
    env_path = os.environ.get('PLATFORMIO_CLI_PATH')
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Already on PATH
    which = shutil.which('pio')
    if which:
        return which

    # 3. Common install locations
    home = os.path.expanduser('~')
    if platform.system() == 'Windows':
        candidates = [
            os.path.join(home, '.platformio', 'penv', 'Scripts', 'pio.exe'),
            os.path.join(home, '.platformio', 'penv', 'Scripts', 'platformio.exe'),
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'PlatformIO', 'pio.exe'),
        ]
    else:
        candidates = [
            os.path.join(home, '.platformio', 'penv', 'bin', 'pio'),
            '/usr/local/bin/pio',
            '/usr/bin/pio',
        ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    return None


# ---------------------------------------------------------------------------
# Version and core/platform checking
# ---------------------------------------------------------------------------

def get_tool_version(tool_path: str, version_args: Optional[List[str]] = None) -> Optional[str]:
    """Run <tool> <version_args> and return the first line of stdout."""
    if version_args is None:
        version_args = ['--version']

    cached = _cache_get(f'version:{tool_path}')
    if cached is not None:
        return cached

    try:
        result = subprocess.run(
            [tool_path] + version_args,
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            version = result.stdout.strip().split('\n')[0]
            _cache_set(f'version:{tool_path}', version)
            return version
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def list_arduino_cores(cli_path: str) -> List[dict]:
    """List installed Arduino cores via `arduino-cli core list --format json`."""
    cached = _cache_get(f'arduino_cores:{cli_path}')
    if cached is not None:
        return cached

    try:
        result = subprocess.run(
            [cli_path, 'core', 'list', '--format', 'json'],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0 and result.stdout.strip():
            cores = json.loads(result.stdout)
            # Handle both list format and dict-with-platforms format
            if isinstance(cores, dict):
                cores = cores.get('platforms', [])
            _cache_set(f'arduino_cores:{cli_path}', cores)
            return cores
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return []


def check_arduino_core_for_board(cli_path: str, fqbn: str) -> dict:
    """
    Check if the required Arduino core for a board FQBN is installed.

    For FQBN like 'esp32:esp32:esp32s3', the core prefix is 'esp32:esp32'.
    For 'm5stack:esp32:m5stick_c', the core prefix is 'm5stack:esp32'.
    """
    parts = fqbn.split(':')
    if len(parts) >= 2:
        core_id = f"{parts[0]}:{parts[1]}"
    else:
        core_id = fqbn

    cores = list_arduino_cores(cli_path)
    installed = False
    installed_version = None

    for core in cores:
        cid = core.get('id', core.get('ID', ''))
        if cid == core_id:
            installed = True
            installed_version = core.get('installed', core.get('installed_version', 'unknown'))
            break

    return {
        'installed': installed,
        'core_id': core_id,
        'installed_version': installed_version,
        'install_command': f'arduino-cli core install {core_id}',
    }


def list_platformio_platforms(cli_path: str) -> List[str]:
    """List installed PlatformIO platforms."""
    cached = _cache_get(f'pio_platforms:{cli_path}')
    if cached is not None:
        return cached

    try:
        result = subprocess.run(
            [cli_path, 'platform', 'list', '--json-output'],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0 and result.stdout.strip():
            platforms_data = json.loads(result.stdout)
            if isinstance(platforms_data, list):
                names = [p.get('name', p.get('title', '')) for p in platforms_data]
            else:
                names = []
            _cache_set(f'pio_platforms:{cli_path}', names)
            return names
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return []


# Board FQBN to PlatformIO platform mapping (mirrors code_generation_callbacks.py)
_FQBN_TO_PIO_PLATFORM = {
    'm5stack:esp32:m5stick_c': 'espressif32',
    'esp32:esp32:esp32': 'espressif32',
    'esp32:esp32:esp32s3': 'espressif32',
    'esp32:esp32:esp32c3': 'espressif32',
    'arduino:avr:uno': 'atmelavr',
    'arduino:avr:nano': 'atmelavr',
    'STM32:stm32:GenF4': 'ststm32',
    'Seeeduino:nrf52:xiaonRF52840Sense': 'nordicnrf52',
}


def check_pio_platform_for_board(cli_path: str, fqbn: str) -> dict:
    """Check if the PlatformIO platform for a board FQBN is installed."""
    required_platform = _FQBN_TO_PIO_PLATFORM.get(fqbn, 'unknown')

    platforms = list_platformio_platforms(cli_path)
    # Platform names from `pio platform list` may be display names,
    # so check case-insensitively
    installed = any(required_platform.lower() in p.lower() for p in platforms)

    return {
        'installed': installed,
        'platform': required_platform,
        'install_command': f'pio platform install {required_platform}',
    }


def make_tool_env(tool_path: str) -> dict:
    """
    Create an environment dict that adds the tool's parent directory to PATH.

    This ensures child processes (e.g. esptool called by arduino-cli) can be found.
    """
    env = os.environ.copy()
    tool_dir = os.path.dirname(os.path.abspath(tool_path))
    env['PATH'] = tool_dir + os.pathsep + env.get('PATH', '')
    return env
