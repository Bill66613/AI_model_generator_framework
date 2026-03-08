"""
Data augmentation utilities for IMU-based Human Activity Recognition.

Provides sensor-signal augmentation methods that generate synthetic training
windows from original sensor data. These methods operate on raw sensor windows
(DataFrames with columns aX, aY, aZ, gX, gY, gZ) before feature extraction.

Augmentation is applied ONLY to training data, between edge-padding and
feature extraction in the Feature Engineering pipeline.

Reference:
  T. T. Um et al., "Data Augmentation of Wearable Sensor Data for Parkinson's
  Disease Monitoring using Convolutional Neural Networks," ICMI 2017.
  
  A. I. Kavathekar et al., "Exploring data augmentation for HAR with
  inertial sensors," Sensors 2022.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
#  Individual augmentation functions
# ---------------------------------------------------------------------------

def jitter(window: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """Add Gaussian noise to sensor readings.

    Simulates sensor noise variation and slight signal differences between
    trials of the same activity.

    Args:
        window: (T, 6) array [aX, aY, aZ, gX, gY, gZ]
        sigma: Standard deviation of additive noise, relative to each
               column's standard deviation. Default 0.05 (5% of signal std).

    Returns:
        Augmented (T, 6) array.
    """
    col_std = np.std(window, axis=0, keepdims=True)
    col_std = np.where(col_std < 1e-6, 1.0, col_std)  # avoid zero
    noise = np.random.randn(*window.shape) * sigma * col_std
    return window + noise


def scaling(window: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """Scale signal magnitude randomly per axis.

    Simulates different movement intensities (e.g., walking slowly vs
    briskly) and slight inter-person variability.

    Args:
        window: (T, 6) array.
        sigma: Standard deviation for scaling factor (drawn from
               N(1, sigma)). Default 0.1 means ±10% variation.

    Returns:
        Augmented (T, 6) array.
    """
    factor = np.random.normal(loc=1.0, scale=sigma, size=(1, window.shape[1]))
    return window * factor


def rotation(window: np.ndarray, angle_range: float = 20.0) -> np.ndarray:
    """Apply random 3D rotation to accelerometer and gyroscope triplets.

    This is the MOST IMPORTANT augmentation for orientation-invariant models.
    It simulates different device mounting angles — the most common source of
    train-deploy distribution shift for wrist/pocket-worn sensors.

    Args:
        window: (T, 6) array [aX, aY, aZ, gX, gY, gZ].
        angle_range: Max rotation angle in degrees per axis.
                     Default 20° (reasonable for wrist rotation).

    Returns:
        Augmented (T, 6) array with both acc & gyro rotated by the same
        random rotation matrix.
    """
    # Random angles (degrees → radians)
    angles = np.deg2rad(np.random.uniform(-angle_range, angle_range, size=3))
    ax, ay, az = angles

    # Rotation matrices around each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx  # combined rotation

    acc = window[:, :3] @ R.T   # rotate accelerometer
    gyro = window[:, 3:] @ R.T  # rotate gyroscope (same R)
    return np.hstack([acc, gyro])


def time_warp(window: np.ndarray, sigma: float = 0.2,
              num_knots: int = 4) -> np.ndarray:
    """Warp the time axis with a smooth random curve.

    Simulates speed variation within a single activity window — e.g., a
    person speeds up or slows down mid-step.

    Args:
        window: (T, 6) array.
        sigma: Standard deviation of warping at knot points.
        num_knots: Number of internal knot points (more = more complex warp).

    Returns:
        Augmented (T, 6) array (same length).
    """
    T = window.shape[0]
    # Generate a smooth warp path via cubic interpolation
    orig_steps = np.arange(T)
    knot_positions = np.linspace(0, T - 1, num_knots + 2)
    warp_values = np.random.normal(loc=1.0, scale=sigma, size=num_knots + 2)
    warp_values[0] = 1.0   # anchor start
    warp_values[-1] = 1.0  # anchor end

    # Cumulative warp function
    cumulative = np.cumsum(np.interp(orig_steps, knot_positions, warp_values))
    # Normalize to [0, T-1]
    cumulative = (cumulative - cumulative[0])
    cumulative = cumulative / cumulative[-1] * (T - 1)

    # Resample via linear interpolation for each channel
    result = np.zeros_like(window)
    for c in range(window.shape[1]):
        result[:, c] = np.interp(orig_steps, cumulative, window[:, c])
    return result


def permutation(window: np.ndarray, num_segments: int = 4) -> np.ndarray:
    """Randomly permute temporal segments.

    Encourages the model to learn features that are invariant to the order
    of sub-movements within a window. Effective for activities like walking
    where each step is roughly interchangeable.

    Args:
        window: (T, 6) array.
        num_segments: Number of equal-length segments to shuffle.

    Returns:
        Augmented (T, 6) array.
    """
    T = window.shape[0]
    segment_len = T // num_segments
    segments = []
    for i in range(num_segments):
        start = i * segment_len
        end = start + segment_len if i < num_segments - 1 else T
        segments.append(window[start:end])

    np.random.shuffle(segments)
    return np.vstack(segments)


# ---------------------------------------------------------------------------
#  Registry of available methods
# ---------------------------------------------------------------------------

AUGMENTATION_METHODS: Dict[str, dict] = {
    'jitter': {
        'fn': jitter,
        'label': '📊 Jittering (Gaussian Noise)',
        'description': (
            'Adds small random noise to simulate sensor noise variation. '
            'Low-risk, always helpful. Recommended for all models.'
        ),
        'params': {'sigma': 0.05},
    },
    'scaling': {
        'fn': scaling,
        'label': '📏 Scaling (Amplitude Variation)',
        'description': (
            'Randomly scales signal amplitude ±10% to simulate different '
            'movement intensities (e.g., brisk vs slow walking).'
        ),
        'params': {'sigma': 0.1},
    },
    'rotation': {
        'fn': rotation,
        'label': '🔄 Rotation (Device Orientation)',
        'description': (
            'Applies random 3D rotation (±20°) to acc & gyro triplets. '
            'Simulates different device mounting angles. '
            'MOST IMPORTANT for real-world robustness.'
        ),
        'params': {'angle_range': 20.0},
    },
    'time_warp': {
        'fn': time_warp,
        'label': '⏱️ Time Warping (Speed Variation)',
        'description': (
            'Warps the time axis smoothly to simulate speed changes '
            'within an activity (e.g., speeding up mid-stride).'
        ),
        'params': {'sigma': 0.2, 'num_knots': 4},
    },
    'permutation': {
        'fn': permutation,
        'label': '🔀 Permutation (Segment Shuffle)',
        'description': (
            'Splits the window into segments and shuffles their order. '
            'Helps models learn order-invariant features. '
            'Best for repetitive activities (walking, running).'
        ),
        'params': {'num_segments': 4},
    },
}


# ---------------------------------------------------------------------------
#  Main augmentation pipeline
# ---------------------------------------------------------------------------

def augment_windows(
    windows: List[pd.DataFrame],
    labels: List[str],
    methods: List[str],
    augmentation_factor: int = 2,
    sensor_cols: Optional[List[str]] = None,
    random_seed: Optional[int] = None,
    static_labels: Optional[List[str]] = None,
) -> Tuple[List[pd.DataFrame], List[str], dict]:
    """Apply selected augmentation methods to a list of sensor windows.

    For each original window, generates ``augmentation_factor`` synthetic
    copies by applying the selected methods in sequence (pipeline style).

    **Class-aware behaviour**: Windows whose label is in *static_labels*
    (e.g. ``still``, ``standing``, ``sitting``) are treated specially.
    These activities are defined by the *absence* of motion, so adding
    jitter / scaling / rotation makes them resemble low-intensity movement
    classes and hurts performance.  Static windows receive only very gentle
    jitter (σ=0.01 — just sensor-readout noise) and no other transforms.

    Args:
        windows: List of DataFrames, each (T, >=6) with sensor columns.
        labels: Corresponding activity labels, same length as *windows*.
        methods: List of method keys from AUGMENTATION_METHODS to apply.
        augmentation_factor: Number of synthetic copies per original window.
                             Total windows = original + factor × original.
        sensor_cols: Column names for 6 sensor axes. Defaults to
                     ['aX','aY','aZ','gX','gY','gZ'].
        random_seed: Seed for reproducibility. None = non-deterministic.
        static_labels: Activity labels that should be treated as static /
                       stationary.  These receive only micro-jitter (σ=0.01).
                       Default heuristic: labels containing 'still', 'stand',
                       'sit', 'lying', 'idle' are auto-detected.

    Returns:
        (augmented_windows, augmented_labels, stats)
        where augmented lists contain ONLY the new synthetic windows
        (caller should extend the originals with these).
        stats is a dict with counts and method details.
    """
    if sensor_cols is None:
        sensor_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']

    if random_seed is not None:
        np.random.seed(random_seed)

    # --- Static-label detection ---
    # If caller didn't specify, auto-detect from common keywords.
    _STATIC_KEYWORDS = ('still', 'stand', 'sit', 'lying', 'idle', 'static')
    if static_labels is None:
        # Auto-detect from the unique labels present
        unique_labels = set(labels)
        static_labels = [
            lbl for lbl in unique_labels
            if any(kw in lbl.lower() for kw in _STATIC_KEYWORDS)
        ]
    static_set = set(static_labels)

    # Resolve method functions
    active_methods = []
    for m in methods:
        if m in AUGMENTATION_METHODS:
            info = AUGMENTATION_METHODS[m]
            active_methods.append((m, info['fn'], info['params']))

    if not active_methods:
        return [], [], {'generated': 0, 'methods': [], 'factor': 0,
                        'static_labels': list(static_set)}

    aug_windows: List[pd.DataFrame] = []
    aug_labels: List[str] = []
    static_skipped = 0

    for df_window, label in zip(windows, labels):
        # Extract sensor data as numpy array
        cols_present = [c for c in sensor_cols if c in df_window.columns]
        if len(cols_present) < 6:
            continue

        arr = df_window[cols_present].values.astype(np.float64)
        is_static = label in static_set

        for _ in range(augmentation_factor):
            if is_static:
                # Static activities: only micro-jitter (sensor readout noise).
                # Full jitter/scaling/rotation would make "still" look like
                # low-intensity movement and cause class confusion.
                augmented = jitter(arr.copy(), sigma=0.01)
                static_skipped += 1
            else:
                augmented = arr.copy()
                # Apply each selected method in sequence (pipeline)
                for _name, fn, params in active_methods:
                    augmented = fn(augmented, **params)

            # Rebuild DataFrame preserving column structure
            df_aug = df_window.copy()
            df_aug[cols_present] = augmented
            aug_windows.append(df_aug)
            aug_labels.append(label)

    stats = {
        'original_count': len(windows),
        'generated': len(aug_windows),
        'total': len(windows) + len(aug_windows),
        'factor': augmentation_factor,
        'methods': [m for m, _, _ in active_methods],
        'static_labels': list(static_set),
        'static_micro_jitter_only': static_skipped,
    }
    return aug_windows, aug_labels, stats
