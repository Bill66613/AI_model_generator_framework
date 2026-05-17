"""
Feature Extraction Utilities for Human Activity Recognition

Provides functions to extract time-domain, frequency-domain, and
orientation-invariant features from raw sensor data windows.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
import logging

from config.config import (
    SENSOR_COLUMNS, ACCEL_COLUMNS, GYRO_COLUMNS
)

logger = logging.getLogger(__name__)


def _fft_with_windowing(data: np.ndarray, sampling_rate: float
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FFT with proper DSP preprocessing (DC removal + Hann window).

    Follows Edge Impulse / standard DSP best practice:
    1. Remove DC component (subtract mean) — eliminates 0 Hz bin.
    2. Apply Hann window — reduces spectral leakage at bin boundaries.
    3. Compute FFT and return only positive-frequency magnitudes.

    Args:
        data: 1-D signal array.
        sampling_rate: Sampling rate in Hz.

    Returns:
        (fft_magnitude_pos, fft_freq_pos) — magnitudes and frequencies for
        positive-frequency bins only (excluding DC).
    """
    n = len(data)
    # Step 1: DC removal
    data_centered = data - np.mean(data)
    # Step 2: Hann window to reduce spectral leakage
    window = np.hanning(n)
    data_windowed = data_centered * window
    # Step 3: FFT (rFFT keeps non-negative frequencies and includes Nyquist)
    fft_vals = np.fft.rfft(data_windowed)
    fft_magnitude = np.abs(fft_vals)
    fft_freq = np.fft.rfftfreq(n, 1.0 / sampling_rate)
    # Exclude DC bin only (keep Nyquist for parity with C++/MicroPython)
    return fft_magnitude[1:], fft_freq[1:]


def _spectral_statistics(fft_magnitude_pos: np.ndarray
                         ) -> Tuple[float, float, float]:
    """Compute RMS, skewness, and kurtosis of FFT magnitude bins.

    These spectral-shape descriptors are used by Edge Impulse and capture
    how the spectral energy is distributed across frequency bins.

    Returns:
        (spectral_rms, spectral_skewness, spectral_kurtosis)
    """
    if len(fft_magnitude_pos) == 0:
        return 0.0, 0.0, 0.0
    nn = len(fft_magnitude_pos)
    sq_sum = np.sum(fft_magnitude_pos ** 2)
    rms = np.sqrt(sq_sum / nn)
    spec_mean = np.sum(fft_magnitude_pos) / nn
    spec_var = (sq_sum / nn) - (spec_mean * spec_mean)
    # Keep these constants aligned with deployment generators for parity.
    spec_std = np.sqrt(spec_var) if spec_var > 0 else 0.0001

    m3 = 0.0
    m4 = 0.0
    for mag in fft_magnitude_pos:
        z = (mag - spec_mean) / (spec_std + 1e-7)
        z2 = z * z
        m3 += z2 * z
        m4 += z2 * z2

    skewness = 0.0
    kurtosis = 0.0
    if nn > 2:
        skewness = (m3 / nn) * (nn * (nn + 1)) / ((nn - 1) * (nn - 2))
    if nn > 3:
        raw_kurt = m4 / nn
        kurtosis = ((nn + 1) * raw_kurt - 3.0 * (nn - 1)) * (nn - 1) / ((nn - 2) * (nn - 3))

    return float(rms), float(skewness), float(kurtosis)


def _compute_centered_magnitude(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """Compute Euclidean magnitude after per-window mean centering.

    Subtracts the per-window mean from each axis before computing magnitude,
    removing static offsets (gravity for accel, orientation-dependent bias for
    gyro) so the result captures only dynamic variation.

    Args:
        df: DataFrame with sensor data (one window)
        columns: Column names (e.g. ['aX','aY','aZ'])

    Returns:
        1-D numpy array of centered-magnitude values
    """
    present = [c for c in columns if c in df.columns]
    if not present:
        return np.zeros(len(df))
    centered = {c: df[c].values - df[c].values.mean() for c in present}
    return np.sqrt(sum(centered[c] ** 2 for c in present))


def extract_orientation_invariant_features(
    df: pd.DataFrame,
    sensor_cols: List[str] = None,
) -> pd.DataFrame:
    """Extract orientation-invariant features using magnitude vectors.

    These features are robust to device orientation changes, making the model
    work regardless of how the sensor is mounted (left/right wrist, rotated, etc.).

    Args:
        df: DataFrame with sensor data
        sensor_cols: Sensor column names (default: SENSOR_COLUMNS from config)

    Returns:
        DataFrame with orientation-invariant magnitude-based features
    """
    if sensor_cols is None:
        sensor_cols = list(SENSOR_COLUMNS)

    features = {}

    # Determine accelerometer and gyroscope columns from config
    accel_cols = [c for c in ACCEL_COLUMNS if c in df.columns]
    gyro_cols = [c for c in GYRO_COLUMNS if c in df.columns]

    # Calculate centered magnitude vectors — per-window mean subtraction
    # removes gravity (accel) and orientation-dependent bias (gyro),
    # so magnitudes capture only dynamic variation.
    acc_mag = _compute_centered_magnitude(df, accel_cols)
    gyro_mag = _compute_centered_magnitude(df, gyro_cols)

    # Statistical features on acceleration magnitude
    for name, mag_data in [('acc_mag', acc_mag), ('gyro_mag', gyro_mag)]:
        data = mag_data

        # Basic statistical features
        features[f'{name}_mean'] = np.mean(data)
        features[f'{name}_std'] = np.std(data)
        features[f'{name}_min'] = np.min(data)
        features[f'{name}_max'] = np.max(data)
        features[f'{name}_range'] = np.max(data) - np.min(data)
        features[f'{name}_median'] = np.median(data)
        features[f'{name}_q25'] = np.percentile(data, 25)
        features[f'{name}_q75'] = np.percentile(data, 75)
        features[f'{name}_iqr'] = np.percentile(
            data, 75) - np.percentile(data, 25)

        # Advanced statistical features
        features[f'{name}_skewness'] = pd.Series(data).skew()
        features[f'{name}_kurtosis'] = pd.Series(data).kurtosis()
        features[f'{name}_rms'] = np.sqrt(np.mean(data**2))
        features[f'{name}_energy'] = np.sum(data**2)

        # Signal characteristics - sign-product method (consistent with C++ deployment)
        mean_val = np.mean(data)
        # Count strict sign changes: data[i-1] * data[i] < 0
        features[f'{name}_zero_crossings'] = int(
            np.sum(data[:-1] * data[1:] < 0))
        centered = data - mean_val
        mean_crossings = int(np.sum(centered[:-1] * centered[1:] < 0))
        features[f'{name}_mean_crossing_rate'] = mean_crossings / len(data)

    # Jerk magnitude (rate of change of acceleration) - also orientation invariant
    features['acc_jerk_mag_mean'] = 0.0
    features['acc_jerk_mag_std'] = 0.0
    features['acc_jerk_mag_max'] = 0.0
    if len(accel_cols) >= 2:
        acc_jerk_mag = np.sqrt(
            sum(np.diff(df[c].values) ** 2 for c in accel_cols))
        features['acc_jerk_mag_mean'] = np.mean(acc_jerk_mag)
        features['acc_jerk_mag_std'] = np.std(acc_jerk_mag)
        features['acc_jerk_mag_max'] = np.max(acc_jerk_mag)

    # -------------------------------------------------------------------
    # Additional features for multi-location HAR improvement
    # -------------------------------------------------------------------

    # Gyro jerk magnitude (rate of change of angular velocity)
    # Captures rotational acceleration — wrist rotation snaps (blocking vs hooking),
    # ankle stance/swing phase transitions, trunk rotation in body wear.
    features['gyro_jerk_mag_mean'] = 0.0
    features['gyro_jerk_mag_std'] = 0.0
    features['gyro_jerk_mag_max'] = 0.0
    if len(gyro_cols) >= 2:
        gyro_jerk_mag = np.sqrt(
            sum(np.diff(df[c].values) ** 2 for c in gyro_cols))
        features['gyro_jerk_mag_mean'] = np.mean(gyro_jerk_mag)
        features['gyro_jerk_mag_std'] = np.std(gyro_jerk_mag)
        features['gyro_jerk_mag_max'] = np.max(gyro_jerk_mag)

    # Signal Magnitude Area (SMA): mean(|aX| + |aY| + |aZ|) using raw (uncentered) axes.
    # Widely used in HAR literature; discriminates sedentary vs active, especially
    # effective for body/back wear and ankle wear where total movement intensity matters.
    features['acc_sma'] = 0.0
    if len(accel_cols) >= 1:
        raw_accel_sum = sum(np.abs(df[c].values) for c in accel_cols)
        features['acc_sma'] = float(np.mean(raw_accel_sum))

    # Tilt angles from static (mean) accelerometer component.
    # Estimates device orientation/inclination relative to gravity vector.
    # Key for body/back wear (seated vs standing posture), ankle wear (foot angle),
    # and any scenario where device orientation is informative.
    features['tilt_pitch'] = 0.0
    features['tilt_roll'] = 0.0
    if len(accel_cols) >= 3:
        ax_m = df[accel_cols[0]].values.mean()
        ay_m = df[accel_cols[1]].values.mean()
        az_m = df[accel_cols[2]].values.mean()
        features['tilt_pitch'] = float(
            np.degrees(np.arctan2(ax_m, np.sqrt(ay_m ** 2 + az_m ** 2))))
        features['tilt_roll'] = float(np.degrees(np.arctan2(ay_m, az_m)))

    # Autocorrelation of acc_mag at lag 1 (periodicity and signal smoothness).
    # High value ≈ smooth/repetitive motion (walking, running);
    # Low/negative value ≈ erratic/impact motion (jumping, punching).
    features['acc_mag_autocorr_lag1'] = 0.0
    if len(acc_mag) > 1:
        ac_mean = np.mean(acc_mag)
        ac_var = np.var(acc_mag)
        if ac_var > 1e-10:
            features['acc_mag_autocorr_lag1'] = float(
                np.sum((acc_mag[:-1] - ac_mean) * (acc_mag[1:] - ac_mean))
                / (len(acc_mag) * ac_var)
            )
        else:
            features['acc_mag_autocorr_lag1'] = 1.0

    # Jerk peak count: number of local maxima above (mean + 0.5 × std) in acc_jerk_mag.
    # Captures gesture cadence (wrist wear: punch/block repetitions),
    # step count per window (ankle wear: walking/running cadence).
    features['acc_jerk_mag_peak_count'] = 0
    if len(accel_cols) >= 2:
        jerk_for_peaks = np.sqrt(
            sum(np.diff(df[c].values) ** 2 for c in accel_cols))
        jm = np.mean(jerk_for_peaks)
        js = np.std(jerk_for_peaks)
        threshold = jm + max(0.5 * js, 1e-4)
        features['acc_jerk_mag_peak_count'] = int(np.sum(
            (jerk_for_peaks[1:-1] > jerk_for_peaks[:-2]) &
            (jerk_for_peaks[1:-1] > jerk_for_peaks[2:]) &
            (jerk_for_peaks[1:-1] > threshold)
        ))

    return pd.DataFrame([features])


def extract_frequency_magnitude_features(
    df: pd.DataFrame,
    sensor_cols: List[str] = None,
    sampling_rate: float = 100,
) -> pd.DataFrame:
    """Extract FFT features from magnitude vectors (orientation invariant).

    Frequency components of magnitude vectors are independent of device orientation.
    Walking has the same step frequency regardless of which wrist or rotation.

    Args:
        df: DataFrame with sensor data
        sensor_cols: Sensor column names (default: SENSOR_COLUMNS from config)
        sampling_rate: Sampling rate in Hz

    Returns:
        DataFrame with frequency domain magnitude features
    """
    if sensor_cols is None:
        sensor_cols = list(SENSOR_COLUMNS)

    features = {}

    # Determine accelerometer and gyroscope columns from config
    accel_cols = [c for c in ACCEL_COLUMNS if c in df.columns]
    gyro_cols = [c for c in GYRO_COLUMNS if c in df.columns]

    # Calculate centered magnitude vectors (consistent with time-domain features)
    acc_mag = _compute_centered_magnitude(df, accel_cols)
    gyro_mag = _compute_centered_magnitude(df, gyro_cols)

    for name, mag_data in [('acc_mag', acc_mag), ('gyro_mag', gyro_mag)]:
        # Proper DSP pipeline: DC removal → Hann window → FFT
        fft_magnitude_pos, fft_freq_pos = _fft_with_windowing(
            mag_data, sampling_rate)

        if len(fft_magnitude_pos) == 0:
            continue

        # Dominant frequency (most important for activity classification)
        dominant_freq_idx = np.argmax(fft_magnitude_pos)
        features[f'{name}_dominant_frequency'] = fft_freq_pos[dominant_freq_idx]
        features[f'{name}_dominant_frequency_magnitude'] = fft_magnitude_pos[dominant_freq_idx]

        # Spectral centroid (center of mass of spectrum)
        total_mag = np.sum(fft_magnitude_pos)
        if total_mag > 0:
            features[f'{name}_spectral_centroid'] = np.sum(
                fft_freq_pos * fft_magnitude_pos) / total_mag
        else:
            features[f'{name}_spectral_centroid'] = 0.0

        # Energy in frequency bands
        low_freq_mask = (fft_freq_pos >= 0) & (fft_freq_pos < 2)
        mid_freq_mask = (fft_freq_pos >= 2) & (fft_freq_pos < 5)
        high_freq_mask = (fft_freq_pos >= 5) & (
            fft_freq_pos <= sampling_rate / 2)

        features[f'{name}_energy_low_freq'] = np.sum(
            fft_magnitude_pos[low_freq_mask] ** 2)
        features[f'{name}_energy_mid_freq'] = np.sum(
            fft_magnitude_pos[mid_freq_mask] ** 2)
        features[f'{name}_energy_high_freq'] = np.sum(
            fft_magnitude_pos[high_freq_mask] ** 2)

        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumsum = np.cumsum(fft_magnitude_pos)
        if cumsum[-1] > 0:
            rolloff_threshold = 0.85 * cumsum[-1]
            rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
            if len(rolloff_idx) > 0:
                features[f'{name}_spectral_rolloff'] = fft_freq_pos[rolloff_idx[0]]
            else:
                features[f'{name}_spectral_rolloff'] = fft_freq_pos[-1]
        else:
            features[f'{name}_spectral_rolloff'] = 0.0

        # Spectral shape descriptors (EI-style: RMS, skewness, kurtosis of bins)
        spec_rms, spec_skew, spec_kurt = _spectral_statistics(fft_magnitude_pos)
        features[f'{name}_spectral_rms'] = spec_rms
        features[f'{name}_spectral_skewness'] = spec_skew
        features[f'{name}_spectral_kurtosis'] = spec_kurt

        # Power spectral entropy (spread of spectral energy across frequency bins).
        # Low entropy = energy concentrated at dominant frequency (rhythmic activity);
        # High entropy = broadband energy spread (erratic/complex motion).
        total = np.sum(fft_magnitude_pos)
        if total > 0:
            probs = fft_magnitude_pos / total
            probs_nz = probs[probs > 1e-12]
            features[f'{name}_spectral_entropy'] = float(-np.sum(probs_nz * np.log2(probs_nz)))
        else:
            features[f'{name}_spectral_entropy'] = 0.0

    return pd.DataFrame([features])


def extract_time_domain_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
) -> pd.DataFrame:
    """Extract time-domain statistical features from sensor data.

    Args:
        df: DataFrame with sensor data
        sensor_cols: List of column names to extract features from

    Returns:
        DataFrame with 15 features per column (mean, std, min, max, range,
        median, q25, q75, iqr, skewness, kurtosis, rms, energy,
        zero_crossings, mean_crossing_rate)
    """
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
        features[f'{col}_iqr'] = np.percentile(
            data, 75) - np.percentile(data, 25)

        # Advanced statistical features
        features[f'{col}_skewness'] = pd.Series(data).skew()
        features[f'{col}_kurtosis'] = pd.Series(data).kurtosis()
        features[f'{col}_rms'] = np.sqrt(np.mean(data**2))
        features[f'{col}_energy'] = np.sum(data**2)

        # Signal characteristics - sign-product method (consistent with C++ deployment)
        features[f'{col}_zero_crossings'] = int(
            np.sum(data[:-1] * data[1:] < 0))
        centered = data - np.mean(data)
        mean_crossings = int(np.sum(centered[:-1] * centered[1:] < 0))
        features[f'{col}_mean_crossing_rate'] = mean_crossings / len(data)

    return pd.DataFrame([features])


def extract_frequency_domain_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    sampling_rate: float = 100,
) -> pd.DataFrame:
    """Extract frequency-domain features using FFT with proper DSP preprocessing.

    Pipeline: DC removal → Hann window → FFT → feature extraction.

    Args:
        df: DataFrame with sensor data
        sensor_cols: List of column names to extract features from
        sampling_rate: Sampling rate in Hz

    Returns:
        DataFrame with 11 features per column (spectral_centroid,
        spectral_rolloff, spectral_bandwidth, dominant_frequency,
        dominant_frequency_magnitude, energy_low/mid/high_freq,
        spectral_rms, spectral_skewness, spectral_kurtosis)
    """
    features = {}

    for col in sensor_cols:
        if col not in df.columns:
            continue

        data = df[col].values

        # Proper DSP pipeline: DC removal → Hann window → FFT
        fft_magnitude_pos, fft_freq_pos = _fft_with_windowing(
            data, sampling_rate)

        if len(fft_magnitude_pos) == 0:
            continue

        total_mag = np.sum(fft_magnitude_pos)
        if total_mag == 0:
            total_mag = 1e-10  # Avoid division by zero

        # Frequency domain features
        spectral_centroid = np.sum(
            fft_freq_pos * fft_magnitude_pos) / total_mag
        features[f'{col}_spectral_centroid'] = spectral_centroid

        cumsum = np.cumsum(fft_magnitude_pos)
        rolloff_idx = np.where(cumsum >= 0.85 * np.sum(fft_magnitude_pos))[0]
        features[f'{col}_spectral_rolloff'] = (
            fft_freq_pos[rolloff_idx[0]] if len(rolloff_idx) > 0
            else fft_freq_pos[-1]
        )

        features[f'{col}_spectral_bandwidth'] = np.sqrt(
            np.sum(((fft_freq_pos - spectral_centroid) ** 2) * fft_magnitude_pos)
            / total_mag
        )

        # Dominant frequency
        dominant_freq_idx = np.argmax(fft_magnitude_pos)
        features[f'{col}_dominant_frequency'] = fft_freq_pos[dominant_freq_idx]
        features[f'{col}_dominant_frequency_magnitude'] = fft_magnitude_pos[dominant_freq_idx]

        # Energy in frequency bands
        low_freq_mask = (fft_freq_pos >= 0) & (fft_freq_pos < 5)
        mid_freq_mask = (fft_freq_pos >= 5) & (fft_freq_pos < 15)
        high_freq_mask = (fft_freq_pos >= 15) & (
            fft_freq_pos <= sampling_rate / 2)

        features[f'{col}_energy_low_freq'] = np.sum(
            fft_magnitude_pos[low_freq_mask] ** 2)
        features[f'{col}_energy_mid_freq'] = np.sum(
            fft_magnitude_pos[mid_freq_mask] ** 2)
        features[f'{col}_energy_high_freq'] = np.sum(
            fft_magnitude_pos[high_freq_mask] ** 2)

        # Spectral shape descriptors (EI-style)
        spec_rms, spec_skew, spec_kurt = _spectral_statistics(fft_magnitude_pos)
        features[f'{col}_spectral_rms'] = spec_rms
        features[f'{col}_spectral_skewness'] = spec_skew
        features[f'{col}_spectral_kurtosis'] = spec_kurt

    return pd.DataFrame([features])


def create_feature_vector(
    df: pd.DataFrame,
    sensor_cols: List[str] = None,
    sampling_rate: float = 100,
    include_frequency: bool = True,
    orientation_robust: bool = True,
    include_per_axis: bool = False,
) -> pd.DataFrame:
    """Create comprehensive feature vector from raw sensor data.

    Args:
        df: Window DataFrame with sensor data columns
        sensor_cols: Sensor column names (default: SENSOR_COLUMNS from config)
        sampling_rate: Sampling rate in Hz (default: 100)
        include_frequency: Include FFT features (default: True)
        orientation_robust: Use magnitude-based features (RECOMMENDED - default: True)
        include_per_axis: Include per-axis features (less robust, default: False)

    Returns:
        DataFrame with extracted features

    Feature Counts:
        - Magnitude only (robust): 41 features (no FFT) or 63 features (with FFT)
        - Per-axis only: 90 features (no FFT) or 156 features (with FFT)
        - Both: ~131 features (no FFT) or ~219 features (with FFT)
    """
    if sensor_cols is None:
        sensor_cols = list(SENSOR_COLUMNS)

    features_list = []

    # ORIENTATION-ROBUST FEATURES (magnitude-based) - RECOMMENDED
    if orientation_robust:
        mag_features = extract_orientation_invariant_features(df, sensor_cols)
        features_list.append(mag_features)

        if include_frequency:
            freq_mag_features = extract_frequency_magnitude_features(
                df, sensor_cols, sampling_rate)
            features_list.append(freq_mag_features)

    # PER-AXIS FEATURES (orientation-dependent) - OPTIONAL
    if include_per_axis:
        time_features = extract_time_domain_features(df, sensor_cols)
        features_list.append(time_features)

        if include_frequency:
            freq_features = extract_frequency_domain_features(
                df, sensor_cols, sampling_rate)
            features_list.append(freq_features)

    # If neither enabled, fall back to per-axis time features
    if not orientation_robust and not include_per_axis:
        logger.warning(
            "No features enabled! Using per-axis time features as fallback.")
        time_features = extract_time_domain_features(df, sensor_cols)
        features_list.append(time_features)

    # Combine all feature sets
    combined_features = pd.concat(features_list, axis=1)

    return combined_features
