"""
Parity tests: Python feature extraction vs C++/MicroPython formulas.

These tests verify the #1 invariant of the HAR framework: training-deployment
parity.  We re-implement the C++ feature formulas in Python and compare them
against the Python feature extraction outputs to catch drift.
"""

import math
import numpy as np
import pandas as pd
import pytest

from utils.feature_extraction import (
    extract_orientation_invariant_features,
    extract_frequency_magnitude_features,
    create_feature_vector,
)
from deployment.code_generator_factory import (
    get_cpp_feature_order,
    compute_feature_reorder_indices,
)
from deployment.validation import validate_before_deployment
from deployment.v2.models.cnn import CNNModelBlock


# ---------------------------------------------------------------------------
# Helpers: Centered magnitude (mirrors Python & C++ centering)
# ---------------------------------------------------------------------------

def _centered_magnitude(df: pd.DataFrame, cols: list) -> np.ndarray:
    """Compute magnitude after per-window mean centering (matches feature extraction)."""
    centered = {c: df[c].values - df[c].values.mean() for c in cols}
    return np.sqrt(sum(centered[c] ** 2 for c in cols))


# ---------------------------------------------------------------------------
# Helpers: C++ formulas re-implemented in pure Python (must mirror
# base_generator.py extract_magnitude_stats and DFT code exactly)
# ---------------------------------------------------------------------------

def _cpp_magnitude_stats(data: np.ndarray) -> dict:
    """Re-implementation of C++ extract_magnitude_stats().

    Mirrors base_generator.py lines ~700-800:
      mean, std, min, max, range, median, q25, q75, iqr,
      skewness, kurtosis, rms, energy, zero_crossings, mean_crossing_rate
    """
    n = len(data)
    mean = np.sum(data) / n
    sum_sq = np.sum(data ** 2)
    variance = sum_sq / n - mean * mean
    std = math.sqrt(max(variance, 0.0))
    rms = math.sqrt(sum_sq / n)
    energy = sum_sq

    sorted_data = np.sort(data)
    minimum = sorted_data[0]
    maximum = sorted_data[-1]
    rang = maximum - minimum

    # Median: C++ floor-division
    if n % 2 == 0:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2.0
    else:
        median = sorted_data[n // 2]

    # Quartiles: C++ nearest-rank (no interpolation)
    q25 = sorted_data[n // 4]
    q75 = sorted_data[(3 * n) // 4]
    iqr = q75 - q25

    # Skewness / kurtosis: bias-corrected with population std z-scores
    pop_std = math.sqrt(max(variance, 0.0)) if variance > 0 else 0.0001
    epsilon = 0.0001
    m3_sum = 0.0
    m4_sum = 0.0
    for v in data:
        z = (v - mean) / (pop_std + epsilon)
        m3_sum += z ** 3
        m4_sum += z ** 4
    g1 = m3_sum / n
    skewness = math.sqrt(n * (n - 1.0)) / (n - 2.0) * g1
    g2 = m4_sum / n - 3.0
    kurtosis = (n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * g2 + 6.0)

    # Zero crossings: strict sign change (product < 0)
    zero_crossings = int(np.sum(data[:-1] * data[1:] < 0))

    # Mean crossing rate
    centered = data - mean
    mean_crossings = int(np.sum(centered[:-1] * centered[1:] < 0))
    mean_crossing_rate = mean_crossings / n

    return {
        'mean': mean, 'std': std, 'min': minimum, 'max': maximum,
        'range': rang, 'median': median, 'q25': q25, 'q75': q75,
        'iqr': iqr, 'skewness': skewness, 'kurtosis': kurtosis,
        'rms': rms, 'energy': energy,
        'zero_crossings': zero_crossings,
        'mean_crossing_rate': mean_crossing_rate,
    }


def _cpp_dft_features(data: np.ndarray, sampling_rate: float) -> dict:
    """Re-implementation of C++ DFT magnitude feature extraction.

    Mirrors base_generator.py extract_frequency_features():
    DC removal → Hann window → DFT → spectral stats.
    """
    n = len(data)
    half_n = n // 2
    freq_step = sampling_rate / n

    # Step 1: DC removal
    sig_mean = np.mean(data)

    # Step 2+3: Hann window + DFT (inline, matching C++ implementation)
    pi_over_nm1 = math.pi / (n - 1)
    dft_mag = []
    dft_freq = []
    for k in range(1, half_n + 1):
        re = 0.0
        im = 0.0
        angle_step = 2.0 * math.pi * k / n
        for i in range(n):
            # Hann window: 0.5 - 0.5*cos(2*PI*i/(N-1))
            w = 0.5 - 0.5 * math.cos(2.0 * pi_over_nm1 * i)
            val = (data[i] - sig_mean) * w  # DC-removed + windowed
            angle = angle_step * i
            re += val * math.cos(angle)
            im -= val * math.sin(angle)
        mag = math.sqrt(re * re + im * im)
        dft_mag.append(mag)
        dft_freq.append(k * freq_step)

    dft_mag = np.array(dft_mag)
    dft_freq = np.array(dft_freq)

    if len(dft_mag) == 0:
        return {}

    # Dominant frequency
    dom_idx = np.argmax(dft_mag)
    dominant_frequency = dft_freq[dom_idx]
    dominant_frequency_magnitude = dft_mag[dom_idx]

    # Spectral centroid
    mag_sum = np.sum(dft_mag)
    if mag_sum > 0:
        spectral_centroid = np.sum(dft_freq * dft_mag) / mag_sum
    else:
        spectral_centroid = 0.0

    # Energy in bands: <2 Hz, 2-5 Hz, >=5 Hz
    energy_low = 0.0
    energy_mid = 0.0
    energy_high = 0.0
    for k in range(len(dft_mag)):
        e = dft_mag[k] ** 2
        freq = dft_freq[k]
        if freq < 2.0:
            energy_low += e
        elif freq < 5.0:
            energy_mid += e
        else:
            energy_high += e

    # Spectral rolloff (85%)
    cumsum = np.cumsum(dft_mag)
    threshold = 0.85 * mag_sum
    rolloff_freq = dft_freq[-1]
    for k in range(len(cumsum)):
        if cumsum[k] >= threshold:
            rolloff_freq = dft_freq[k]
            break

    # Spectral shape descriptors (RMS, skewness, kurtosis of bins)
    sq_sum = np.sum(dft_mag ** 2)
    spec_rms = math.sqrt(sq_sum / half_n) if half_n > 0 else 0.0
    spec_mean = mag_sum / half_n if half_n > 0 else 0.0
    spec_var = (sq_sum / half_n - spec_mean * spec_mean) if half_n > 0 else 0.0
    spec_std = math.sqrt(spec_var) if spec_var > 0 else 0.0001

    m3 = 0.0
    m4 = 0.0
    for m in dft_mag:
        z = (m - spec_mean) / (spec_std + 1e-7)
        z2 = z * z
        m3 += z2 * z
        m4 += z2 * z2
    nn = half_n
    spec_skew = 0.0
    spec_kurt = 0.0
    if nn > 2:
        spec_skew = (m3 / nn) * (nn * (nn + 1)) / ((nn - 1) * (nn - 2))
    if nn > 3:
        raw_kurt = m4 / nn
        spec_kurt = ((nn + 1) * raw_kurt - 3.0 * (nn - 1)) * (nn - 1) / ((nn - 2) * (nn - 3))
        # excess kurtosis (Fisher)

    spec_entropy = 0.0
    if mag_sum > 0:
        probs = dft_mag / mag_sum
        probs_nz = probs[probs > 1e-12]
        if len(probs_nz) > 0:
            spec_entropy = -np.sum(probs_nz * np.log2(probs_nz))

    return {
        'dominant_frequency': dominant_frequency,
        'dominant_frequency_magnitude': dominant_frequency_magnitude,
        'spectral_centroid': spectral_centroid,
        'energy_low_freq': energy_low,
        'energy_mid_freq': energy_mid,
        'energy_high_freq': energy_high,
        'spectral_rolloff': rolloff_freq,
        'spectral_rms': spec_rms,
        'spectral_skewness': spec_skew,
        'spectral_kurtosis': spec_kurt,
        'spectral_entropy': spec_entropy,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_window():
    """Create a deterministic 150-sample synthetic IMU window."""
    rng = np.random.RandomState(42)
    n = 150
    t = np.arange(n) / 100.0  # 1.5 s @ 100 Hz
    return pd.DataFrame({
        'aX': 0.5 * np.sin(2 * np.pi * 2 * t) + rng.normal(0, 0.1, n),
        'aY': 0.3 * np.cos(2 * np.pi * 1.5 * t) + rng.normal(0, 0.05, n),
        'aZ': 9.8 + 0.2 * np.sin(2 * np.pi * 3 * t) + rng.normal(0, 0.08, n),
        'gX': 0.1 * np.sin(2 * np.pi * 2 * t) + rng.normal(0, 0.02, n),
        'gY': 0.05 * np.cos(2 * np.pi * 1 * t) + rng.normal(0, 0.01, n),
        'gZ': 0.08 * np.sin(2 * np.pi * 2.5 * t) + rng.normal(0, 0.015, n),
    })


# ---------------------------------------------------------------------------
# Time-domain parity: orientation-invariant magnitude stats
# ---------------------------------------------------------------------------

class TestTimeDomainParity:
    """Compare Python extract_orientation_invariant_features() with C++ formulas."""

    # Features that should match exactly (integer or simple sums)
    EXACT_FEATURES = ['zero_crossings']

    # Features where C++ uses nearest-rank quartiles vs Python interpolation
    QUARTILE_FEATURES = ['q25', 'q75', 'iqr']

    # Features where bias-corrected skewness/kurtosis may differ slightly
    # due to population-vs-sample std in z-score denominator
    HIGHER_TOL_FEATURES = ['skewness', 'kurtosis']

    @pytest.mark.parametrize("mag_name,accel_cols,gyro_cols", [
        ("acc_mag", ['aX', 'aY', 'aZ'], None),
        ("gyro_mag", None, ['gX', 'gY', 'gZ']),
    ])
    def test_magnitude_stats_parity(self, synthetic_window, mag_name, accel_cols, gyro_cols):
        """Magnitude time-domain features: Python vs C++ formulas."""
        df = synthetic_window
        py_feats = extract_orientation_invariant_features(df)

        # Compute magnitude the same way Python does (centered)
        cols = accel_cols if accel_cols else gyro_cols
        mag = _centered_magnitude(df, cols)
        cpp_stats = _cpp_magnitude_stats(mag)

        for stat_name, cpp_val in cpp_stats.items():
            py_col = f'{mag_name}_{stat_name}'
            assert py_col in py_feats.columns, f"Missing Python feature: {py_col}"
            py_val = py_feats[py_col].iloc[0]

            if stat_name in self.EXACT_FEATURES:
                assert cpp_val == py_val, (
                    f"{py_col}: C++={cpp_val}, Python={py_val}")
            elif stat_name in self.QUARTILE_FEATURES:
                # Nearest-rank vs interpolation: allow wider tolerance
                assert abs(cpp_val - py_val) < 0.1 * (abs(py_val) + 1e-6), (
                    f"{py_col}: C++={cpp_val}, Python={py_val} (quartile mismatch)")
            elif stat_name in self.HIGHER_TOL_FEATURES:
                assert abs(cpp_val - py_val) < 0.05 * (abs(py_val) + 1.0), (
                    f"{py_col}: C++={cpp_val:.6f}, Python={py_val:.6f}")
            else:
                # Standard tolerance: 0.1% relative or 1e-6 absolute
                assert abs(cpp_val - py_val) < max(1e-6, 0.001 * abs(py_val)), (
                    f"{py_col}: C++={cpp_val:.8f}, Python={py_val:.8f}")

    def test_jerk_features_parity(self, synthetic_window):
        """Jerk features: Python vs C++ (simple diff-based)."""
        df = synthetic_window
        py_feats = extract_orientation_invariant_features(df)

        # C++ jerk: sqrt(sum(diff(aX)^2, diff(aY)^2, diff(aZ)^2))
        jerk = np.sqrt(
            np.diff(df['aX'].values)**2
            + np.diff(df['aY'].values)**2
            + np.diff(df['aZ'].values)**2
        )
        cpp_mean = np.mean(jerk)
        cpp_std = np.std(jerk)  # population std (ddof=0)
        cpp_max = np.max(jerk)

        assert abs(cpp_mean - py_feats['acc_jerk_mag_mean'].iloc[0]) < 1e-6
        assert abs(cpp_std - py_feats['acc_jerk_mag_std'].iloc[0]) < 1e-6
        assert abs(cpp_max - py_feats['acc_jerk_mag_max'].iloc[0]) < 1e-6


# ---------------------------------------------------------------------------
# Frequency-domain parity: DFT magnitude features
# ---------------------------------------------------------------------------

class TestFrequencyDomainParity:
    """Compare Python extract_frequency_magnitude_features() with C++ DFT.
    """

    def _check_dft_features(self, py_feats, prefix, mag_data):
        cpp_dft = _cpp_dft_features(mag_data, 100.0)
        for feat_name, cpp_val in cpp_dft.items():
            py_col = f'{prefix}_{feat_name}'
            assert py_col in py_feats.columns, f"Missing: {py_col}"
            py_val = py_feats[py_col].iloc[0]

            # Standard tolerance
            tol = max(0.1, 0.01 * abs(py_val))

            assert abs(cpp_val - py_val) < tol, (
                f"{py_col}: C++_DFT={cpp_val:.4f}, Python_FFT={py_val:.4f}")

    @pytest.mark.parametrize("mag_name,cols", [
        ("acc_mag", ['aX', 'aY', 'aZ']),
        ("gyro_mag", ['gX', 'gY', 'gZ']),
    ])
    def test_mag_dft_parity(self, synthetic_window, mag_name, cols):
        """Magnitude DFT features: Python FFT vs C++ DFT loop."""
        df = synthetic_window
        py_feats = extract_frequency_magnitude_features(df, sampling_rate=100)
        mag = _centered_magnitude(df, cols)
        self._check_dft_features(py_feats, mag_name, mag)


# ---------------------------------------------------------------------------
# Feature count and naming
# ---------------------------------------------------------------------------

class TestFeatureCounts:
    """Verify feature counts match documented values for each mode."""

    def test_orientation_invariant_time_only_41(self, synthetic_window):
        feats = create_feature_vector(
            synthetic_window, include_frequency=False,
            orientation_robust=True, include_per_axis=False)
        assert feats.shape[1] == 41, f"Expected 41, got {feats.shape[1]}: {sorted(feats.columns.tolist())}"

    def test_orientation_invariant_63(self, synthetic_window):
        feats = create_feature_vector(
            synthetic_window, include_frequency=True,
            orientation_robust=True, include_per_axis=False)
        assert feats.shape[1] == 63, f"Expected 63, got {feats.shape[1]}: {sorted(feats.columns.tolist())}"

    def test_time_domain_90(self, synthetic_window):
        feats = create_feature_vector(
            synthetic_window, include_frequency=False,
            orientation_robust=False, include_per_axis=True)
        assert feats.shape[1] == 90, f"Expected 90, got {feats.shape[1]}: {sorted(feats.columns.tolist())}"

    def test_all_features_156(self, synthetic_window):
        feats = create_feature_vector(
            synthetic_window, include_frequency=True,
            orientation_robust=False, include_per_axis=True)
        assert feats.shape[1] == 156, f"Expected 156, got {feats.shape[1]}: {sorted(feats.columns.tolist())}"


# ---------------------------------------------------------------------------
# Feature reorder indices
# ---------------------------------------------------------------------------

class TestFeatureReorderIndices:
    """Verify feature reordering logic in code_generator_factory."""

    def test_cpp_feature_order_count(self):
        """C++ feature order has 33 entries for time-only orientation-invariant."""
        # Simulate a model with orientation_invariant_time_only features
        feats_time_only = [
            f'{mag}_{stat}'
            for mag in ['acc_mag', 'gyro_mag']
            for stat in ['mean', 'std', 'min', 'max', 'range', 'median',
                         'q25', 'q75', 'iqr', 'skewness', 'kurtosis',
                         'rms', 'energy', 'zero_crossings', 'mean_crossing_rate']
        ] + ['acc_jerk_mag_mean', 'acc_jerk_mag_std', 'acc_jerk_mag_max']
        cpp_order = get_cpp_feature_order(feats_time_only)
        assert len(cpp_order) == 33

    def test_reorder_roundtrip(self):
        """Reorder indices correctly map alphabetical → C++ order."""
        # Training order is alphabetical (as pandas would produce)
        training_features = sorted([
            f'{mag}_{stat}'
            for mag in ['acc_mag', 'gyro_mag']
            for stat in ['mean', 'std', 'min', 'max', 'range', 'median',
                         'q25', 'q75', 'iqr', 'skewness', 'kurtosis',
                         'rms', 'energy', 'zero_crossings', 'mean_crossing_rate']
        ] + ['acc_jerk_mag_mean', 'acc_jerk_mag_std', 'acc_jerk_mag_max'])

        cpp_order = get_cpp_feature_order(training_features)
        indices = compute_feature_reorder_indices(training_features, cpp_order)

        # Applying the index mapping should transform C++ order → training order
        # indices[i] tells us: training feature i came from C++ position indices[i]
        assert len(indices) == len(training_features)
        # All indices should be valid
        assert all(0 <= idx < len(cpp_order) for idx in indices)
        # No duplicate indices (bijection)
        assert len(set(indices)) == len(indices)

    def test_per_axis_no_reorder(self):
        """Per-axis features should return identity order (no reordering needed)."""
        per_axis_features = [f'{col}_{stat}'
                             for col in ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
                             for stat in ['mean', 'std']]
        cpp_order = get_cpp_feature_order(per_axis_features)
        # For per-axis, get_cpp_feature_order falls back to original order
        assert cpp_order == per_axis_features


# ---------------------------------------------------------------------------
# Population std invariant
# ---------------------------------------------------------------------------

class TestPopulationStd:
    """Verify that std uses ddof=0 (population) not ddof=1 (sample)."""

    def test_python_uses_population_std(self, synthetic_window):
        """np.std(data) defaults to ddof=0 — confirm feature extraction agrees."""
        df = synthetic_window
        feats = extract_orientation_invariant_features(df)

        acc_mag = _centered_magnitude(df, ['aX', 'aY', 'aZ'])
        pop_std = np.std(acc_mag, ddof=0)
        sample_std = np.std(acc_mag, ddof=1)

        py_std = feats['acc_mag_std'].iloc[0]

        # Must match population std, not sample std
        assert abs(py_std - pop_std) < 1e-10
        assert abs(py_std - sample_std) > 1e-6  # Must NOT match sample std


# ---------------------------------------------------------------------------
# Kalman filter parity: Python implementation vs generated C++ equations
# ---------------------------------------------------------------------------

def _cpp_kalman_filter(signal: np.ndarray, dt: float, Q_scale: float, R: float) -> np.ndarray:
    """Re-implement the generated C++ Kalman filter equations in Python.

    Mirrors deployment/base_generator.py _generate_kalman_filter_implementation():
    - constant-velocity model, H = [1, 0]
    - lazy-init: first sample seeds state as [z[0], 0.0]
    - scalar update form (H = [1,0] makes S = P[0,0] + R)

    PARITY REQUIREMENT: This function MUST be kept in sync with both
    utils/data_processing.kalman_filter() (Python training path) and the C++
    code emitted by BaseCodeGenerator._generate_kalman_filter_implementation()
    (deployment path).  Any change to the filter equations in one place must be
    reflected in all three locations to maintain training-deployment parity.
    """
    n = len(signal)
    if n == 0:
        return signal.copy()

    # Constant Q matrix entries (matches C++ codegen)
    q00 = Q_scale * (dt ** 3) / 3.0
    q01 = Q_scale * (dt ** 2) / 2.0
    q11 = Q_scale * dt

    # State and covariance — lazy init on first sample
    x0 = signal[0]
    x1 = 0.0
    p00 = R
    p01 = 0.0
    p10 = 0.0
    p11 = R

    output = np.empty(n, dtype=np.float64)

    for k in range(n):
        # --- Predict ---
        nx0 = x0 + dt * x1
        nx1 = x1
        np00 = p00 + dt * (p10 + p01) + dt * dt * p11 + q00
        np01 = p01 + dt * p11 + q01
        np10 = p10 + dt * p11 + q01
        np11 = p11 + q11
        x0, x1 = nx0, nx1
        p00, p01, p10, p11 = np00, np01, np10, np11

        # --- Update (scalar form, H = [1, 0]) ---
        S = p00 + R
        K0 = p00 / S
        K1 = p10 / S
        y = signal[k] - x0
        x0 += K0 * y
        x1 += K1 * y
        new_p00 = (1.0 - K0) * p00
        new_p01 = (1.0 - K0) * p01
        new_p10 = p10 - K1 * p00
        new_p11 = p11 - K1 * p01
        p00, p01, p10, p11 = new_p00, new_p01, new_p10, new_p11

        output[k] = x0

    return output


class TestKalmanFilterParity:
    """Verify Python kalman_filter() matches the generated C++ equations."""

    def test_kalman_single_channel_parity(self):
        """Single synthetic channel: Python vs C++ re-implementation."""
        from utils.data_processing import kalman_filter

        rng = np.random.RandomState(7)
        n = 100
        t = np.arange(n) / 100.0
        raw = np.sin(2 * np.pi * 2 * t) + rng.normal(0, 0.2, n)

        Q_scale = 1e-3
        R_noise = 0.1
        fs = 100
        dt = 1.0 / fs

        df = pd.DataFrame({'ch0': raw})
        py_out = kalman_filter(df, process_noise=Q_scale, measurement_noise=R_noise,
                               fs=fs)['ch0'].values
        cpp_out = _cpp_kalman_filter(raw, dt, Q_scale, R_noise)

        # Should be numerically identical (same algorithm, same lazy-init)
        np.testing.assert_allclose(py_out, cpp_out, rtol=1e-9, atol=1e-9,
                                   err_msg="Python kalman_filter() diverges from C++ equations")

    def test_kalman_empty_column(self):
        """Empty column returns empty without crash."""
        from utils.data_processing import kalman_filter

        df = pd.DataFrame({'a': pd.Series([], dtype=float),
                           'b': pd.Series([], dtype=float)})
        result = kalman_filter(df)
        assert result.shape == (0, 2)

    def test_kalman_lazy_init_matches_python(self):
        """C++ lazy-init (state seeded from first sample) matches Python."""
        from utils.data_processing import kalman_filter

        # Step signal: starts at 5.0, then drops — lazy-init should match Python's x=[z[0],0]
        raw = np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        df = pd.DataFrame({'x': raw})
        py_out = kalman_filter(df, process_noise=1e-3, measurement_noise=0.1,
                               fs=100)['x'].values
        cpp_out = _cpp_kalman_filter(raw, dt=0.01, Q_scale=1e-3, R=0.1)

        np.testing.assert_allclose(py_out, cpp_out, rtol=1e-9, atol=1e-9)

    def test_kalman_multi_channel_parity(self):
        """All 6 sensor channels agree between Python and C++ equations."""
        from utils.data_processing import kalman_filter

        rng = np.random.RandomState(42)
        n = 50
        t = np.arange(n) / 100.0
        cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
        data = {c: np.sin(2 * np.pi * (i + 1) * t) + rng.normal(0, 0.1, n)
                for i, c in enumerate(cols)}
        df = pd.DataFrame(data)

        Q_scale = 1e-3
        R_noise = 0.1
        fs = 100
        dt = 1.0 / fs

        py_result = kalman_filter(df, process_noise=Q_scale,
                                  measurement_noise=R_noise, fs=fs)

        for col in cols:
            cpp_out = _cpp_kalman_filter(df[col].values, dt, Q_scale, R_noise)
            np.testing.assert_allclose(
                py_result[col].values, cpp_out, rtol=1e-9, atol=1e-9,
                err_msg=f"Parity failure on channel {col}"
            )


class TestAdditionalParityGuards:
    def test_orientation_features_require_full_imu_columns(self, synthetic_window):
        accel_only = synthetic_window[['aX', 'aY', 'aZ']].copy()
        with pytest.raises(ValueError, match="require full 6-axis IMU columns"):
            create_feature_vector(
                accel_only, include_frequency=False,
                orientation_robust=True, include_per_axis=False
            )

    def test_fft_lowpass_window_mode_matches_manual_chunking(self):
        from utils.data_processing import fft_lowpass_filter
        rng = np.random.RandomState(123)
        sig = rng.normal(size=301)
        df = pd.DataFrame({'aX': sig})
        out = fft_lowpass_filter(df, cutoff=10, fs=100, window_size=150)['aX'].values

        expected = sig.copy()
        for start in (0, 150, 300):
            end = min(start + 150, len(sig))
            chunk = sig[start:end]
            if len(chunk) <= 1:
                continue
            cutoff_bin = max(1, int(np.ceil(10 * len(chunk) / 100)))
            X = np.fft.rfft(chunk)
            X[cutoff_bin:] = 0.0
            expected[start:end] = np.fft.irfft(X, n=len(chunk))

        np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-9)

    def test_cnn2d_direct_codegen_raises_clear_error(self):
        model_data = {
            "cnn_weights": {
                "layers": [
                    {"type": "conv2d", "weights": [[[[0.1]]]], "bias": [0.0]},
                    {"type": "dense", "weights": [[1.0]], "bias": [0.0], "in_features": 1, "out_features": 1},
                ]
            }
        }
        with pytest.raises(NotImplementedError, match="not supported"):
            CNNModelBlock(model_data, feature_names=["ch0"], classes=["a"], window_size=150)

    def test_v2_validation_reports_missing_files_explicitly(self):
        model_data = {"model_type": "neural_network", "feature_names": ["f1"], "classes": ["a"]}
        files = {"har_config.h": "#define HAR_NUM_FEATURES 1\n"}
        report = validate_before_deployment(model_data, files)
        issues = report.get("checks", {}).get("code", {}).get("issues", [])
        assert any("Missing required v2 files" in issue for issue in issues)

    def test_pytorch_cnn2d_derives_window_and_channels_from_training_data(self, monkeypatch):
        from utils.edge_ml_model import EdgeMLModel

        class _DummyTrainer:
            def __init__(self, model):
                self.model = model

            def train(self, *args, **kwargs):
                return {}

            def predict(self, X):
                return np.zeros(len(X), dtype=int)

            def predict_proba(self, X):
                return np.ones((len(X), 2), dtype=float) * 0.5

        monkeypatch.setattr("utils.edge_ml_model.PyTorchTrainer", _DummyTrainer)

        X = np.random.RandomState(0).randn(8, 120, 4)
        y = pd.Series(["walk", "run", "walk", "run", "walk", "run", "walk", "run"])
        model = EdgeMLModel("pytorch_cnn2d", window_size=150, n_channels=6, max_iter=1)
        model.train(X, y, use_cross_validation=False)
        assert model.model_params["window_size"] == 120
        assert model.model_params["n_channels"] == 4
