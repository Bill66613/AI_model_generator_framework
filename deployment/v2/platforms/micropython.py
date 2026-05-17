"""
MicroPythonCodeGen — generates a self-contained MicroPython HAR module.

Unlike C++ generators, this produces pure Python files:
  har_model.py  — complete module (feature extraction + model + inference API)
  main.py       — example script that uses har_model.py

Supports Random Forest, Neural Network (MLP, pytorch_mlp), and SVM models.
CNN models (pytorch_cnn) are noted as unsupported in the output stub.

Feature extraction formulas match Python utils/feature_extraction.py exactly
(same as v2/feature_block.py guarantees for C++).

PARITY NOTE:
Any formula change in feature_block.py must be mirrored here.
Key verified formulas:
  - skewness / kurtosis: population std for z-scores
  - zero_crossing_rate: count where data[i-1]*data[i] < 0
  - mean_crossing_rate: count where (x-mean)[i-1]*(x-mean)[i] < 0
  - spectral_rolloff: 85% cumulative magnitude (not energy)
  - band thresholds: low [0,5) Hz, mid [5,15) Hz, high [15,fs/2]
"""

from __future__ import annotations
from typing import Any, Dict, List

import numpy as np


class MicroPythonCodeGen:
    """Generates har_model.py and main.py for MicroPython deployment."""

    def __init__(
        self,
        model_data: Dict[str, Any],
        feature_names: List[str],
        classes: List[str],
        feature_means: List[float],
        feature_stds: List[float],
        window_size: int,
        sampling_rate: int,
        precision: int,
        model_type: str,
        sketch_name: str = "har_sketch",
    ):
        self.model_data = model_data
        self.feature_names = feature_names
        self.classes = classes
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.precision = precision
        self.model_type = model_type
        self.sketch_name = sketch_name

        # Extract model-specific weights
        self._trees: List[Dict] = model_data.get("trees", [])
        self._support_vectors: List = model_data.get("support_vectors", [])
        self._dual_coef: List = model_data.get("dual_coefficients", [])
        self._intercept: List = model_data.get("intercept", [])
        self._gamma: float = float(model_data.get("gamma", 0.1))

        self._nn_weights: List = []
        self._nn_biases: List = []
        self._extract_nn_params()

    # ------------------------------------------------------------------
    # Neural-network parameter extraction
    # ------------------------------------------------------------------

    def _extract_nn_params(self):
        if self.model_type not in ("neural_network", "pytorch_mlp"):
            return
        model_obj = self.model_data.get("model_object")
        if model_obj and hasattr(model_obj, "model") and hasattr(model_obj.model, "coefs_"):
            coefs = model_obj.model.coefs_
            intercepts = model_obj.model.intercepts_
            reorder = self.model_data.get("_feature_reorder_indices")
            c0 = coefs[0][reorder, :] if reorder is not None else coefs[0]
            self._nn_weights = [c0.tolist()] + [c.tolist() for c in coefs[1:]]
            self._nn_biases = [b.tolist() for b in intercepts]
            return
        # pytorch_mlp path: pytorch_coefs
        pytorch_coefs = self.model_data.get("pytorch_coefs", {})
        if pytorch_coefs:
            self._nn_weights = pytorch_coefs.get("coefs_", [])
            self._nn_biases = pytorch_coefs.get("intercepts_", [])
            return
        # Legacy dict path
        weights = self.model_data.get("weights", {})
        if weights:
            iw = weights.get("input_weights", [])
            hb = weights.get("hidden_biases", [])
            ow = weights.get("output_weights", [])
            ob = weights.get("output_biases", [])
            if iw:
                self._nn_weights = [iw] + ([ow] if ow else [])
                self._nn_biases = [hb] + ([ob] if ob else [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_files(self) -> Dict[str, str]:
        return {
            "har_model.py": self._generate_module(),
            "main.py": self._generate_main(),
        }

    # ------------------------------------------------------------------
    # har_model.py
    # ------------------------------------------------------------------

    def _generate_module(self) -> str:
        lines: List[str] = []
        lines.append(self._header_comment())
        lines.append("import math\n")
        lines.append(self._constants())
        lines.append(self._activity_names())
        lines.append(self._scaling_arrays())
        lines.append(self._model_data())
        lines.append(self._har_init())
        lines.append(self._scale_features())
        lines.append(self._softmax_and_predict())
        lines.append(self._predict_internal())
        lines.append(self._extract_features())
        lines.append(self._get_activity_name())
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Header comment
    # ------------------------------------------------------------------

    def _header_comment(self) -> str:
        return f'''\
"""
har_model.py — Auto-generated MicroPython HAR inference module
Generated by HAR Edge Framework v2

Model type   : {self.model_type}
Features     : {len(self.feature_names)}
Classes      : {len(self.classes)}
Platform     : MicroPython
"""
'''

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    def _constants(self) -> str:
        return (
            f"NUM_FEATURES = {len(self.feature_names)}\n"
            f"NUM_CLASSES  = {len(self.classes)}\n"
            f"SAMPLING_RATE = {self.sampling_rate}\n"
            f"WINDOW_SIZE  = {self.window_size}\n"
            f"CONFIDENCE_THRESHOLD = 0.6\n"
        )

    def _activity_names(self) -> str:
        names = ", ".join(f'"{c}"' for c in self.classes)
        return f"ACTIVITY_NAMES = [{names}]\n"

    def _scaling_arrays(self) -> str:
        prec = self.precision
        means = ", ".join(f"{m:.{prec}f}" for m in self.feature_means)
        stds  = ", ".join(f"{s:.{prec}f}" for s in self.feature_stds)
        return (
            f"FEATURE_MEANS = [{means}]\n"
            f"FEATURE_STDS  = [{stds}]\n"
        )

    # ------------------------------------------------------------------
    # Model weights
    # ------------------------------------------------------------------

    def _model_data(self) -> str:
        if self.model_type == "random_forest":
            return self._rf_data()
        elif self.model_type in ("neural_network", "pytorch_mlp"):
            return self._nn_data()
        elif self.model_type == "svm":
            return self._svm_data()
        elif self.model_type in ("pytorch_cnn", "pytorch_cnn2d"):
            return "# CNN models are not supported on MicroPython\n"
        return "# No model data\n"

    def _rf_data(self) -> str:
        if not self._trees:
            return "TREES = []  # No trees extracted\n"
        lines = ["TREES = ["]
        for tree in self._trees:
            fi   = tree.get("feature_indices", [])
            th   = tree.get("thresholds", [])
            lc   = tree.get("left_children", [])
            rc   = tree.get("right_children", [])
            vals = tree.get("values", [])
            leaf_classes = []
            for v in vals:
                inner = v[0] if isinstance(v, list) and isinstance(v[0], list) else v
                try:
                    leaf_classes.append(int(np.argmax(inner)))
                except Exception:
                    leaf_classes.append(0)
            lines.append("    {")
            lines.append(f"        'fi': {fi},")
            lines.append(f"        'th': {self._fmt_list(th)},")
            lines.append(f"        'lc': {lc},")
            lines.append(f"        'rc': {rc},")
            lines.append(f"        'pred': {leaf_classes},")
            lines.append("    },")
        lines.append("]\n")
        return "\n".join(lines)

    def _nn_data(self) -> str:
        if not self._nn_weights:
            return "NN_WEIGHTS = []\nNN_BIASES = []\n"
        lines = ["NN_WEIGHTS = ["]
        for i, w in enumerate(self._nn_weights):
            r = len(w)
            c = len(w[0]) if w else 0
            lines.append(f"    # Layer {i}: {r} x {c}")
            lines.append(f"    {self._fmt_2d(w)},")
        lines.append("]\n")
        lines.append("NN_BIASES = [")
        for i, b in enumerate(self._nn_biases):
            lines.append(f"    # Layer {i}: {len(b)}")
            lines.append(f"    {self._fmt_list(b)},")
        lines.append("]\n")
        return "\n".join(lines)

    def _svm_data(self) -> str:
        lines: List[str] = []
        lines.append(f"SVM_GAMMA = {self._gamma}")
        if self._support_vectors:
            lines.append(f"SVM_NUM_SV = {len(self._support_vectors)}")
            lines.append("SVM_SUPPORT_VECTORS = [")
            for sv in self._support_vectors:
                lines.append(f"    {self._fmt_list(sv)},")
            lines.append("]")
        else:
            lines.append("SVM_NUM_SV = 0\nSVM_SUPPORT_VECTORS = []")
        if self._dual_coef:
            lines.append("SVM_DUAL_COEF = [")
            for row in self._dual_coef:
                lines.append(f"    {self._fmt_list(row)},")
            lines.append("]")
        else:
            lines.append("SVM_DUAL_COEF = []")
        lines.append(f"SVM_INTERCEPT = {self._fmt_list(self._intercept)}")
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Core inference functions
    # ------------------------------------------------------------------

    def _har_init(self) -> str:
        return "\ndef har_init():\n    pass\n"

    def _scale_features(self) -> str:
        return '''
def _scale_features(features):
    """Apply StandardScaler: (x - mean) / std."""
    scaled = [0.0] * NUM_FEATURES
    for i in range(NUM_FEATURES):
        std = FEATURE_STDS[i] if FEATURE_STDS[i] > 1e-4 else 1.0
        val = features[i]
        if val != val:  # NaN check
            val = 0.0
        s = (val - FEATURE_MEANS[i]) / std
        if s < -10.0: s = -10.0
        elif s > 10.0: s = 10.0
        scaled[i] = s
    return scaled
'''

    def _softmax_and_predict(self) -> str:
        return '''
def _softmax(scores):
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def har_predict(features):
    """Predict activity class from a raw (unscaled) feature vector.

    Args:
        features: list of NUM_FEATURES floats

    Returns:
        (int, float) — (predicted_class or -1, confidence)
    """
    scaled = _scale_features(features)
    result, scores = _predict_internal(scaled)
    if result < 0 or result >= NUM_CLASSES:
        return -1, 0.0
    probs = _softmax(scores)
    confidence = max(probs)
    if confidence < CONFIDENCE_THRESHOLD:
        return -1, confidence
    return result, confidence
'''

    def _predict_internal(self) -> str:
        if self.model_type == "random_forest":
            return self._rf_predict()
        elif self.model_type in ("neural_network", "pytorch_mlp"):
            return self._nn_predict()
        elif self.model_type == "svm":
            return self._svm_predict()
        return "\ndef _predict_internal(features):\n    return 0, [1.0]\n"

    def _rf_predict(self) -> str:
        return '''
def _predict_tree(tree, features):
    fi, th, lc, rc, pred = tree['fi'], tree['th'], tree['lc'], tree['rc'], tree['pred']
    node = 0
    while fi[node] >= 0:
        node = lc[node] if features[fi[node]] <= th[node] else rc[node]
    return pred[node]


def _predict_internal(features):
    votes = [0] * NUM_CLASSES
    for tree in TREES:
        cls = _predict_tree(tree, features)
        if 0 <= cls < NUM_CLASSES:
            votes[cls] += 1
    total = len(TREES) or 1
    scores = [v / total for v in votes]
    best = max(range(NUM_CLASSES), key=lambda i: votes[i])
    return best, scores
'''

    def _nn_predict(self) -> str:
        return '''
def _predict_internal(features):
    current = list(features)
    n_layers = len(NN_WEIGHTS)
    for li in range(n_layers):
        w, b = NN_WEIGHTS[li], NN_BIASES[li]
        out = [0.0] * len(b)
        for j in range(len(b)):
            s = b[j]
            for i in range(len(current)):
                s += current[i] * w[i][j]
            out[j] = s if (s > 0.0 or li == n_layers - 1) else 0.0
        current = out
    best = max(range(len(current)), key=lambda i: current[i])
    return best, current
'''

    def _svm_predict(self) -> str:
        return '''
def _rbf_kernel(x1, x2, gamma):
    s = sum((a - b) ** 2 for a, b in zip(x1, x2))
    e = -gamma * s
    return math.exp(e) if e > -80.0 else 0.0


def _predict_internal(features):
    scores = [SVM_INTERCEPT[cls] if SVM_INTERCEPT else 0.0
              for cls in range(NUM_CLASSES)]
    for sv_idx in range(SVM_NUM_SV):
        k = _rbf_kernel(features, SVM_SUPPORT_VECTORS[sv_idx], SVM_GAMMA)
        for cls in range(NUM_CLASSES):
            scores[cls] += SVM_DUAL_COEF[cls][sv_idx] * k
    best = max(range(NUM_CLASSES), key=lambda i: scores[i])
    return best, scores
'''

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self) -> str:
        """Dispatch to the correct feature extraction function."""
        fset = set(self.feature_names)
        has_acc_mag  = any(f.startswith("acc_mag_") for f in fset)
        has_per_axis = any(
            f.startswith(("aX_", "aY_", "aZ_", "gX_", "gY_", "gZ_")) for f in fset
        )
        has_freq = any(
            "dominant_frequency" in f or "spectral" in f or "energy_low_freq" in f
            for f in fset
        )
        is_raw = (
            not has_per_axis and not has_acc_mag
            and any(f in fset for f in ("aX", "aY", "aZ", "gX", "gY", "gZ"))
        )
        is_cnn = bool(self.feature_names) and all(f.startswith("cnn_in_") for f in self.feature_names)

        if is_cnn:
            return self._cnn_extract_stub()
        if is_raw:
            return self._raw_extract()
        if has_per_axis and has_freq:
            return self._per_axis_extract(include_freq=True)
        if has_per_axis:
            return self._per_axis_extract(include_freq=False)
        # Orientation-invariant (acc_mag / gyro_mag based)
        include_freq = has_freq
        return self._orientation_invariant_extract(include_freq=include_freq)

    # ------------------------------------------------------------------
    # _magnitude_stats helper (shared by orientation-invariant modes)
    # ------------------------------------------------------------------

    def _magnitude_stats_fn(self) -> str:
        return '''
def _magnitude_stats(mag):
    """15 statistical features from a magnitude/signal vector.
    Matches Python utils/feature_extraction.py exactly.
    """
    n = len(mag)
    s = sum(mag)
    ssq = sum(x * x for x in mag)
    mean = s / n
    var = ssq / n - mean * mean
    if var < 0: var = 0.0
    std = math.sqrt(var)
    rms = math.sqrt(ssq / n)
    energy = ssq
    mn, mx = min(mag), max(mag)

    s_data = sorted(mag)
    mid = n // 2
    median = (s_data[mid-1] + s_data[mid]) / 2 if n % 2 == 0 else s_data[mid]

    def _pct(p):
        pos = p * (n - 1)
        lo = int(pos)
        frac = pos - lo
        hi = min(lo + 1, n - 1)
        return s_data[lo] + frac * (s_data[hi] - s_data[lo])

    q25, q75 = _pct(0.25), _pct(0.75)
    iqr = q75 - q25

    # Skewness / kurtosis — population std for z-score normalization
    pop_std = math.sqrt(var) if var > 0 else 1e-4
    m3 = m4 = 0.0
    for v in mag:
        z = (v - mean) / (pop_std + 1e-7)
        z2 = z * z
        m3 += z * z2
        m4 += z2 * z2
    g1 = m3 / n
    skew = 0.0
    if n >= 3:
        skew = math.sqrt(n * (n - 1)) / (n - 2) * g1
    g2 = m4 / n - 3.0
    kurt = -3.0
    if n >= 4:
        kurt = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6.0)

    # Zero crossings and mean crossing rate
    zc = mc = 0
    for i in range(1, n):
        if mag[i-1] * mag[i] < 0: zc += 1
        if (mag[i-1] - mean) * (mag[i] - mean) < 0: mc += 1

    return [mean, std, mn, mx, mx - mn,
            median, q25, q75, iqr,
            skew, kurt, rms, energy, float(zc), mc / n]

'''

    # ------------------------------------------------------------------
    # Orientation-invariant extraction
    # ------------------------------------------------------------------

    def _orientation_invariant_extract(self, include_freq: bool = False) -> str:
        code = self._magnitude_stats_fn()

        if include_freq:
            code += self._frequency_features_fn()

        code += '''
def extract_features(sensor_data, samples):
    """Orientation-invariant feature extraction.

    Args:
        sensor_data: list of [aX, aY, aZ, gX, gY, gZ] lists
        samples: int — number of samples (== len(sensor_data))

    Returns:
        list of NUM_FEATURES floats
    """
    if samples <= 1:
        return [0.0] * NUM_FEATURES

    # Mean-center: removes gravity (accel) and orientation-dependent bias
    ax_s = ay_s = az_s = gx_s = gy_s = gz_s = 0.0
    for r in sensor_data[:samples]:
        ax_s += r[0]; ay_s += r[1]; az_s += r[2]
        gx_s += r[3]; gy_s += r[4]; gz_s += r[5]
    n = float(samples)
    ax_m, ay_m, az_m = ax_s/n, ay_s/n, az_s/n
    gx_m, gy_m, gz_m = gx_s/n, gy_s/n, gz_s/n

    acc_mag = gyro_mag = jerk_mag = []
    acc_mag, gyro_mag, jerk_mag = [], [], []
    prev = sensor_data[0][:3]
    for i in range(samples):
        r = sensor_data[i]
        cax, cay, caz = r[0]-ax_m, r[1]-ay_m, r[2]-az_m
        cgx, cgy, cgz = r[3]-gx_m, r[4]-gy_m, r[5]-gz_m
        acc_mag.append(math.sqrt(cax**2 + cay**2 + caz**2))
        gyro_mag.append(math.sqrt(cgx**2 + cgy**2 + cgz**2))
        if i > 0:
            d = [r[j] - prev[j] for j in range(3)]
            jerk_mag.append(math.sqrt(d[0]**2 + d[1]**2 + d[2]**2))
        prev = r[:3]

    feats = _magnitude_stats(acc_mag) + _magnitude_stats(gyro_mag)

    nj = len(jerk_mag)
    if nj > 0:
        jm = sum(jerk_mag) / nj
        jsq = sum(x*x for x in jerk_mag) / nj
        feats.extend([jm, math.sqrt(max(jsq - jm*jm, 0.0)), max(jerk_mag)])
    else:
        feats.extend([0.0, 0.0, 0.0])
'''
        if include_freq:
            code += '''
    feats.extend(_frequency_features(acc_mag, SAMPLING_RATE))
    feats.extend(_frequency_features(gyro_mag, SAMPLING_RATE))
'''
        code += '''
    while len(feats) < NUM_FEATURES:
        feats.append(0.0)
    return feats[:NUM_FEATURES]
'''
        return code

    # ------------------------------------------------------------------
    # Per-axis extraction (time_domain and all modes)
    # ------------------------------------------------------------------

    def _15_stats_fn(self) -> str:
        """Inline helper for 15 per-axis stats (shares logic with _magnitude_stats)."""
        return '''
def _axis_stats(col):
    """15 stats for one sensor axis column. Matches _magnitude_stats formula."""
    return _magnitude_stats(col)

'''

    def _per_axis_extract(self, include_freq: bool = False) -> str:
        code = self._magnitude_stats_fn()  # re-uses same function for per-axis

        if include_freq:
            code += self._frequency_features_fn()

        code += '''
def extract_features(sensor_data, samples):
    """Per-axis feature extraction (6 axes × 15 stats = 90 time features'''
        if include_freq:
            code += ', plus frequency features'
        code += ''').

    Args:
        sensor_data: list of [aX, aY, aZ, gX, gY, gZ] lists
        samples: int

    Returns:
        list of NUM_FEATURES floats
    """
    if samples <= 1:
        return [0.0] * NUM_FEATURES

    feats = []
    for axis in range(6):
        col = [sensor_data[i][axis] for i in range(samples)]
        feats.extend(_magnitude_stats(col))
'''
        if include_freq:
            code += '''
    for axis in range(6):
        col = [sensor_data[i][axis] for i in range(samples)]
        feats.extend(_frequency_features(col, SAMPLING_RATE))
'''
        code += '''
    while len(feats) < NUM_FEATURES:
        feats.append(0.0)
    return feats[:NUM_FEATURES]
'''
        return code

    # ------------------------------------------------------------------
    # Frequency features helper
    # ------------------------------------------------------------------

    def _frequency_features_fn(self) -> str:
        """DFT-based spectral features matching C++ _per_axis_freq_block.

        Feature order (11 per axis, matching feature_block.py):
          spectral_centroid, spectral_rolloff, spectral_bandwidth,
          dominant_frequency, dominant_frequency_magnitude,
          energy_low_freq, energy_mid_freq, energy_high_freq,
          spectral_rms, spectral_skewness, spectral_kurtosis

        Band thresholds: low [0,5) Hz, mid [5,15) Hz, high [15,fs/2] Hz
        Rolloff threshold: 85% cumulative magnitude
        """
        return '''
def _frequency_features(signal, fs):
    """11 spectral features via DFT. Matches feature_block.py _per_axis_freq_block."""
    n = len(signal)
    half_n = n // 2
    freq_step = fs / n
    two_pi_over_n = 6.283185307 / n
    pi_over_nm1 = 3.141592654 / max(n - 1, 1)

    sig_mean = sum(signal) / n

    dft_mag = []
    max_mag = 0.0
    max_idx = 0
    mag_sum = 0.0
    weighted_freq = 0.0

    for k in range(1, half_n + 1):
        re = im = 0.0
        angle_step = two_pi_over_n * k
        for i in range(n):
            w = 0.5 - 0.5 * math.cos(2.0 * pi_over_nm1 * i)  # Hann
            val = (signal[i] - sig_mean) * w
            angle = angle_step * i
            re += val * math.cos(angle)
            im -= val * math.sin(angle)
        mag = math.sqrt(re*re + im*im)
        dft_mag.append(mag)
        freq = k * freq_step
        if mag > max_mag:
            max_mag = mag
            max_idx = k - 1
        mag_sum += mag
        weighted_freq += freq * mag

    # Spectral centroid
    centroid = weighted_freq / mag_sum if mag_sum > 0 else 0.0

    # Spectral rolloff (85% of cumulative magnitude)
    threshold = 0.85 * mag_sum
    cumsum = 0.0
    rolloff = half_n * freq_step
    for k in range(half_n):
        cumsum += dft_mag[k]
        if cumsum >= threshold:
            rolloff = (k + 1) * freq_step
            break

    # Spectral bandwidth
    bandwidth = 0.0
    if mag_sum > 0:
        for k in range(half_n):
            freq = (k + 1) * freq_step
            bandwidth += (freq - centroid) ** 2 * dft_mag[k]
        bandwidth = math.sqrt(bandwidth / mag_sum)

    # Dominant frequency
    dom_freq = (max_idx + 1) * freq_step
    dom_mag  = max_mag

    # Band energies: low [0,5), mid [5,15), high [15, fs/2]
    e_low = e_mid = e_high = 0.0
    for k in range(half_n):
        freq = (k + 1) * freq_step
        m2 = dft_mag[k] * dft_mag[k]
        if freq < 5.0:   e_low  += m2
        elif freq < 15.0: e_mid += m2
        else:             e_high += m2

    # Spectral RMS
    sq_sum = sum(m*m for m in dft_mag)
    spec_rms = math.sqrt(sq_sum / half_n) if half_n > 0 else 0.0

    # Spectral skewness / kurtosis (population std for z-score normalization)
    spec_mean = mag_sum / half_n if half_n > 0 else 0.0
    spec_var  = sq_sum / half_n - spec_mean * spec_mean if half_n > 0 else 0.0
    spec_std  = math.sqrt(spec_var) if spec_var > 0 else 1e-4

    m3 = m4 = 0.0
    for m in dft_mag:
        z = (m - spec_mean) / (spec_std + 1e-7)
        z2 = z * z
        m3 += z * z2
        m4 += z2 * z2
    nn = half_n
    spec_skew = spec_kurt = 0.0
    if nn >= 3:
        g1 = m3 / nn
        spec_skew = math.sqrt(nn * (nn - 1)) / (nn - 2) * g1
    if nn >= 4:
        g2 = m4 / nn - 3.0
        spec_kurt = (nn - 1) / ((nn - 2) * (nn - 3)) * ((nn + 1) * g2 + 6.0)

    return [centroid, rolloff, bandwidth,
            dom_freq, dom_mag,
            e_low, e_mid, e_high,
            spec_rms, spec_skew, spec_kurt]

'''

    # ------------------------------------------------------------------
    # Raw extraction stub
    # ------------------------------------------------------------------

    def _raw_extract(self) -> str:
        fset = set(self.feature_names)
        axes = [("aX", 0), ("aY", 1), ("aZ", 2), ("gX", 3), ("gY", 4), ("gZ", 5)]
        extracts = []
        for name, idx in axes:
            if name in fset:
                extracts.append(
                    f"    feats.append(sum(r[{idx}] for r in sensor_data[:samples]) / samples)"
                )
        body = "\n".join(extracts) if extracts else "    pass"
        return f'''
def extract_features(sensor_data, samples):
    """Raw mode: mean of each axis over the window."""
    feats = []
{body}
    while len(feats) < NUM_FEATURES:
        feats.append(0.0)
    return feats[:NUM_FEATURES]
'''

    def _cnn_extract_stub(self) -> str:
        return '''
def extract_features(sensor_data, samples):
    """CNN models are not supported on MicroPython."""
    raise NotImplementedError("CNN inference is not supported on MicroPython")
'''

    # ------------------------------------------------------------------
    # get_activity_name
    # ------------------------------------------------------------------

    def _get_activity_name(self) -> str:
        return '''
def get_activity_name(class_id):
    if 0 <= class_id < NUM_CLASSES:
        return ACTIVITY_NAMES[class_id]
    return "unknown"
'''

    # ------------------------------------------------------------------
    # main.py
    # ------------------------------------------------------------------

    def _generate_main(self) -> str:
        module = self.sketch_name.replace("-", "_")
        return f'''\
"""
main.py — MicroPython HAR example
Upload alongside har_model.py (or rename it as required by your board).
Replace the dummy sensor reading with your actual IMU driver.
"""

import time
from har_model import (
    har_init, har_predict, extract_features,
    get_activity_name, WINDOW_SIZE, NUM_FEATURES, SAMPLING_RATE,
)


def read_sensor_window():
    """Read one window of IMU data.

    Returns a list of [aX, aY, aZ, gX, gY, gZ] rows, length WINDOW_SIZE.
    Replace the placeholder below with real I2C / SPI IMU driver calls, e.g.:
        ax, ay, az = imu.acceleration   # m/s²
        gx, gy, gz = imu.gyro           # deg/s
    """
    window = []
    for _ in range(WINDOW_SIZE):
        # TODO: replace with real sensor read
        window.append([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])
        time.sleep_ms(1000 // SAMPLING_RATE)
    return window


def main():
    print("HAR Model ({mt}) — MicroPython".format(mt="{mt}".format(mt="{self.model_type}")))
    print("Features:", NUM_FEATURES, "| Window:", WINDOW_SIZE, "@ ", SAMPLING_RATE, "Hz")

    har_init()

    iteration = 0
    while True:
        sensor_data = read_sensor_window()
        features = extract_features(sensor_data, WINDOW_SIZE)
        predicted, confidence = har_predict(features)
        name = get_activity_name(predicted)
        print("[{{:04d}}] Predicted: {{}} (class {{}}, conf {{:.2f}})".format(
            iteration, name, predicted, confidence))
        iteration += 1


main()
'''

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _fmt_list(self, lst, precision: int = None) -> str:
        prec = precision or self.precision
        if not lst:
            return "[]"
        return "[" + ", ".join(f"{v:.{prec}f}" for v in lst) + "]"

    def _fmt_2d(self, mat, precision: int = None) -> str:
        prec = precision or self.precision
        if not mat:
            return "[]"
        rows = ["[" + ", ".join(f"{v:.{prec}f}" for v in row) + "]" for row in mat]
        inner = ",\n        ".join(rows)
        return f"[\n        {inner}\n    ]"
