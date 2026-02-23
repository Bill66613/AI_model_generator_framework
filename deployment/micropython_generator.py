"""
MicroPython Code Generator
Generates a self-contained MicroPython module for HAR inference on
MicroPython-compatible boards (ESP32, RP2040, STM32, etc.).

Uses only `math` and `array` — no numpy/scipy dependency.
Supports Random Forest, Neural Network (MLP), and SVM (RBF kernel) models.
"""

import math
import numpy as np
from typing import Dict, Any, List
from .base_generator import BaseCodeGenerator


class MicroPythonCodeGenerator(BaseCodeGenerator):
    """Generates a pure-MicroPython HAR inference module.

    Unlike the C/C++ generators that produce header+source+example, this
    generator emits a single ``har_model.py`` module and a separate
    ``main.py`` example script.  The module is entirely self-contained so
    it can be uploaded to the board via ``ampy``, ``mpremote``, or Thonny.
    """

    def __init__(self, model_data: Dict[str, Any], platform: str = 'micropython',
                 optimization: str = 'balanced', overlap: float = 0.5):
        super().__init__(model_data, platform, optimization, overlap)

        self.model_type_name = model_data.get('model_type', 'unknown')

        # Model-specific data
        self.trees = model_data.get('trees', [])
        self.weights = model_data.get('weights', {})
        self.support_vectors = model_data.get('support_vectors', [])
        self.dual_coefficients = model_data.get('dual_coefficients', [])
        self.intercept = model_data.get('intercept', [])
        self.gamma = model_data.get('gamma', 0.1)

        # Neural network specifics
        self._nn_all_weights = []
        self._nn_all_biases = []
        self._nn_hidden_sizes = []
        self._extract_nn_params()

    # -------------------------------------------------------------- #
    # Neural-network parameter extraction (from model object)
    # -------------------------------------------------------------- #

    def _extract_nn_params(self):
        """Extract NN weights from model_object or pre-extracted weights dict."""
        if self.model_type_name != 'neural_network':
            return

        model_obj = self.model_data.get('model_object')
        if model_obj and hasattr(model_obj, 'model') and hasattr(model_obj.model, 'coefs_'):
            coefs = model_obj.model.coefs_
            intercepts = model_obj.model.intercepts_

            # Apply feature reorder to first weight matrix if needed
            reorder = self.model_data.get('_feature_reorder_indices')
            if reorder is not None:
                coefs_0 = coefs[0][reorder, :]
            else:
                coefs_0 = coefs[0]

            self._nn_all_weights = [coefs_0.tolist()] + [c.tolist() for c in coefs[1:]]
            self._nn_all_biases = [b.tolist() for b in intercepts]
            if hasattr(model_obj.model, 'hidden_layer_sizes'):
                hl = model_obj.model.hidden_layer_sizes
                self._nn_hidden_sizes = list(hl) if isinstance(hl, tuple) else [hl]
        elif self.weights:
            # Fallback to pre-extracted dict
            iw = self.weights.get('input_weights', [])
            hb = self.weights.get('hidden_biases', [])
            ow = self.weights.get('output_weights', [])
            ob = self.weights.get('output_biases', [])
            if iw and hb:
                self._nn_all_weights = [iw]
                self._nn_all_biases = [hb]
                if ow:
                    self._nn_all_weights.append(ow)
                if ob:
                    self._nn_all_biases.append(ob)

    # ================================================================ #
    #  Header / Source / Example — MicroPython equivalents              #
    # ================================================================ #

    def generate_header(self) -> str:
        """MicroPython doesn't have a separate header file.

        Return a stub docstring that documents the public API.
        """
        return f'''"""
HAR Model — MicroPython Module (auto-generated)
Model type : {self.model_type_name}
Features   : {len(self.feature_names)}
Classes    : {len(self.classes)}
Optimization: {self.optimization}

Public API
----------
har_init()                        — one-time initialisation (no-op for now)
har_predict(features: list) -> int — predict activity class from feature vector
extract_features(sensor_data, samples) -> list  — extract features from raw sensor window
get_activity_name(class_id) -> str — human-readable activity label
"""
'''

    def generate_implementation(self, header_filename: str = None) -> str:
        """Generate the self-contained har_model.py module."""
        lines: List[str] = []
        lines.append(self._py_header_comment())
        lines.append("import math\n")
        lines.append(self._py_constants())
        lines.append(self._py_activity_names())
        lines.append(self._py_scaling_arrays())
        lines.append(self._py_model_data())
        lines.append(self._py_har_init())
        lines.append(self._py_scale_features())
        lines.append(self._py_predict())
        lines.append(self._py_predict_internal())
        lines.append(self._py_extract_features())
        lines.append(self._py_get_activity_name())
        return "\n".join(lines)

    def generate_example_sketch(self, header_filename: str = None) -> str:
        """Generate a main.py example that imports har_model."""
        module = header_filename.replace('.py', '').replace('.h', '') if header_filename else 'har_model'
        # Strip path and extension to get the import name
        import os
        module = os.path.basename(module).split('.')[0]
        # Ensure it's a valid Python identifier
        module = module.replace('-', '_')

        return f'''"""
HAR Model — MicroPython Example
Upload this file as main.py alongside the generated har_model module.
Replace the dummy sensor reading with your actual IMU driver.
"""

import time
from {module} import (
    har_init, har_predict, extract_features,
    get_activity_name, WINDOW_SIZE, NUM_FEATURES, SAMPLING_RATE,
)


def read_sensor_window():
    """Read one window of IMU data.

    Returns a list of [aX, aY, aZ, gX, gY, gZ] lists, length WINDOW_SIZE.
    Replace the dummy data below with your I2C / SPI IMU driver calls.
    """
    window = []
    for _ in range(WINDOW_SIZE):
        # TODO: Replace with real sensor read, e.g.:
        #   ax, ay, az = imu.acceleration
        #   gx, gy, gz = imu.gyro
        window.append([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])
        time.sleep_ms(1000 // SAMPLING_RATE)
    return window


def main():
    print("HAR Model — {{model}} ({{opt}})".format(
        model="{self.model_type_name.upper()}",
        opt="{self.optimization.title()}"))
    print("Features:", NUM_FEATURES, "| Window:", WINDOW_SIZE, "@ ", SAMPLING_RATE, "Hz")

    har_init()

    for iteration in range(10):  # Change to `while True:` for continuous
        sensor_data = read_sensor_window()
        features = extract_features(sensor_data, WINDOW_SIZE)
        predicted = har_predict(features)
        name = get_activity_name(predicted)
        print("[{{:04d}}] Predicted: {{}} (class {{}})".format(iteration, name, predicted))

    print("Done.")


main()
'''

    # ================================================================ #
    #  Private helpers — Python code generation                        #
    # ================================================================ #

    def _py_header_comment(self) -> str:
        return f'''"""
har_model.py — Auto-generated MicroPython HAR inference module
Model type   : {self.model_type_name}
Features     : {len(self.feature_names)}
Classes      : {len(self.classes)}
Platform     : MicroPython
Optimization : {self.optimization}
"""
'''

    def _py_constants(self) -> str:
        return (
            f"NUM_FEATURES = {len(self.feature_names)}\n"
            f"NUM_CLASSES = {len(self.classes)}\n"
            f"SAMPLING_RATE = {self.sampling_rate}\n"
            f"WINDOW_SIZE = {self.window_size}\n"
        )

    def _py_activity_names(self) -> str:
        names = ', '.join(f'"{c}"' for c in self.classes)
        return f"ACTIVITY_NAMES = [{names}]\n"

    def _py_scaling_arrays(self) -> str:
        prec = self.feature_precision
        means = ', '.join(f'{m:.{prec}f}' for m in self.feature_means)
        stds = ', '.join(f'{s:.{prec}f}' for s in self.feature_stds)
        return (
            f"FEATURE_MEANS = [{means}]\n"
            f"FEATURE_STDS  = [{stds}]\n"
        )

    # --- Model-specific data ---------------------------------------- #

    def _py_model_data(self) -> str:
        if self.model_type_name == 'random_forest':
            return self._py_rf_data()
        elif self.model_type_name == 'neural_network':
            return self._py_nn_data()
        elif self.model_type_name == 'svm':
            return self._py_svm_data()
        return "# No model-specific data\n"

    def _py_rf_data(self) -> str:
        """Serialise decision trees as nested Python lists."""
        if not self.trees:
            return "TREES = []  # No trees extracted\n"

        lines = ["TREES = ["]
        for tree in self.trees:
            fi = tree.get('feature_indices', [])
            th = tree.get('thresholds', [])
            lc = tree.get('left_children', [])
            rc = tree.get('right_children', [])
            vals = tree.get('values', [])
            lines.append("    {")
            lines.append(f"        'fi': {fi},")
            lines.append(f"        'th': {self._fmt_list(th)},")
            lines.append(f"        'lc': {lc},")
            lines.append(f"        'rc': {rc},")
            # For leaf values, extract the predicted class
            leaf_classes = []
            for v in vals:
                if isinstance(v, list) and len(v) > 0:
                    inner = v[0] if isinstance(v[0], list) else v
                    try:
                        leaf_classes.append(int(np.argmax(inner)))
                    except Exception:
                        leaf_classes.append(0)
                else:
                    leaf_classes.append(0)
            lines.append(f"        'pred': {leaf_classes},")
            lines.append("    },")
        lines.append("]\n")
        return "\n".join(lines)

    def _py_nn_data(self) -> str:
        """Serialise neural network weights as Python lists."""
        if not self._nn_all_weights:
            return "NN_WEIGHTS = []\nNN_BIASES = []\n"

        lines = ["NN_WEIGHTS = ["]
        for i, w in enumerate(self._nn_all_weights):
            lines.append(f"    # Layer {i}: {len(w)} x {len(w[0]) if w else 0}")
            lines.append(f"    {self._fmt_2d(w)},")
        lines.append("]\n")

        lines.append("NN_BIASES = [")
        for i, b in enumerate(self._nn_all_biases):
            lines.append(f"    # Layer {i}: {len(b)}")
            lines.append(f"    {self._fmt_list(b)},")
        lines.append("]\n")
        return "\n".join(lines)

    def _py_svm_data(self) -> str:
        """Serialise SVM support vectors, dual coefficients, etc."""
        lines = []
        lines.append(f"SVM_GAMMA = {self.gamma}")

        if self.support_vectors:
            lines.append(f"SVM_NUM_SV = {len(self.support_vectors)}")
            lines.append("SVM_SUPPORT_VECTORS = [")
            for sv in self.support_vectors:
                lines.append(f"    {self._fmt_list(sv)},")
            lines.append("]")
        else:
            lines.append("SVM_NUM_SV = 0")
            lines.append("SVM_SUPPORT_VECTORS = []")

        if self.dual_coefficients:
            lines.append("SVM_DUAL_COEF = [")
            for row in self.dual_coefficients:
                lines.append(f"    {self._fmt_list(row)},")
            lines.append("]")
        else:
            lines.append("SVM_DUAL_COEF = []")

        intercepts = self._fmt_list(self.intercept) if self.intercept else "[]"
        lines.append(f"SVM_INTERCEPT = {intercepts}")
        lines.append("")
        return "\n".join(lines)

    # --- Core inference functions ----------------------------------- #

    def _py_har_init(self) -> str:
        return "\ndef har_init():\n    \"\"\"One-time initialisation (currently a no-op).\"\"\"\n    pass\n"

    def _py_scale_features(self) -> str:
        return '''
def _scale_features(features):
    """Apply StandardScaler: (x - mean) / std."""
    scaled = [0.0] * NUM_FEATURES
    for i in range(NUM_FEATURES):
        std = FEATURE_STDS[i] if FEATURE_STDS[i] > 1e-4 else 1.0
        val = features[i]
        if val != val:  # NaN check (MicroPython doesn't have math.isnan on all ports)
            val = 0.0
        s = (val - FEATURE_MEANS[i]) / std
        # Clamp to [-10, 10]
        if s < -10.0:
            s = -10.0
        elif s > 10.0:
            s = 10.0
        scaled[i] = s
    return scaled
'''

    def _py_predict(self) -> str:
        return '''
def har_predict(features):
    """Predict activity class from a raw (unscaled) feature vector.

    Args:
        features: list of NUM_FEATURES floats

    Returns:
        int — predicted class index
    """
    scaled = _scale_features(features)
    result = _predict_internal(scaled)
    if result < 0 or result >= NUM_CLASSES:
        return 0
    return result
'''

    def _py_predict_internal(self) -> str:
        if self.model_type_name == 'random_forest':
            return self._py_rf_predict()
        elif self.model_type_name == 'neural_network':
            return self._py_nn_predict()
        elif self.model_type_name == 'svm':
            return self._py_svm_predict()
        return (
            "\ndef _predict_internal(features):\n"
            "    return 0  # No model loaded\n"
        )

    def _py_rf_predict(self) -> str:
        return '''
def _predict_tree(tree, features):
    """Traverse a single decision tree."""
    fi = tree['fi']
    th = tree['th']
    lc = tree['lc']
    rc = tree['rc']
    pred = tree['pred']
    node = 0
    while fi[node] >= 0:
        if features[fi[node]] <= th[node]:
            node = lc[node]
        else:
            node = rc[node]
    return pred[node]


def _predict_internal(features):
    """Random Forest — majority vote across all trees."""
    votes = [0] * NUM_CLASSES
    for tree in TREES:
        cls = _predict_tree(tree, features)
        if 0 <= cls < NUM_CLASSES:
            votes[cls] += 1
    best = 0
    for i in range(1, NUM_CLASSES):
        if votes[i] > votes[best]:
            best = i
    return best
'''

    def _py_nn_predict(self) -> str:
        return '''
def _relu(x):
    return x if x > 0 else 0.0


def _predict_internal(features):
    """Neural Network — forward pass through all layers."""
    current = list(features)
    num_layers = len(NN_WEIGHTS)
    for layer_idx in range(num_layers):
        w = NN_WEIGHTS[layer_idx]
        b = NN_BIASES[layer_idx]
        out_size = len(b)
        next_layer = [0.0] * out_size
        for j in range(out_size):
            s = b[j]
            for i in range(len(current)):
                s += current[i] * w[i][j]
            # ReLU for hidden layers, linear for output layer
            if layer_idx < num_layers - 1:
                s = _relu(s)
            next_layer[j] = s
        current = next_layer

    # Argmax
    best = 0
    for i in range(1, len(current)):
        if current[i] > current[best]:
            best = i
    return best
'''

    def _py_svm_predict(self) -> str:
        return '''
def _rbf_kernel(x1, x2, gamma):
    """RBF kernel: K(x,y) = exp(-gamma * ||x-y||^2)."""
    s = 0.0
    for i in range(len(x1)):
        d = x1[i] - x2[i]
        s += d * d
    exponent = -gamma * s
    if exponent < -80.0:
        return 0.0
    return math.exp(exponent)


def _predict_internal(features):
    """SVM — One-vs-Rest with RBF kernel."""
    scores = [0.0] * NUM_CLASSES
    for sv_idx in range(SVM_NUM_SV):
        k = _rbf_kernel(features, SVM_SUPPORT_VECTORS[sv_idx], SVM_GAMMA)
        for cls in range(NUM_CLASSES):
            scores[cls] += SVM_DUAL_COEF[cls][sv_idx] * k
    for cls in range(NUM_CLASSES):
        scores[cls] += SVM_INTERCEPT[cls]
    best = 0
    for i in range(1, NUM_CLASSES):
        if scores[i] > scores[best]:
            best = i
    return best
'''

    # --- Feature extraction ----------------------------------------- #

    def _py_extract_features(self) -> str:
        """Generate feature extraction matching the C++ version exactly."""
        # Check if orientation-robust
        orientation_robust = False
        include_frequency = False
        feature_names = self.feature_names or []
        has_acc_mag = any(str(n).startswith('acc_mag_') for n in feature_names)
        has_gyro_mag = any(str(n).startswith('gyro_mag_') for n in feature_names)
        if has_acc_mag and has_gyro_mag:
            orientation_robust = True

        # Auto-detect frequency features from names
        freq_suffixes = ('_dominant_frequency', '_spectral_centroid',
                         '_energy_low_freq', '_spectral_rolloff')
        if any(str(n).endswith(freq_suffixes) for n in feature_names):
            include_frequency = True

        # Also check feature_config
        try:
            fc = self.model_data.get('model_info', {}).get('feature_config', {})
            if fc.get('orientation_robust', False):
                orientation_robust = True
            if fc.get('include_frequency', False):
                include_frequency = True
        except Exception:
            pass

        if orientation_robust:
            return self._py_orientation_robust_extraction(include_frequency)
        return self._py_per_axis_extraction()

    def _py_orientation_robust_extraction(self, include_frequency=False) -> str:
        code = '''
def _magnitude_stats(mag):
    """Extract 15 statistical features from a magnitude vector."""
    n = len(mag)
    s = sum(mag)
    ssq = sum(x * x for x in mag)
    mean = s / n
    var = ssq / n - mean * mean
    if var < 0:
        var = 0.0
    std = math.sqrt(var)
    rms = math.sqrt(ssq / n)
    energy = ssq
    mn = min(mag)
    mx = max(mag)
    rng = mx - mn

    # Sort for median/quartiles
    s_data = sorted(mag)
    mid = n // 2
    median = (s_data[mid - 1] + s_data[mid]) / 2 if n % 2 == 0 else s_data[mid]
    q25 = s_data[n // 4]
    q75 = s_data[(3 * n) // 4]
    iqr = q75 - q25

    # Skewness/kurtosis (bias-corrected)
    sample_var = var * n / (n - 1 + 0.001)
    sample_std = math.sqrt(sample_var) if sample_var > 0 else 0.0001
    m3 = 0.0
    m4 = 0.0
    for v in mag:
        z = (v - mean) / (sample_std + 0.0001)
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
    zc = 0
    mc = 0
    for i in range(1, n):
        if mag[i - 1] * mag[i] < 0:
            zc += 1
        if (mag[i - 1] - mean) * (mag[i] - mean) < 0:
            mc += 1

    return [mean, std, mn, mx, rng, median, q25, q75, iqr,
            skew, kurt, rms, energy, float(zc), mc / n]

'''
        # Add DFT frequency feature extraction if needed
        if include_frequency:
            code += '''
def _frequency_features(signal, sampling_rate):
    """Extract 7 frequency features via DFT (matches C extract_frequency_features)."""
    n = len(signal)
    half_n = n // 2
    freq_step = sampling_rate / n
    two_pi_over_n = 6.283185307 / n

    max_mag = 0.0
    max_idx = 0
    mag_sum = 0.0
    weighted_freq_sum = 0.0
    dft_mag = []

    for k in range(1, half_n + 1):
        re = 0.0
        im = 0.0
        angle_step = two_pi_over_n * k
        for i in range(n):
            angle = angle_step * i
            re += signal[i] * math.cos(angle)
            im -= signal[i] * math.sin(angle)
        mag = math.sqrt(re * re + im * im)
        dft_mag.append(mag)

        freq = k * freq_step
        if mag > max_mag:
            max_mag = mag
            max_idx = k - 1
        mag_sum += mag
        weighted_freq_sum += freq * mag

    feats = []
    # 0: Dominant frequency
    feats.append((max_idx + 1) * freq_step)
    # 1: Dominant frequency magnitude
    feats.append(max_mag)
    # 2: Spectral centroid
    feats.append(weighted_freq_sum / mag_sum if mag_sum > 0 else 0.0)

    # Energy in bands: 0-2 Hz, 2-5 Hz, 5+ Hz
    e_low = 0.0
    e_mid = 0.0
    e_high = 0.0
    for k in range(half_n):
        freq = (k + 1) * freq_step
        e = dft_mag[k] * dft_mag[k]
        if freq < 2.0:
            e_low += e
        elif freq < 5.0:
            e_mid += e
        else:
            e_high += e
    feats.append(e_low)
    feats.append(e_mid)
    feats.append(e_high)

    # 6: Spectral rolloff (85%)
    threshold = 0.85 * mag_sum
    cumsum = 0.0
    rolloff = half_n * freq_step
    for k in range(half_n):
        cumsum += dft_mag[k]
        if cumsum >= threshold:
            rolloff = (k + 1) * freq_step
            break
    feats.append(rolloff)

    return feats

'''

        code += '''
def extract_features(sensor_data, samples):
    """Orientation-robust feature extraction.

    Args:
        sensor_data: list of [aX, aY, aZ, gX, gY, gZ] lists
        samples: number of samples (== len(sensor_data))

    Returns:
        list of NUM_FEATURES floats
    """
    if samples <= 1:
        return [0.0] * NUM_FEATURES

    acc_mag = []
    gyro_mag = []
    jerk_mag = []
    prev = sensor_data[0][:3]
    for i in range(samples):
        row = sensor_data[i]
        am = math.sqrt(row[0] ** 2 + row[1] ** 2 + row[2] ** 2)
        gm = math.sqrt(row[3] ** 2 + row[4] ** 2 + row[5] ** 2)
        acc_mag.append(am)
        gyro_mag.append(gm)
        if i > 0:
            d = [row[j] - prev[j] for j in range(3)]
            jerk_mag.append(math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2))
        prev = row[:3]

    feats = _magnitude_stats(acc_mag) + _magnitude_stats(gyro_mag)

    # Jerk features (mean, std, max)
    nj = len(jerk_mag)
    if nj > 0:
        jm = sum(jerk_mag) / nj
        jsq = sum(x * x for x in jerk_mag) / nj
        jvar = jsq - jm * jm
        feats.append(jm)
        feats.append(math.sqrt(max(jvar, 0.0)))
        feats.append(max(jerk_mag))
    else:
        feats.extend([0.0, 0.0, 0.0])
'''
        # Add frequency feature extraction call if needed
        if include_frequency:
            code += '''
    # Frequency features (7 per magnitude via DFT)
    feats.extend(_frequency_features(acc_mag, SAMPLING_RATE))
    feats.extend(_frequency_features(gyro_mag, SAMPLING_RATE))
'''

        code += '''
    # Pad / truncate to NUM_FEATURES
    while len(feats) < NUM_FEATURES:
        feats.append(0.0)
    return feats[:NUM_FEATURES]
'''
        return code

    def _py_per_axis_extraction(self) -> str:
        return '''
def extract_features(sensor_data, samples):
    """Per-axis feature extraction (15 features x 6 axes = 90 features).

    Args:
        sensor_data: list of [aX, aY, aZ, gX, gY, gZ] lists
        samples: number of samples (== len(sensor_data))

    Returns:
        list of NUM_FEATURES floats
    """
    if samples <= 1:
        return [0.0] * NUM_FEATURES

    feats = []
    n = float(samples)
    for axis in range(6):
        col = [sensor_data[i][axis] for i in range(samples)]
        s = sum(col)
        ssq = sum(x * x for x in col)
        mean = s / n
        var = ssq / n - mean * mean
        if var < 0:
            var = 0.0
        std = math.sqrt(var)

        s_data = sorted(col)
        mid = samples // 2
        median = (s_data[mid - 1] + s_data[mid]) / 2 if samples % 2 == 0 else s_data[mid]
        q25 = s_data[samples // 4]
        q75 = s_data[(3 * samples) // 4]
        iqr = q75 - q25

        # Skewness / kurtosis
        sample_std = math.sqrt(var * n / (n - 1 + 0.001))
        m3 = 0.0
        m4 = 0.0
        for v in col:
            z = (v - mean) / (sample_std + 0.001)
            z2 = z * z
            m3 += z * z2
            m4 += z2 * z2
        g1 = m3 / n
        skew = 0.0
        if samples >= 3:
            skew = math.sqrt(n * (n - 1)) / (n - 2) * g1
        g2 = m4 / n - 3.0
        kurt = -3.0
        if samples >= 4:
            kurt = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6.0)

        rms = math.sqrt(ssq / n)
        energy = ssq

        # Zero crossings
        zc = 0
        for i in range(1, samples):
            if col[i - 1] * col[i] < 0:
                zc += 1

        # Mean crossing rate
        mc = 0
        for i in range(1, samples):
            if (col[i - 1] - mean) * (col[i] - mean) < 0:
                mc += 1

        feats.extend([
            mean, std, min(col), max(col), max(col) - min(col),
            median, q25, q75, iqr,
            skew, kurt, rms, energy,
            float(zc), mc / n,
        ])

    while len(feats) < NUM_FEATURES:
        feats.append(0.0)
    return feats[:NUM_FEATURES]
'''

    def _py_get_activity_name(self) -> str:
        return '''
def get_activity_name(class_id):
    """Return human-readable activity label."""
    if 0 <= class_id < NUM_CLASSES:
        return ACTIVITY_NAMES[class_id]
    return "UNKNOWN"
'''

    # ================================================================ #
    #  Abstract method stubs — not used for MicroPython                #
    # ================================================================ #

    def _get_model_specific_declarations(self) -> str:
        return "// MicroPython — no C declarations"

    def _generate_model_specific_implementation(self) -> str:
        return "// MicroPython — no C implementation"

    def _generate_prediction_function(self) -> str:
        return "// MicroPython — no C prediction function"

    def _generate_utility_functions(self) -> str:
        return "// MicroPython — no C utility functions"

    # ================================================================ #
    #  Formatting helpers                                               #
    # ================================================================ #

    def _fmt_list(self, lst, precision=6) -> str:
        """Format a 1-D numeric list as a Python literal."""
        if not lst:
            return "[]"
        items = ', '.join(f'{v:.{precision}f}' for v in lst)
        return f'[{items}]'

    def _fmt_2d(self, mat, precision=6) -> str:
        """Format a 2-D list as a Python literal."""
        if not mat:
            return "[]"
        rows = []
        for row in mat:
            items = ', '.join(f'{v:.{precision}f}' for v in row)
            rows.append(f'[{items}]')
        inner = ',\n        '.join(rows)
        return f'[\n        {inner}\n    ]'
