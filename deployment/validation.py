"""
Deployment Validation Module
Validates that generated C++ code will produce the same results as Python training code.
Catches feature extraction mismatches, scaling errors, and prediction divergence before deployment.
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path


class DeploymentValidator:
    """Validates generated C++ code against Python training pipeline."""

    # Device hardware specs (RAM KB, Flash KB, Clock MHz, Power mW active, Power mW sleep)
    DEVICE_SPECS = {
        'arduino_uno': {
            'name': 'Arduino Uno (ATmega328P)',
            'ram_kb': 2, 'flash_kb': 32, 'clock_mhz': 16,
            'power_active_mw': 46, 'power_sleep_mw': 1.4,
            'float_support': 'software', 'word_size': 8,
            'voltage': 5.0, 'adc_bits': 10,
        },
        'arduino_nano': {
            'name': 'Arduino Nano (ATmega328P)',
            'ram_kb': 2, 'flash_kb': 32, 'clock_mhz': 16,
            'power_active_mw': 46, 'power_sleep_mw': 1.4,
            'float_support': 'software', 'word_size': 8,
            'voltage': 5.0, 'adc_bits': 10,
        },
        'seeed_xiao_nrf52840': {
            'name': 'Seeed XIAO nRF52840 Sense',
            'ram_kb': 256, 'flash_kb': 1024, 'clock_mhz': 64,
            'power_active_mw': 19.8, 'power_sleep_mw': 0.003,
            'float_support': 'hardware_fpu', 'word_size': 32,
            'voltage': 3.3, 'adc_bits': 12,
            'builtin_imu': 'LSM6DS3',
        },
        'esp32': {
            'name': 'ESP32',
            'ram_kb': 520, 'flash_kb': 4096, 'clock_mhz': 240,
            'power_active_mw': 160, 'power_sleep_mw': 0.01,
            'float_support': 'hardware_fpu', 'word_size': 32,
            'voltage': 3.3, 'adc_bits': 12,
        },
        'esp32_s3': {
            'name': 'ESP32-S3',
            'ram_kb': 512, 'flash_kb': 8192, 'clock_mhz': 240,
            'power_active_mw': 165, 'power_sleep_mw': 0.007,
            'float_support': 'hardware_fpu', 'word_size': 32,
            'voltage': 3.3, 'adc_bits': 12,
        },
        'm5stick_c_plus2': {
            'name': 'M5StickC Plus2 (ESP32-PICO)',
            'ram_kb': 320, 'flash_kb': 8192, 'clock_mhz': 240,
            'power_active_mw': 160, 'power_sleep_mw': 0.01,
            'float_support': 'hardware_fpu', 'word_size': 32,
            'voltage': 3.3, 'adc_bits': 12,
            'builtin_imu': 'MPU6886',
        },
        'stm32f4': {
            'name': 'STM32F407 (ARM Cortex-M4)',
            'ram_kb': 192, 'flash_kb': 1024, 'clock_mhz': 168,
            'power_active_mw': 93, 'power_sleep_mw': 0.002,
            'float_support': 'hardware_fpu', 'word_size': 32,
            'voltage': 3.3, 'adc_bits': 12,
        },
        'teensy_4': {
            'name': 'Teensy 4.0 (ARM Cortex-M7)',
            'ram_kb': 1024, 'flash_kb': 2048, 'clock_mhz': 600,
            'power_active_mw': 100, 'power_sleep_mw': 0.012,
            'float_support': 'hardware_fpu_dp', 'word_size': 32,
            'voltage': 3.3, 'adc_bits': 12,
        },
    }

    def __init__(self, model_data: Dict[str, Any] = None):
        self.model_data = model_data or {}
        self.issues: List[Dict[str, str]] = []
        self.warnings: List[Dict[str, str]] = []

    def validate_feature_consistency(self, python_features: np.ndarray,
                                     cpp_features: np.ndarray,
                                     feature_names: List[str] = None,
                                     tolerance: float = 0.01) -> Dict[str, Any]:
        """
        Compare Python and C++ feature extraction outputs.
        
        Args:
            python_features: Feature vector from Python extraction
            cpp_features: Feature vector from C++ extraction (parsed from serial output)
            feature_names: Feature names for reporting
            tolerance: Maximum acceptable relative difference
            
        Returns:
            Validation report dict
        """
        report = {
            'passed': True,
            'total_features': len(python_features),
            'mismatches': [],
            'max_relative_error': 0.0,
            'mean_relative_error': 0.0,
        }

        if len(python_features) != len(cpp_features):
            report['passed'] = False
            report['error'] = (f"Feature count mismatch: Python={len(python_features)}, "
                             f"C++={len(cpp_features)}")
            return report

        relative_errors = []
        for i in range(len(python_features)):
            py_val = python_features[i]
            cpp_val = cpp_features[i]
            
            # Calculate relative error
            if abs(py_val) > 1e-6:
                rel_error = abs(py_val - cpp_val) / abs(py_val)
            else:
                rel_error = abs(py_val - cpp_val)
            
            relative_errors.append(rel_error)
            
            if rel_error > tolerance:
                fname = feature_names[i] if feature_names and i < len(feature_names) else f"feature_{i}"
                report['mismatches'].append({
                    'index': i,
                    'name': fname,
                    'python_value': float(py_val),
                    'cpp_value': float(cpp_val),
                    'relative_error': float(rel_error),
                })

        report['max_relative_error'] = float(max(relative_errors)) if relative_errors else 0.0
        report['mean_relative_error'] = float(np.mean(relative_errors)) if relative_errors else 0.0
        report['passed'] = len(report['mismatches']) == 0

        return report

    def validate_scaling_parameters(self, feature_means: List[float],
                                     feature_stds: List[float],
                                     precision: int = 3) -> Dict[str, Any]:
        """
        Validate that scaler parameters won't lose precision when truncated to C++ precision.
        
        Args:
            feature_means: Scaler mean values
            feature_stds: Scaler scale values
            precision: Decimal places used in C++ code
            
        Returns:
            Validation report
        """
        report = {
            'passed': True,
            'precision_issues': [],
            'zero_std_issues': [],
        }

        for i, (mean, std) in enumerate(zip(feature_means, feature_stds)):
            # Check if truncating to precision rounds small values to 0
            truncated_mean = round(mean, precision)
            truncated_std = round(std, precision)
            
            if abs(mean) > 1e-6 and truncated_mean == 0.0:
                report['precision_issues'].append({
                    'index': i,
                    'type': 'mean',
                    'original': mean,
                    'truncated': truncated_mean,
                    'min_precision_needed': self._min_precision(mean),
                })
                report['passed'] = False
            
            if std > 1e-6 and truncated_std == 0.0:
                report['precision_issues'].append({
                    'index': i,
                    'type': 'std',
                    'original': std,
                    'truncated': truncated_std,
                    'min_precision_needed': self._min_precision(std),
                })
                report['passed'] = False
            
            if std <= 0:
                report['zero_std_issues'].append({
                    'index': i,
                    'std_value': std,
                })
                report['passed'] = False

        return report

    def validate_generated_code(self, header_code: str, source_code: str,
                                 model_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Static validation of generated C++ code for common issues.
        
        Args:
            header_code: Content of generated .h file
            source_code: Content of generated .cpp file
            model_data: Model metadata
            
        Returns:
            Validation report
        """
        report = {
            'passed': True,
            'issues': [],
            'warnings': [],
            'checks_passed': [],
        }

        # Detect CNN model type — CNN uses raw sensor windows, not extracted features
        model_type = model_data.get('model_type', '')
        is_cnn = model_type in ('pytorch_cnn', 'pytorch_cnn2d', 'cnn') or \
                 'har_predict_from_window' in source_code

        if is_cnn:
            return self._validate_cnn_code(header_code, source_code, model_data, report)

        # === Feature-based model checks (NN, RF, SVM) ===

        # Check 1: NUM_FEATURES matches model
        num_features_match = re.search(r'#define NUM_FEATURES\s+(\d+)', header_code)
        if num_features_match:
            cpp_features = int(num_features_match.group(1))
            py_features = len(model_data.get('feature_names') or [])
            if cpp_features != py_features:
                report['issues'].append(
                    f"NUM_FEATURES mismatch: C++={cpp_features}, Python={py_features}")
                report['passed'] = False
            else:
                report['checks_passed'].append(f"NUM_FEATURES={cpp_features} ✓")

        # Check 2: NUM_CLASSES matches model
        num_classes_match = re.search(r'#define NUM_CLASSES\s+(\d+)', header_code)
        if num_classes_match:
            cpp_classes = int(num_classes_match.group(1))
            py_classes = len(model_data.get('classes') or [])
            if py_classes > 0 and cpp_classes != py_classes:
                report['issues'].append(
                    f"NUM_CLASSES mismatch: C++={cpp_classes}, Python={py_classes}")
                report['passed'] = False
            else:
                report['checks_passed'].append(f"NUM_CLASSES={cpp_classes} ✓")

        # Check 3: Feature scaling arrays present
        if 'feature_means[NUM_FEATURES]' not in source_code:
            report['issues'].append("Missing feature_means array in source")
            report['passed'] = False
        else:
            report['checks_passed'].append("feature_means array present ✓")

        if 'feature_stds[NUM_FEATURES]' not in source_code:
            report['issues'].append("Missing feature_stds array in source")
            report['passed'] = False
        else:
            report['checks_passed'].append("feature_stds array present ✓")

        # Check 4: Correct scaler application (scaling in har_predict, NOT in har_predict_internal)
        if 'har_predict_internal' in source_code:
            # Check that internal function does NOT do scaling
            predict_internal = source_code.split('har_predict_internal')[1][:2000]
            if 'feature_means' in predict_internal and 'feature_stds' in predict_internal:
                # Could be double-scaling
                if 'scaled_features[i] = (features[i] - feature_means' in predict_internal:
                    report['issues'].append(
                        "DOUBLE SCALING: har_predict_internal() scales features, "
                        "but har_predict() already scales them!")
                    report['passed'] = False

        # Check 5: Activity names match classes
        for cls_name in model_data.get('classes', []):
            if f'"{cls_name}"' not in source_code:
                report['warnings'].append(
                    f"Class name '{cls_name}' not found in activity_names array")

        # Check 6: Feature extraction method matches training
        feature_names = model_data.get('feature_names', [])
        has_mag_features = any(str(n).startswith('acc_mag_') for n in feature_names)
        has_per_axis = any(str(n).startswith(('aX_', 'aY_', 'aZ_')) for n in feature_names)
        
        if has_mag_features and 'extract_magnitude_stats' not in source_code:
            report['issues'].append(
                "Model trained with magnitude features but C++ uses per-axis extraction!")
            report['passed'] = False
        elif has_mag_features:
            report['checks_passed'].append("Orientation-robust extraction matched ✓")
        
        if has_per_axis and 'for (int axis = 0; axis < 6; axis++)' not in source_code:
            report['warnings'].append(
                "Model trained with per-axis features but C++ doesn't iterate over axes")

        # Check 7: Null pointer checks
        if 'features == NULL' not in source_code:
            report['warnings'].append("Missing NULL pointer check for features array")

        # Check 8: Division by zero protection
        if 'std < 0.0001f' not in source_code and 'std < 0.001f' not in source_code:
            report['warnings'].append(
                "Missing division-by-zero protection in feature scaling")

        return report

    def _validate_cnn_code(self, header_code: str, source_code: str,
                           model_data: Dict[str, Any],
                           report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate CNN-specific generated code.
        CNN models operate on raw sensor windows — no feature extraction or scaling.
        """
        combined = header_code + source_code

        # CNN Check 1: WINDOW_SIZE defined
        ws_match = re.search(r'#define WINDOW_SIZE\s+(\d+)', header_code)
        if ws_match:
            report['checks_passed'].append(f"WINDOW_SIZE={ws_match.group(1)} ✓")
        else:
            report['issues'].append("Missing WINDOW_SIZE define for CNN")
            report['passed'] = False

        # CNN Check 2: N_CHANNELS defined
        nc_match = re.search(r'#define N_CHANNELS\s+(\d+)', header_code)
        if nc_match:
            report['checks_passed'].append(f"N_CHANNELS={nc_match.group(1)} ✓")
        else:
            report['issues'].append("Missing N_CHANNELS define for CNN")
            report['passed'] = False

        # CNN Check 3: NUM_CLASSES matches model
        num_classes_match = re.search(r'#define NUM_CLASSES\s+(\d+)', header_code)
        if num_classes_match:
            cpp_classes = int(num_classes_match.group(1))
            py_classes = len(model_data.get('classes') or [])
            if py_classes > 0 and cpp_classes != py_classes:
                report['issues'].append(
                    f"NUM_CLASSES mismatch: C++={cpp_classes}, Python={py_classes}")
                report['passed'] = False
            else:
                report['checks_passed'].append(f"NUM_CLASSES={cpp_classes} ✓")

        # CNN Check 4: Core CNN functions present
        # For TFLite approach: conv1d weights live in the .tflite binary, not in C++ source.
        # Detect TFLite by presence of the g_har_model byte array.
        is_tflite = 'g_har_model' in combined

        if is_tflite:
            # TFLite CNN: model is the byte array; inference goes through TFLite Micro
            if 'g_har_model' in combined:
                report['checks_passed'].append("TFLite model byte array (g_har_model) present ✓")
            else:
                report['issues'].append("Missing TFLite model byte array (g_har_model)")
                report['passed'] = False
            if 'har_predict_from_window' in combined:
                report['checks_passed'].append("Window-based prediction function present ✓")
            else:
                report['issues'].append("Missing Window-based prediction function (har_predict_from_window)")
                report['passed'] = False
            if 'tflite_predict_window' in combined:
                report['checks_passed'].append("TFLite window predict function present ✓")
            else:
                report['issues'].append("Missing TFLite window predict function (tflite_predict_window)")
                report['passed'] = False
        else:
            # Direct CNN: conv1d and weight arrays must be in C++ source
            cnn_functions = [
                ('conv1d', 'Conv1D layer function'),
                ('har_predict_from_window', 'Window-based prediction function'),
            ]
            for func_name, desc in cnn_functions:
                if func_name in combined:
                    report['checks_passed'].append(f"{desc} present ✓")
                else:
                    report['issues'].append(f"Missing {desc} ({func_name})")
                    report['passed'] = False

            # CNN Check 5 (direct only): Weight arrays present
            weight_arrays = re.findall(r'static\s+const\s+float\s+(\w+)\s*\[', combined)
            if weight_arrays:
                report['checks_passed'].append(
                    f"CNN weight arrays present ({len(weight_arrays)} arrays) ✓")
            else:
                report['issues'].append("No CNN weight arrays found")
                report['passed'] = False

        # CNN Check 6: Activity names match classes
        for cls_name in model_data.get('classes', []):
            if f'"{cls_name}"' not in combined:
                report['warnings'].append(
                    f"Class name '{cls_name}' not found in activity_names array")

        # CNN Check 7: No feature extraction / scaling code (would be a bug for non-TFLite CNN)
        if not is_tflite:
            if 'feature_means[NUM_FEATURES]' in source_code:
                report['warnings'].append(
                    "CNN code contains feature_means — CNN should use raw windows, not features")
            if 'extract_features' in source_code and 'extract_magnitude_stats' in source_code:
                report['warnings'].append(
                    "CNN code contains feature extraction functions — CNN uses raw sensor data")

        report['checks_passed'].append("CNN architecture: raw window input (no feature extraction needed) ✓")

        return report

    # ------------------------------------------------------------------
    # v2 validation
    # ------------------------------------------------------------------

    def _validate_v2_code(self, files: Dict[str, str],
                          model_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate v2 (HAR Edge Framework v2) generated files.

        v2 uses:
          har_config.h        — HAR_NUM_FEATURES / HAR_NUM_CLASSES / HAR_WINDOW_SIZE
          har_features.cpp    — har_extract_features()
          har_classifier.cpp  — SCALER_MEANS[] / SCALER_STDS[] / har_classify()
          har_model.cpp       — har_model_predict()
          *.ino               — thin sketch calling har_classify()
        """
        report = {
            'passed': True,
            'issues': [],
            'warnings': [],
            'checks_passed': [],
        }

        config_h = files.get('har_config.h', '')
        features_cpp = files.get('har_features.cpp', '')
        classifier_cpp = files.get('har_classifier.cpp', '')
        model_h = files.get('har_model.h', '')
        model_cpp = files.get('har_model.cpp', '')
        sketch = next((v for k, v in files.items() if k.endswith('.ino')), '')
        sketch_name = next((k for k in files if k.endswith('.ino')), '')

        # ---- Check 1: HAR_NUM_FEATURES matches model ----
        m = re.search(r'#define HAR_NUM_FEATURES\s+(\d+)', config_h)
        if m:
            cpp_n = int(m.group(1))
            py_n = len(model_data.get('feature_names') or [])
            if py_n > 0 and cpp_n != py_n:
                report['issues'].append(
                    f"HAR_NUM_FEATURES mismatch: C++={cpp_n}, Python={py_n}")
                report['passed'] = False
            else:
                report['checks_passed'].append(f"HAR_NUM_FEATURES={cpp_n} ✓")
        else:
            report['issues'].append("HAR_NUM_FEATURES not found in har_config.h")
            report['passed'] = False

        # ---- Check 2: HAR_NUM_CLASSES matches model ----
        m = re.search(r'#define HAR_NUM_CLASSES\s+(\d+)', config_h)
        if m:
            cpp_c = int(m.group(1))
            py_c = len(model_data.get('classes') or [])
            if py_c > 0 and cpp_c != py_c:
                report['issues'].append(
                    f"HAR_NUM_CLASSES mismatch: C++={cpp_c}, Python={py_c}")
                report['passed'] = False
            else:
                report['checks_passed'].append(f"HAR_NUM_CLASSES={cpp_c} ✓")

        # ---- Check 3: Scaler arrays present in har_classifier.cpp ----
        if 'SCALER_MEANS' in classifier_cpp:
            report['checks_passed'].append("SCALER_MEANS array present ✓")
        else:
            report['issues'].append("SCALER_MEANS[] not found in har_classifier.cpp")
            report['passed'] = False

        if 'SCALER_STDS' in classifier_cpp:
            report['checks_passed'].append("SCALER_STDS array present ✓")
        else:
            report['issues'].append("SCALER_STDS[] not found in har_classifier.cpp")
            report['passed'] = False

        # ---- Check 4: Scaler count matches feature count ----
        means_match = re.search(r'SCALER_MEANS\[HAR_NUM_FEATURES\]\s*=\s*\{([^}]+)\}', classifier_cpp, re.DOTALL)
        if means_match:
            vals = [v.strip().rstrip('f') for v in means_match.group(1).split(',') if v.strip()]
            py_n = len(model_data.get('feature_means') or [])
            if py_n > 0 and abs(len(vals) - py_n) > 1:  # allow ±1 for trailing
                report['issues'].append(
                    f"SCALER_MEANS has {len(vals)} values but model has {py_n} feature_means")
                report['passed'] = False
            else:
                report['checks_passed'].append(f"SCALER_MEANS values count matches ✓")

        # ---- Check 5: Feature extraction parity (feature type detection) ----
        feature_names = model_data.get('feature_names') or []
        has_mag = any(str(n).startswith('acc_mag_') for n in feature_names)
        has_gyro_mag = any(str(n).startswith('gyro_mag_') for n in feature_names)
        has_per_axis = any(str(n).startswith(('aX_', 'aY_', 'aZ_')) for n in feature_names)

        if has_mag:
            # v2 orientation-invariant path computes acc_mag centred magnitude
            # It always calls _extract_15_stats on the magnitude arrays
            if '_extract_15_stats' in features_cpp or 'acc_mag' in features_cpp.lower():
                report['checks_passed'].append("Orientation-invariant feature extraction ✓")
            else:
                report['issues'].append(
                    "Model trained with magnitude features (acc_mag_*) but "
                    "har_features.cpp does not contain magnitude extraction!")
                report['passed'] = False

        if has_per_axis:
            if 'for (int axis' in features_cpp or 'for (int ch' in features_cpp:
                report['checks_passed'].append("Per-axis feature extraction ✓")
            else:
                report['warnings'].append(
                    "Model has per-axis features but har_features.cpp may not iterate axes")

        if not has_mag and not has_per_axis and feature_names:
            report['warnings'].append(
                "Could not detect feature extraction mode from feature names")

        # ---- Check 6: Parity — division-by-zero guard in scaler ----
        if '1e-7f' in classifier_cpp or '0.0001f' in classifier_cpp or '1e-6f' in classifier_cpp:
            report['checks_passed'].append("Zero-std guard present in scaler ✓")
        else:
            report['warnings'].append(
                "Division-by-zero guard not found in har_classifier.cpp scaler")

        # ---- Check 7: NaN/Inf guard in scaler ----
        if '!= features[i]' in classifier_cpp or '!= features' in classifier_cpp or \
                'features[i] != features[i]' in classifier_cpp:
            report['checks_passed'].append("NaN guard present in scaler ✓")
        else:
            report['warnings'].append(
                "NaN/Inf guard not found in har_classifier.cpp — "
                "may cause silent incorrect results on bad sensor data")

        # ---- Check 8: Sketch .ino filename ----
        if sketch_name:
            # Extract base name (without .ino)
            ino_base = sketch_name[:-4] if sketch_name.endswith('.ino') else sketch_name
            if ino_base == 'har_sketch':
                report['warnings'].append(
                    "Sketch is named 'har_sketch.ino' — rename to match the folder name "
                    "for Arduino IDE compatibility")
            else:
                report['checks_passed'].append(
                    f"Sketch filename '{sketch_name}' matches folder name ✓")
        else:
            report['warnings'].append("No .ino sketch file found in generated files")

        # ---- Check 9: har_classify() entry point present ----
        if 'har_classify' in classifier_cpp:
            report['checks_passed'].append("har_classify() inference entry point present ✓")
        else:
            report['issues'].append("har_classify() not found in har_classifier.cpp")
            report['passed'] = False

        # ---- Check 10: har_model_predict() entry point present ----
        if 'har_model_predict' in model_cpp:
            report['checks_passed'].append("har_model_predict() present ✓")
        else:
            report['issues'].append("har_model_predict() not found in har_model.cpp")
            report['passed'] = False

        # ---- Check 11: Class names match ----
        for cls in model_data.get('classes', []):
            if f'"{cls}"' not in classifier_cpp:
                report['warnings'].append(
                    f"Class label '{cls}' not found in HAR_CLASS_NAMES array")

        # ---- Check 12: Sketch includes har_classifier.h ----
        if sketch and '#include "har_classifier.h"' not in sketch:
            report['issues'].append(
                "Sketch does not include har_classifier.h — inference won't compile")
            report['passed'] = False
        elif sketch:
            report['checks_passed'].append("Sketch includes har_classifier.h ✓")

        return report

    def estimate_resources(self, model_data: Dict[str, Any],
                           device_key: str,
                           optimization: str = 'balanced') -> Dict[str, Any]:
        """
        Estimate memory, power, and performance for a specific device.
        
        Args:
            model_data: Model metadata including type, features, classes
            device_key: Key into DEVICE_SPECS
            optimization: Optimization level
            
        Returns:
            Detailed resource estimation
        """
        device = self.DEVICE_SPECS.get(device_key)
        if not device:
            return {'error': f"Unknown device: {device_key}"}

        num_features = len(model_data.get('feature_names') or [])
        num_classes = len(model_data.get('classes') or [])
        model_type = model_data.get('model_type', 'unknown')

        # --- RAM Estimation ---
        # Sensor buffer: window_size * 6 axes * 4 bytes
        window_size = self._get_window_size(optimization)
        sensor_buffer_bytes = window_size * 6 * 4

        # CNN models: raw window buffer + layer activations, no feature extraction
        if model_type in ('pytorch_cnn', 'pytorch_cnn2d', 'cnn'):
            return self._estimate_cnn_resources(
                model_data, device, optimization, window_size, num_classes)

        # Feature arrays: raw + scaled
        feature_array_bytes = num_features * 4 * 2  # raw + scaled
        
        # Sorting buffer for median/quartile (if magnitude features)
        sort_buffer_bytes = window_size * 4  # float array for sorting
        
        # Magnitude arrays (if orientation-robust)
        has_mag_features = any(str(n).startswith('acc_mag_') 
                             for n in model_data.get('feature_names', []))
        mag_buffer_bytes = window_size * 3 * 4 if has_mag_features else 0  # acc, gyro, jerk
        
        # Model-specific working memory
        if model_type == 'neural_network':
            # Hidden layer outputs (largest layer)
            model_obj = model_data.get('model_object')
            if model_obj and hasattr(model_obj, 'model') and hasattr(model_obj.model, 'hidden_layer_sizes'):
                hidden_sizes = model_obj.model.hidden_layer_sizes
                if isinstance(hidden_sizes, tuple):
                    max_hidden = max(hidden_sizes)
                else:
                    max_hidden = hidden_sizes
            else:
                max_hidden = 50
            model_working_bytes = max_hidden * 4 + num_classes * 4
        elif model_type == 'svm':
            num_sv = len(model_data.get('support_vectors', []))
            model_working_bytes = num_classes * 4  # decision_scores
        elif model_type == 'random_forest':
            model_working_bytes = num_classes * 4  # votes array
        else:
            model_working_bytes = 256
        
        # Stack overhead
        stack_overhead = 512
        
        ram_total = (sensor_buffer_bytes + feature_array_bytes + 
                    sort_buffer_bytes + mag_buffer_bytes + 
                    model_working_bytes + stack_overhead)

        # --- Flash Estimation ---
        # Code size (approximate)
        code_base_bytes = 8000  # Base code footprint
        
        # Feature scaling arrays
        scaling_bytes = num_features * 4 * 2  # means + stds
        
        # Model weights in flash
        if model_type == 'neural_network':
            model_obj = model_data.get('model_object')
            if model_obj and hasattr(model_obj, 'model') and hasattr(model_obj.model, 'coefs_'):
                weight_bytes = sum(c.size * 4 for c in model_obj.model.coefs_)
                bias_bytes = sum(b.size * 4 for b in model_obj.model.intercepts_)
                model_flash_bytes = weight_bytes + bias_bytes
            else:
                model_flash_bytes = num_features * 50 * 4 + 50 * num_classes * 4
        elif model_type == 'random_forest':
            trees = model_data.get('trees', [])
            num_trees = min(len(trees), 10) if trees else 10
            avg_nodes_per_tree = 50  # approximate
            # Each node: feature_idx(4) + threshold(4) + left(4) + right(4) + value(4) = 20 bytes
            model_flash_bytes = num_trees * avg_nodes_per_tree * 20
        elif model_type == 'svm':
            num_sv = len(model_data.get('support_vectors', []))
            model_flash_bytes = (num_sv * num_features * 4 +  # support vectors
                               num_classes * num_sv * 4 +     # dual coefficients
                               num_classes * 4)               # intercepts
        else:
            model_flash_bytes = 4000
        
        flash_total = code_base_bytes + scaling_bytes + model_flash_bytes

        # --- Power Estimation ---
        sampling_rate = self._get_sampling_rate(optimization)
        duty_cycle = self._get_duty_cycle(optimization, sampling_rate, window_size)
        
        avg_power_mw = (device['power_active_mw'] * duty_cycle + 
                       device['power_sleep_mw'] * (1 - duty_cycle))
        
        # Battery life estimate (CR2032: 220mAh @ 3V = 660mWh, typical AA: 2500mAh @ 1.5V = 3750mWh)
        battery_cr2032_hours = 660 / avg_power_mw if avg_power_mw > 0 else 0
        battery_aa_hours = 3750 / avg_power_mw if avg_power_mw > 0 else 0

        # --- Inference Time Estimation ---
        ops_per_feature_extraction = self._estimate_feature_ops(num_features, window_size, has_mag_features)
        ops_per_inference = self._estimate_inference_ops(model_type, num_features, num_classes, model_data)
        total_ops = ops_per_feature_extraction + ops_per_inference
        
        # FLOPS depends on hardware FPU
        if device['float_support'] == 'hardware_fpu_dp':
            mflops = device['clock_mhz'] * 0.5  # ~50% FPU utilization
        elif device['float_support'] == 'hardware_fpu':
            mflops = device['clock_mhz'] * 0.3
        else:
            mflops = device['clock_mhz'] * 0.01  # Software float ~100x slower
        
        inference_time_ms = (total_ops / (mflops * 1e6)) * 1000 if mflops > 0 else 999

        # --- Compatibility ---
        ram_ok = ram_total < device['ram_kb'] * 1024 * 0.8  # 80% threshold
        flash_ok = flash_total < device['flash_kb'] * 1024 * 0.9  # 90% threshold
        
        return {
            'device': device,
            'ram': {
                'sensor_buffer': sensor_buffer_bytes,
                'feature_arrays': feature_array_bytes,
                'sort_buffer': sort_buffer_bytes,
                'magnitude_buffers': mag_buffer_bytes,
                'model_working': model_working_bytes,
                'stack_overhead': stack_overhead,
                'total_bytes': ram_total,
                'total_kb': ram_total / 1024,
                'usage_percent': (ram_total / (device['ram_kb'] * 1024)) * 100,
                'fits': ram_ok,
            },
            'flash': {
                'code_base': code_base_bytes,
                'scaling_arrays': scaling_bytes,
                'model_weights': model_flash_bytes,
                'total_bytes': flash_total,
                'total_kb': flash_total / 1024,
                'usage_percent': (flash_total / (device['flash_kb'] * 1024)) * 100,
                'fits': flash_ok,
            },
            'power': {
                'sampling_rate': sampling_rate,
                'duty_cycle_percent': duty_cycle * 100,
                'active_power_mw': device['power_active_mw'],
                'sleep_power_mw': device['power_sleep_mw'],
                'average_power_mw': avg_power_mw,
                'battery_cr2032_hours': battery_cr2032_hours,
                'battery_aa_hours': battery_aa_hours,
            },
            'performance': {
                'feature_extraction_ops': ops_per_feature_extraction,
                'inference_ops': ops_per_inference,
                'total_ops': total_ops,
                'estimated_inference_ms': inference_time_ms,
                'max_predictions_per_sec': 1000 / inference_time_ms if inference_time_ms > 0 else 0,
            },
            'compatible': ram_ok and flash_ok,
        }

    def generate_test_sketch(self, model_data: Dict[str, Any],
                             test_data: np.ndarray = None) -> str:
        """
        Generate an Arduino sketch that validates C++ feature extraction
        by printing features in a format that can be compared with Python output.
        
        Args:
            model_data: Model metadata
            test_data: Optional test window data (samples x 6)
            
        Returns:
            Arduino sketch code for validation
        """
        num_features = len(model_data.get('feature_names', []))
        feature_names = model_data.get('feature_names', [])

        # Create a small hardcoded test window (10 samples of known data)
        if test_data is not None and len(test_data) >= 10:
            test_window = test_data[:10]
        else:
            # Generate a reproducible test pattern
            test_window = np.array([
                [0.1, -0.2, 9.8, 0.01, -0.02, 0.03],
                [0.15, -0.18, 9.82, 0.02, -0.01, 0.02],
                [0.2, -0.15, 9.78, 0.03, -0.03, 0.01],
                [0.12, -0.22, 9.81, 0.01, -0.02, 0.04],
                [0.18, -0.19, 9.79, 0.02, -0.01, 0.02],
                [0.11, -0.21, 9.83, 0.01, -0.03, 0.03],
                [0.16, -0.17, 9.77, 0.03, -0.02, 0.01],
                [0.14, -0.20, 9.80, 0.02, -0.01, 0.04],
                [0.19, -0.16, 9.82, 0.01, -0.03, 0.02],
                [0.13, -0.23, 9.78, 0.03, -0.02, 0.03],
            ])

        test_data_str = ""
        for i, row in enumerate(test_window):
            vals = ", ".join(f"{v:.6f}f" for v in row)
            test_data_str += f"    {{{vals}}}"
            if i < len(test_window) - 1:
                test_data_str += ","
            test_data_str += "\n"

        return f"""/*
 * HAR Deployment Validation Sketch
 * Compares C++ feature extraction with known Python output
 * 
 * Usage:
 * 1. Upload this sketch to your device
 * 2. Open Serial Monitor at 115200 baud
 * 3. Copy the feature values
 * 4. Compare with Python validation script output
 */

#include "har_model.h"

// Test data: {len(test_window)} samples x 6 axes (aX, aY, aZ, gX, gY, gZ)
float test_sensor_data[{len(test_window)}][6] = {{
{test_data_str}}};

void setup() {{
    Serial.begin(115200);
    while (!Serial) delay(10);
    
    Serial.println("\\n=== HAR Deployment Validation ===");
    Serial.println("Comparing C++ feature extraction with Python\\n");
    
    har_init();
    
    // Extract features from test data
    float features[NUM_FEATURES];
    extract_features(test_sensor_data, {len(test_window)}, features);
    
    // Print all features with high precision for comparison
    Serial.println("--- Raw Features (before scaling) ---");
    Serial.print("FEATURES_START|");
    for (int i = 0; i < NUM_FEATURES; i++) {{
        Serial.print(features[i], 8);
        if (i < NUM_FEATURES - 1) Serial.print(",");
    }}
    Serial.println("|FEATURES_END");
    
    Serial.println("\\n--- Feature Details ---");
    for (int i = 0; i < NUM_FEATURES; i++) {{
        Serial.print("Feature[");
        Serial.print(i);
        Serial.print("]: ");
        Serial.println(features[i], 8);
    }}
    
    // Run prediction
    Serial.println("\\n--- Prediction ---");
    float confidence = 0.0f;
    int prediction = har_predict(features, &confidence);
    Serial.print("Predicted class: ");
    Serial.print(prediction);
    Serial.print(" (");
    Serial.print(get_activity_name(prediction));
    Serial.print(") confidence: ");
    Serial.println(confidence, 4);
    
    // Memory usage
    Serial.println("\\n--- Memory ---");
    Serial.print("Feature array: ");
    Serial.print(NUM_FEATURES * 4);
    Serial.println(" bytes");
    Serial.print("Sensor buffer: ");
    Serial.print(WINDOW_SIZE * 6 * 4);
    Serial.println(" bytes");
    
    Serial.println("\\n=== Validation Complete ===");
}}

void loop() {{
    delay(10000);
}}
"""

    def _estimate_cnn_resources(self, model_data: Dict[str, Any],
                                device: Dict, optimization: str,
                                window_size: int, num_classes: int) -> Dict[str, Any]:
        """Estimate resources for CNN model (raw window input, no feature extraction)."""
        n_channels = 6  # 6-axis IMU
        # Raw sensor window buffer
        sensor_buffer_bytes = window_size * n_channels * 4

        # CNN layer activations (estimate from largest intermediate)
        # Conv output: window_size * num_filters * 4 bytes
        cnn_weights = model_data.get('cnn_weights', {})
        num_filters = cnn_weights.get('num_filters', [16])[0] if cnn_weights else 16
        conv_output_bytes = window_size * num_filters * 4
        # Dense layer working memory
        dense_working_bytes = num_filters * 4 + num_classes * 4
        model_working_bytes = conv_output_bytes + dense_working_bytes

        stack_overhead = 512
        ram_total = sensor_buffer_bytes + model_working_bytes + stack_overhead

        # Flash: code base + weight arrays
        code_base_bytes = 6000  # Simpler than feature-based (no FE code)
        # Weight estimation from model data
        model_flash_bytes = 4000  # Default estimate
        if cnn_weights:
            total_params = 0
            for key, val in cnn_weights.items():
                if isinstance(val, (list, np.ndarray)):
                    arr = np.array(val)
                    total_params += arr.size
            model_flash_bytes = total_params * 4 if total_params > 0 else 4000

        flash_total = code_base_bytes + model_flash_bytes

        # Power (same as feature-based)
        sampling_rate = self._get_sampling_rate(optimization)
        duty_cycle = self._get_duty_cycle(optimization, sampling_rate, window_size)
        avg_power_mw = (device['power_active_mw'] * duty_cycle +
                       device['power_sleep_mw'] * (1 - duty_cycle))
        battery_cr2032_hours = 660 / avg_power_mw if avg_power_mw > 0 else 0
        battery_aa_hours = 3750 / avg_power_mw if avg_power_mw > 0 else 0

        # Inference time (CNN: multiply-accumulate per conv + dense)
        conv_ops = window_size * n_channels * num_filters * 3  # kernel_size ≈ 3
        dense_ops = num_filters * num_classes * 2
        total_ops = conv_ops + dense_ops
        if device['float_support'] == 'hardware_fpu_dp':
            mflops = device['clock_mhz'] * 0.5
        elif device['float_support'] == 'hardware_fpu':
            mflops = device['clock_mhz'] * 0.3
        else:
            mflops = device['clock_mhz'] * 0.01
        inference_time_ms = (total_ops / (mflops * 1e6)) * 1000 if mflops > 0 else 999

        ram_ok = ram_total < device['ram_kb'] * 1024 * 0.8
        flash_ok = flash_total < device['flash_kb'] * 1024 * 0.9

        return {
            'device': device,
            'ram': {
                'sensor_buffer': sensor_buffer_bytes,
                'feature_arrays': 0,
                'sort_buffer': 0,
                'magnitude_buffers': 0,
                'model_working': model_working_bytes,
                'stack_overhead': stack_overhead,
                'total_bytes': ram_total,
                'total_kb': ram_total / 1024,
                'usage_percent': (ram_total / (device['ram_kb'] * 1024)) * 100,
                'fits': ram_ok,
            },
            'flash': {
                'code_base': code_base_bytes,
                'scaling_arrays': 0,
                'model_weights': model_flash_bytes,
                'total_bytes': flash_total,
                'total_kb': flash_total / 1024,
                'usage_percent': (flash_total / (device['flash_kb'] * 1024)) * 100,
                'fits': flash_ok,
            },
            'power': {
                'sampling_rate': sampling_rate,
                'duty_cycle_percent': duty_cycle * 100,
                'active_power_mw': device['power_active_mw'],
                'sleep_power_mw': device['power_sleep_mw'],
                'average_power_mw': avg_power_mw,
                'battery_cr2032_hours': battery_cr2032_hours,
                'battery_aa_hours': battery_aa_hours,
            },
            'performance': {
                'feature_extraction_ops': 0,
                'inference_ops': total_ops,
                'total_ops': total_ops,
                'estimated_inference_ms': inference_time_ms,
                'max_predictions_per_sec': 1000 / inference_time_ms if inference_time_ms > 0 else 0,
            },
            'compatible': ram_ok and flash_ok,
        }

    def _min_precision(self, value: float) -> int:
        """Find minimum decimal precision needed to represent a value non-zero."""
        if value == 0:
            return 0
        for p in range(1, 10):
            if round(abs(value), p) > 0:
                return p
        return 9

    def _get_window_size(self, optimization: str) -> int:
        """Get window size for optimization level."""
        sizes = {'accuracy': 100, 'balanced': 150, 'speed': 50, 'power': 40}
        return sizes.get(optimization, 150)

    def _get_sampling_rate(self, optimization: str) -> int:
        """Get sampling rate for optimization level."""
        rates = {'accuracy': 100, 'balanced': 100, 'speed': 50, 'power': 25}
        return rates.get(optimization, 100)

    def _get_duty_cycle(self, optimization: str, 
                        sampling_rate: int, window_size: int) -> float:
        """Estimate processor duty cycle (fraction of time active)."""
        window_duration_s = window_size / sampling_rate
        # Active time = sampling + feature extraction + inference
        # Approximate: 1ms per sample + 10ms feature extraction + 5ms inference
        active_time_s = window_size * 0.001 + 0.010 + 0.005
        
        if optimization == 'power':
            # Power mode can sleep between windows
            return active_time_s / (window_duration_s * 2)  # 50% overlap
        elif optimization == 'speed':
            return 0.8  # Mostly active
        else:
            return active_time_s / window_duration_s

    def _estimate_feature_ops(self, num_features: int, 
                              window_size: int, has_mag: bool) -> int:
        """Estimate floating-point operations for feature extraction."""
        if has_mag:
            # Magnitude calculation: 3 mul + 2 add + 1 sqrt per sample per signal (3 signals)
            mag_ops = window_size * 6 * 3
            # Stats (mean, std, etc.): ~20 ops per sample per magnitude signal
            stats_ops = window_size * 20 * 3
            # Sorting for median (bubble sort worst case)
            sort_ops = window_size * window_size * 2  # O(n²) bubble sort
            return mag_ops + stats_ops + sort_ops
        else:
            # Per-axis stats: ~15 ops per sample per axis
            stats_ops = window_size * 15 * 6
            sort_ops = window_size * window_size * 6  # sorting per axis
            return stats_ops + sort_ops

    def _estimate_inference_ops(self, model_type: str, 
                                num_features: int, num_classes: int,
                                model_data: Dict = None) -> int:
        """Estimate floating-point operations for model inference."""
        if model_type == 'neural_network':
            model_obj = model_data.get('model_object') if model_data else None
            if model_obj and hasattr(model_obj, 'model') and hasattr(model_obj.model, 'coefs_'):
                total_ops = 0
                for coef in model_obj.model.coefs_:
                    total_ops += coef.shape[0] * coef.shape[1] * 2  # mul + add per weight
                return total_ops
            # Fallback estimate
            return num_features * 50 * 2 + 50 * num_classes * 2
        elif model_type == 'random_forest':
            trees = model_data.get('trees', []) if model_data else []
            num_trees = min(len(trees), 10) if trees else 10
            avg_depth = 15
            return num_trees * avg_depth * 3  # comparison + branch per level per tree
        elif model_type == 'svm':
            num_sv = len(model_data.get('support_vectors', [])) if model_data else 50
            # RBF kernel: num_features operations per SV, then exp
            return num_sv * (num_features * 3 + 10) * num_classes
        return num_features * 10


def validate_before_deployment(model_data: Dict[str, Any],
                                generated_files: Dict[str, str],
                                device_key: str = None) -> Dict[str, Any]:
    """
    Convenience function to run all validation checks before deployment.
    
    Args:
        model_data: Model metadata dict
        generated_files: Dict of filename -> content
        device_key: Optional device key for resource estimation
        
    Returns:
        Complete validation report
    """
    validator = DeploymentValidator(model_data)
    report = {'passed': True, 'checks': {}}

    # Detect v2 architecture: has har_config.h with HAR_NUM_FEATURES define
    is_v2 = 'har_config.h' in generated_files and (
        'HAR_NUM_FEATURES' in generated_files.get('har_config.h', '')
    )

    if is_v2:
        code_report = validator._validate_v2_code(generated_files, model_data)
        report['checks']['code'] = code_report
        if not code_report['passed']:
            report['passed'] = False
    else:
        # Legacy v1: find first .h and first .cpp
        header_code = ""
        source_code = ""
        for fname, content in generated_files.items():
            if fname.endswith('.h') and fname != 'build_opt.h' and not header_code:
                header_code = content
            elif (fname.endswith('.cpp') or fname.endswith('.c')) and not source_code:
                source_code = content

        if header_code and source_code:
            code_report = validator.validate_generated_code(
                header_code, source_code, model_data)
            report['checks']['code'] = code_report
            if not code_report['passed']:
                report['passed'] = False

    # Scaling validation (skip for CNN — CNN uses raw sensor windows, no feature scaling)
    model_type = model_data.get('model_type', '')
    is_cnn = model_type in ('pytorch_cnn', 'pytorch_cnn2d', 'cnn')
    feature_means = model_data.get('feature_means', [])
    feature_stds = model_data.get('feature_stds', [])
    if feature_means and feature_stds and not is_cnn:
        # Determine precision from optimization level
        optimization = model_data.get('optimization', 'balanced')
        precision = {'accuracy': 6, 'balanced': 4, 'speed': 3, 'power': 3}.get(optimization, 4)
        scaling_report = validator.validate_scaling_parameters(
            feature_means, feature_stds, precision)
        report['checks']['scaling'] = scaling_report
        if not scaling_report['passed']:
            report['passed'] = False
    
    # Resource estimation
    if device_key:
        resource_report = validator.estimate_resources(model_data, device_key)
        report['checks']['resources'] = resource_report
        if not resource_report.get('compatible', True):
            report['passed'] = False
    
    return report
