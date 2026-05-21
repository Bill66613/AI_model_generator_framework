"""
Code Generator Factory
Factory pattern implementation for creating appropriate code generators
"""

import os
import numpy as np
from datetime import datetime
from typing import Dict, Any, Type, List, Optional
from .v2 import HARCodeGenerator as HARCodeGeneratorV2


# ---------------------------------------------------------------------------
# Exceptions (previously in base_generator.py)
# ---------------------------------------------------------------------------

class ValidationError(ValueError):
    """Raised when input parameters are invalid."""


class ModelDataError(ValidationError):
    """Raised when model data is missing or malformed."""


class OptimizationError(ValidationError):
    """Raised when optimization settings are invalid."""


# Backward-compat stub so old code that catches BaseCodeGenerator still works
class BaseCodeGenerator:
    """Stub kept for backward compatibility. All generation is now v2."""


# Model types that the v2 clean architecture supports
_V2_SUPPORTED_MODELS = frozenset(
    ['random_forest', 'neural_network', 'pytorch_mlp', 'svm',
     'pytorch_cnn', 'pytorch_cnn2d'])

# Valid deployment approaches
DEPLOYMENT_APPROACHES = ('direct', 'tflite_micro', 'onnx_runtime')


def get_cpp_feature_order(feature_names: List[str]) -> List[str]:
    """Get the canonical C++ feature extraction order.

    The C++ extract_features() function outputs features in a fixed order:
      - extract_magnitude_stats(acc_mag): 15 features
      - extract_magnitude_stats(gyro_mag): 15 features  
      - jerk features: 3 features (mean, std, max)

    This order may differ from the model's training order (which depends on
    DataFrame column ordering, typically alphabetical). This function returns
    the C++ extraction order so we can reorder model parameters to match.
    """
    # The 15 stats per magnitude, in the order extract_magnitude_stats() outputs them
    magnitude_stats = [
        'mean', 'std', 'min', 'max', 'range',
        'median', 'q25', 'q75', 'iqr',
        'skewness', 'kurtosis', 'rms', 'energy',
        'zero_crossings', 'mean_crossing_rate'
    ]

    # Detect which magnitude groups are present from feature names
    feature_set = set(str(f) for f in feature_names)

    has_acc_mag = any(f.startswith('acc_mag_') for f in feature_set)
    has_gyro_mag = any(f.startswith('gyro_mag_') for f in feature_set)
    has_jerk = any(f.startswith('acc_jerk_mag_') for f in feature_set)

    # Check for per-axis features (non-orientation-robust)
    has_per_axis = any(f.startswith(
        ('aX_', 'aY_', 'aZ_', 'gX_', 'gY_', 'gZ_')) for f in feature_set)

    if has_per_axis:
        # Per-axis mode: features extracted axis by axis
        # Cannot reliably determine C++ order for per-axis mode,
        # so return the original feature names (no reordering)
        return list(feature_names)

    cpp_order = []

    if has_acc_mag:
        for stat in magnitude_stats:
            name = f'acc_mag_{stat}'
            if name in feature_set:
                cpp_order.append(name)

    if has_gyro_mag:
        for stat in magnitude_stats:
            name = f'gyro_mag_{stat}'
            if name in feature_set:
                cpp_order.append(name)

    if has_jerk:
        for suffix in ['mean', 'std', 'max']:
            name = f'acc_jerk_mag_{suffix}'
            if name in feature_set:
                cpp_order.append(name)

    # Gyro jerk magnitude features (after acc_jerk_mag)
    has_gyro_jerk = any(f.startswith('gyro_jerk_mag_') for f in feature_set)
    if has_gyro_jerk:
        for suffix in ['mean', 'std', 'max']:
            name = f'gyro_jerk_mag_{suffix}'
            if name in feature_set:
                cpp_order.append(name)

    # Scalar features extracted after jerk blocks
    for scalar_feat in ['acc_sma', 'tilt_pitch', 'tilt_roll',
                        'acc_mag_autocorr_lag1', 'acc_jerk_mag_peak_count']:
        if scalar_feat in feature_set:
            cpp_order.append(scalar_feat)

    # Frequency-domain features (DFT on magnitudes) - 11 features per signal
    # Order matches C extract_frequency_features(): dominant_frequency,
    # dominant_frequency_magnitude, spectral_centroid, energy_low_freq,
    # energy_mid_freq, energy_high_freq, spectral_rolloff,
    # spectral_rms, spectral_skewness, spectral_kurtosis, spectral_entropy
    freq_stats = [
        'dominant_frequency', 'dominant_frequency_magnitude',
        'spectral_centroid', 'energy_low_freq', 'energy_mid_freq',
        'energy_high_freq', 'spectral_rolloff',
        'spectral_rms', 'spectral_skewness', 'spectral_kurtosis',
        'spectral_entropy'
    ]
    has_acc_freq = any(f.startswith('acc_mag_dominant_') or f.startswith('acc_mag_spectral_')
                       or f.startswith('acc_mag_energy_low') for f in feature_set)
    has_gyro_freq = any(f.startswith('gyro_mag_dominant_') or f.startswith('gyro_mag_spectral_')
                        or f.startswith('gyro_mag_energy_low') for f in feature_set)

    if has_acc_freq:
        for stat in freq_stats:
            name = f'acc_mag_{stat}'
            if name in feature_set:
                cpp_order.append(name)

    if has_gyro_freq:
        for stat in freq_stats:
            name = f'gyro_mag_{stat}'
            if name in feature_set:
                cpp_order.append(name)

    # Verify we captured all features
    if set(cpp_order) != feature_set:
        missing = feature_set - set(cpp_order)
        extra = set(cpp_order) - feature_set
        print(
            f"Warning: Feature order mismatch. Missing from C++ order: {missing}, Extra: {extra}")
        # Fall back to original order if we can't determine C++ order
        return list(feature_names)

    return cpp_order


def compute_feature_reorder_indices(model_feature_names: List[str],
                                    cpp_feature_order: List[str]) -> Optional[List[int]]:
    """Compute reordering indices to map from model order to C++ order.

    Returns a list of indices such that:
        cpp_order_value[i] = model_order_value[indices[i]]

    i.e., for each position in the C++ output, which model-order index to pull from.
    Returns None if orders already match (no reordering needed).
    """
    model_names = [str(f) for f in model_feature_names]

    if model_names == cpp_feature_order:
        return None  # Already in correct order

    # Build lookup: feature_name -> model index
    name_to_model_idx = {name: i for i, name in enumerate(model_names)}

    reorder = []
    for cpp_name in cpp_feature_order:
        if cpp_name not in name_to_model_idx:
            print(
                f"Warning: C++ feature '{cpp_name}' not found in model features")
            return None  # Can't reorder safely
        reorder.append(name_to_model_idx[cpp_name])

    return reorder


def reorder_model_parameters(enhanced_data: Dict[str, Any],
                             reorder_indices: List[int],
                             cpp_feature_order: List[str]) -> Dict[str, Any]:
    """Reorder all model parameters from model (training) order to C++ extraction order.

    This ensures that feature_means[i], feature_stds[i], and weight matrix row [i]
    all correspond to the feature that C++ extract_features() places at position [i].
    """
    print(
        f"Reordering {len(reorder_indices)} features from model order to C++ extraction order")

    # Reorder scaler parameters
    if 'feature_means' in enhanced_data:
        old_means = enhanced_data['feature_means']
        enhanced_data['feature_means'] = [old_means[i]
                                          for i in reorder_indices]

    if 'feature_stds' in enhanced_data:
        old_stds = enhanced_data['feature_stds']
        enhanced_data['feature_stds'] = [old_stds[i] for i in reorder_indices]

    # Reorder neural network input weights (rows correspond to features)
    if 'weights' in enhanced_data:
        weights = enhanced_data['weights']
        if 'input_weights' in weights:
            # shape: [n_features, hidden_size]
            old_w = np.array(weights['input_weights'])
            new_w = old_w[reorder_indices, :]  # Reorder rows
            weights['input_weights'] = new_w.tolist()
        # Also reorder the full multi-layer storage if present
        if 'all_coefs' in weights and len(weights['all_coefs']) > 0:
            old_w = np.array(weights['all_coefs'][0])
            new_w = old_w[reorder_indices, :]  # Reorder first layer rows
            weights['all_coefs'][0] = new_w.tolist()
        enhanced_data['weights'] = weights

    # Reorder pytorch MLP input weights (first layer rows correspond to features)
    if 'pytorch_coefs' in enhanced_data:
        pytorch_coefs = enhanced_data['pytorch_coefs']
        if len(pytorch_coefs) > 0:
            old_w = np.array(pytorch_coefs[0])
            new_w = old_w[reorder_indices, :]  # Reorder input rows
            pytorch_coefs[0] = new_w.tolist() if hasattr(new_w, 'tolist') else new_w
            enhanced_data['pytorch_coefs'] = pytorch_coefs

    # Reorder Random Forest feature indices in tree splits
    if 'trees' in enhanced_data:
        # Build reverse mapping: model_idx -> cpp_idx
        model_to_cpp = [0] * len(reorder_indices)
        for cpp_idx, model_idx in enumerate(reorder_indices):
            model_to_cpp[model_idx] = cpp_idx

        for tree in enhanced_data['trees']:
            if 'feature_indices' in tree:
                old_indices = tree['feature_indices']
                tree['feature_indices'] = [
                    model_to_cpp[fi] if fi >= 0 else fi  # -2 means leaf node
                    for fi in old_indices
                ]

    # Reorder SVM support vectors (columns correspond to features)
    if 'support_vectors' in enhanced_data:
        # shape: [n_sv, n_features]
        old_sv = np.array(enhanced_data['support_vectors'])
        new_sv = old_sv[:, reorder_indices]  # Reorder columns
        enhanced_data['support_vectors'] = new_sv.tolist()

    # Update feature names to C++ order
    enhanced_data['feature_names'] = cpp_feature_order

    return enhanced_data


def extract_real_model_parameters(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract actual parameters from trained model object.

    Also reorders all parameters to match the C++ feature extraction order,
    since the model may have been trained with features in a different order
    (e.g., alphabetical from DataFrame columns) than the C++ code extracts them.
    """
    model_obj = model_data.get('model_object')
    enhanced_data = model_data.copy()

    if model_obj and hasattr(model_obj, 'model'):
        try:
            # Extract feature scaling parameters
            if hasattr(model_obj, 'scaler') and model_obj.scaler:
                enhanced_data['feature_means'] = model_obj.scaler.mean_.tolist()
                enhanced_data['feature_stds'] = model_obj.scaler.scale_.tolist()

            # Extract model-specific parameters
            if model_obj.model_type == 'random_forest':
                enhanced_data['trees'] = extract_random_forest_trees(
                    model_obj.model)
            elif model_obj.model_type == 'neural_network':
                enhanced_data['weights'] = extract_neural_network_weights(
                    model_obj.model)
            elif model_obj.model_type == 'pytorch_mlp':
                # PyTorch MLP — export weights in sklearn-compatible format
                if hasattr(model_obj, '_pytorch_trainer'):
                    mlp_export = model_obj._pytorch_trainer.export_mlp_weights()
                    enhanced_data['weights'] = {
                        'input_weights': mlp_export['coefs_'][0].tolist() if hasattr(mlp_export['coefs_'][0], 'tolist') else mlp_export['coefs_'][0],
                        'hidden_biases': mlp_export['intercepts_'][0].tolist() if hasattr(mlp_export['intercepts_'][0], 'tolist') else mlp_export['intercepts_'][0],
                        'hidden_size': mlp_export['hidden_layer_sizes'][0] if mlp_export['hidden_layer_sizes'] else 64,
                    }
                    # Store full coefs/intercepts for multi-layer extraction
                    enhanced_data['pytorch_coefs'] = mlp_export['coefs_']
                    enhanced_data['pytorch_intercepts'] = mlp_export['intercepts_']
                    enhanced_data['pytorch_hidden_layer_sizes'] = mlp_export['hidden_layer_sizes']
            elif model_obj.model_type in ('pytorch_cnn', 'pytorch_cnn2d'):
                # PyTorch CNN — export layer descriptions for CNNCodeGenerator
                if hasattr(model_obj, '_pytorch_trainer'):
                    cnn_export = model_obj._pytorch_trainer.export_cnn_weights()
                    enhanced_data['cnn_weights'] = cnn_export
            elif model_obj.model_type == 'svm':
                # Extract and merge SVM parameters
                svm_params = extract_svm_parameters(model_obj.model)
                enhanced_data.update(svm_params)

            # Extract label encoding
            if hasattr(model_obj, 'label_encoder') and model_obj.label_encoder:
                enhanced_data['label_mapping'] = {
                    i: label for i, label in enumerate(model_obj.label_encoder.classes_)
                }

            # Reorder parameters to match C++ feature extraction order
            # (skip for CNN which operates on raw sensor windows)
            feature_names = enhanced_data.get('feature_names', [])
            if feature_names and model_obj.model_type not in ('pytorch_cnn', 'pytorch_cnn2d'):
                cpp_order = get_cpp_feature_order(feature_names)
                reorder_indices = compute_feature_reorder_indices(
                    feature_names, cpp_order)
                if reorder_indices is not None:
                    enhanced_data = reorder_model_parameters(
                        enhanced_data, reorder_indices, cpp_order)
                    # Store reorder indices so model-specific generators can also reorder
                    # their own directly-extracted weights (e.g., NeuralNetworkCodeGenerator
                    # reads from model_obj.model.coefs_ directly)
                    enhanced_data['_feature_reorder_indices'] = reorder_indices
                    print(f"Feature order remapped: model order → C++ extraction order")
                else:
                    print(f"Feature order already matches C++ extraction order")

        except Exception as e:
            print(f"Warning: Could not extract real model parameters: {e}")
            import traceback
            traceback.print_exc()

    return enhanced_data


def extract_random_forest_trees(rf_model) -> List[Dict]:
    """Extract decision tree structure from RandomForest model."""
    trees = []
    try:
        for i, tree in enumerate(rf_model.estimators_):
            tree_data = {
                'tree_id': i,
                'feature_indices': tree.tree_.feature.tolist(),
                'thresholds': tree.tree_.threshold.tolist(),
                'left_children': tree.tree_.children_left.tolist(),
                'right_children': tree.tree_.children_right.tolist(),
                'values': tree.tree_.value.tolist()
            }
            trees.append(tree_data)
    except Exception as e:
        print(f"Could not extract trees: {e}")
    return trees


def extract_neural_network_weights(nn_model) -> Dict[str, Any]:
    """Extract weights and biases from neural network model.

    For multi-layer MLPs (hidden_layer_sizes with 2+ layers), all coefs_
    and intercepts_ are stored in 'all_coefs' / 'all_intercepts' lists.
    The legacy 'input_weights'/'output_weights' keys still store the first
    and last layer for backward compatibility.
    """
    weights = {}
    try:
        if hasattr(nn_model, 'coefs_') and hasattr(nn_model, 'intercepts_'):
            n_layers = len(nn_model.coefs_)
            # Legacy keys for 1-hidden-layer compat
            weights['input_weights'] = nn_model.coefs_[0].tolist()
            weights['hidden_biases'] = nn_model.intercepts_[0].tolist()
            if n_layers > 1:
                weights['output_weights'] = nn_model.coefs_[-1].tolist()
                weights['output_biases'] = nn_model.intercepts_[-1].tolist()
            weights['hidden_size'] = len(nn_model.intercepts_[0])

            # Full multi-layer storage
            weights['all_coefs'] = [c.tolist() for c in nn_model.coefs_]
            weights['all_intercepts'] = [b.tolist() for b in nn_model.intercepts_]
    except Exception as e:
        print(f"Could not extract neural network weights: {e}")
    return weights


def extract_svm_parameters(svm_model) -> Dict[str, Any]:
    """Extract support vectors and parameters from SVM model.

    IMPORTANT: scikit-learn's SVC uses One-vs-One (OvO) internally, where
    ``dual_coef_`` has shape ``[n_classes-1, n_SV]`` — a packed format that
    does NOT correspond to a simple per-class coefficient matrix.

    The generated C / MicroPython code uses a One-vs-Rest (OvR) decision
    scheme: for each class *k* the score is a linear combination of kernel
    evaluations plus an intercept, and the predicted class is ``argmax``.

    To bridge the gap, this function converts the OvO representation into
    "effective OvR" coefficients by summing the signed per-pair decision
    functions for each class.  The result is:

        dual_coefficients  – shape [n_classes, n_SV]
        intercept          – shape [n_classes]

    which the C code can use directly with ``argmax(scores)``.
    """
    params = {}
    try:
        import numpy as np

        if not hasattr(svm_model, 'support_vectors_'):
            return params

        params['support_vectors'] = svm_model.support_vectors_.tolist()

        # ----------------------------------------------------------------
        # Convert OvO dual_coef_ → OvR dual coefficients
        # ----------------------------------------------------------------
        n_classes = len(svm_model.classes_)
        n_sv = len(svm_model.support_vectors_)
        n_support = list(int(x) for x in svm_model.n_support_)

        # Build cumulative start index per class in the SV array
        class_start = [0]
        for ns in n_support:
            class_start.append(class_start[-1] + ns)

        ovr_coef = np.zeros((n_classes, n_sv))
        ovr_intercept = np.zeros(n_classes)

        # For each OvO pair (ci, cj) with ci < cj, the decision function is:
        #   f(x) = sum_s pair_coef[s] * K(x, sv_s) + intercept_[pair_idx]
        # If f(x) > 0, class ci wins; otherwise class cj wins.
        #
        # Per-pair dual coefficients are extracted from libsvm's packed
        # dual_coef_ matrix:
        #   - SVs of class ci: row = cj - 1
        #   - SVs of class cj: row = ci
        pair_idx = 0
        for ci in range(n_classes):
            for cj in range(ci + 1, n_classes):
                pair_coef = np.zeros(n_sv)

                # Coefficients for class ci's support vectors
                for s in range(class_start[ci], class_start[ci + 1]):
                    pair_coef[s] = svm_model.dual_coef_[cj - 1, s]

                # Coefficients for class cj's support vectors
                for s in range(class_start[cj], class_start[cj + 1]):
                    pair_coef[s] = svm_model.dual_coef_[ci, s]

                # Accumulate: class ci gets +decision, class cj gets -decision
                ovr_coef[ci] += pair_coef
                ovr_intercept[ci] += svm_model.intercept_[pair_idx]
                ovr_coef[cj] -= pair_coef
                ovr_intercept[cj] -= svm_model.intercept_[pair_idx]

                pair_idx += 1

        params['dual_coefficients'] = ovr_coef.tolist()
        params['intercept'] = ovr_intercept.tolist()

        # ----------------------------------------------------------------
        # Also store native OvO data for OvO-voting prediction scheme
        # ----------------------------------------------------------------
        # sklearn's packed dual_coef_: shape [n_classes-1, n_SV]
        params['ovo_dual_coef'] = svm_model.dual_coef_.tolist()
        # sklearn's per-pair intercepts: shape [n_pairs]
        params['ovo_intercept'] = svm_model.intercept_.tolist()
        # Number of SVs per class
        params['n_support'] = n_support

        n_pairs = n_classes * (n_classes - 1) // 2
        print(f"SVM OvO→OvR conversion: {n_classes} classes, {n_sv} SVs, "
              f"{n_pairs} pairs → OvR coef [{n_classes}×{n_sv}], "
              f"OvO packed [{n_classes - 1}×{n_sv}]")

        # ----------------------------------------------------------------
        # Extract actual gamma value
        # ----------------------------------------------------------------
        gamma_resolved = None

        # Try the private _gamma attribute first (scikit-learn stores the computed value here)
        if hasattr(svm_model, '_gamma'):
            try:
                gamma_resolved = float(svm_model._gamma)
            except (TypeError, ValueError):
                pass

        # If _gamma didn't work, check the gamma parameter
        if gamma_resolved is None and hasattr(svm_model, 'gamma'):
            gamma_val = svm_model.gamma
            if isinstance(gamma_val, (int, float)):
                gamma_resolved = float(gamma_val)
            elif isinstance(gamma_val, str):
                n_features = svm_model.support_vectors_.shape[1]
                if gamma_val == 'scale':
                    sv_var = np.var(svm_model.support_vectors_)
                    gamma_resolved = 1.0 / \
                        (n_features * sv_var) if sv_var > 0 else 1.0 / n_features
                else:  # 'auto'
                    gamma_resolved = 1.0 / n_features

        if gamma_resolved is not None:
            params['gamma'] = gamma_resolved
        else:
            params['gamma'] = 0.1
            print("⚠ Could not extract SVM gamma – using default 0.1")

    except Exception as e:
        print(f"Could not extract SVM parameters: {e}")
        import traceback
        traceback.print_exc()
    return params


def create_organized_filename(model_type: str, platform: str, file_type: str,
                              model_data: Dict[str, Any] = None, optimization: str = 'balanced',
                              quantization: str = 'none') -> str:
    """
    Create organized filename with descriptive naming including optimization level.

    Args:
        model_type: Type of model (random_forest, neural_network, svm)
        platform: Target platform (arduino, arm_cortex_m, etc.)
        file_type: Type of file (header, source, sketch)
        model_data: Optional model data for additional info
        optimization: Optimization level (accuracy, speed, power, balanced)
        quantization: Weight quantization mode ('none', 'int8', 'int16', 'float16')

    Returns:
        Descriptive filename with optimization level
    """
    # Get timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get model-specific info
    num_features = len(model_data.get(
        'feature_names', [])) if model_data else 0
    num_classes = len(model_data.get('classes', [])) if model_data else 0

    # Create descriptive base name with optimization
    base_name = f"har_{model_type}_{platform}"

    if num_features > 0:
        base_name += f"_f{num_features}"
    if num_classes > 0:
        base_name += f"_c{num_classes}"

    # Add optimization level
    base_name += f"_{optimization}"

    # Add quantization if not default
    if quantization and quantization != 'none':
        base_name += f"_{quantization}"

    # Add file extension based on type and platform
    if file_type == 'header':
        return f"{base_name}.h"
    elif file_type == 'source':
        if platform == 'micropython':
            return f"{base_name}.py"
        elif platform in ('generic_c', 'esp_idf', 'zephyr'):
            return f"{base_name}.c"
        return f"{base_name}.cpp"
    elif file_type == 'cortex_source':
        return f"{base_name}.c"
    elif file_type == 'sketch':
        if platform in ('generic_c', 'esp_idf', 'zephyr'):
            return f"{base_name}_example.c"
        elif platform == 'generic_cpp':
            return f"{base_name}_example.cpp"
        elif platform == 'micropython':
            return f"{base_name}_example.py"
        # Arduino .ino file must match folder name - no _example suffix
        return f"{base_name}.ino"
    else:
        return f"{base_name}.{file_type}"


def create_output_folder_structure(base_output_dir: str, model_type: str,
                                   platform: str, model_data: Dict[str, Any] = None,
                                   optimization: str = 'balanced',
                                   deployment_approach: str = 'direct',
                                   quantization: str = 'none') -> str:
    """
    Create organized folder structure for generated code.
    For Arduino-based platforms, creates folder matching the .ino filename.
    Each deployment approach gets its own subfolder to avoid .ino conflicts.

    Args:
        base_output_dir: Base directory for all generated code
        model_type: Type of model
        platform: Target platform
        model_data: Model data containing features and classes info
        optimization: Optimization strategy
        deployment_approach: 'direct', 'tflite_micro', or 'onnx_runtime'

    Returns:
        Full path to the specific model/platform folder
    """
    # Map deployment approach to folder prefix
    approach_prefix = {
        'direct': '',
        'tflite_micro': 'tflite_',
        'onnx_runtime': 'onnx_',
    }.get(deployment_approach, '')

    # For Arduino-based platforms, folder must match .ino filename
    if platform in ['arduino', 'seeed_xiao', 'esp32', 'm5stack', 'teensy']:
        num_features = len(model_data.get(
            'feature_names', [])) if model_data else 0
        num_classes = len(model_data.get('classes', [])) if model_data else 0

        folder_name = f"har_{approach_prefix}{model_type}_{platform}_f{num_features}_c{num_classes}_{optimization}"
        if quantization and quantization != 'none':
            folder_name += f"_{quantization}"
        folder_path = os.path.join(
            base_output_dir, f"{model_type}_models", folder_name)
    elif platform in ('generic_c', 'generic_cpp', 'esp_idf', 'micropython', 'zephyr'):
        folder_path = os.path.join(
            base_output_dir, f"{model_type}_models", platform)
    else:
        folder_path = os.path.join(
            base_output_dir, f"{model_type}_models", platform)

    # Create directories if they don't exist
    os.makedirs(folder_path, exist_ok=True)

    return folder_path


class CodeGeneratorFactory:
    """Factory for creating code generators based on model type and platform.

    NOTE: All generation is now handled by the v2 architecture (deployment/v2/).
    This class is kept for backward-compatibility of callers that use
    create_generator() directly.  Prefer calling generate_deployment_code()
    which routes entirely through v2.
    """

    # Kept for API compatibility — not used for actual generation.
    _generators: Dict[str, Any] = {}

    @classmethod
    def create_generator(cls, model_type: str, model_data: Dict[str, Any],
                         platform: str = 'arduino', optimization: str = 'balanced',
                         overlap: float = 0.5, quantization: str = 'none',
                         deployment_approach: str = 'direct',
                         confidence_threshold: float = 0.6,
                         smoothing_window: int = 1):
        """
        Redirect to v2 HARCodeGenerator.

        Returns an HARCodeGeneratorV2 instance; call .generate_files() on it.
        """
        if model_type not in _V2_SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: '{model_type}'. "
                f"Supported types: {sorted(_V2_SUPPORTED_MODELS)}")

        if 'model_object' in model_data:
            model_data = extract_real_model_parameters(model_data)

        return HARCodeGeneratorV2(
            model_data=model_data,
            platform=platform,
            optimization=optimization,
            overlap=overlap,
            confidence_threshold=confidence_threshold,
            smoothing_window=smoothing_window,
            quantization=quantization,
            deployment_approach=deployment_approach,
        )

    @classmethod
    def get_supported_models(cls) -> list:
        """Get list of supported model types."""
        return sorted(_V2_SUPPORTED_MODELS)

    @classmethod
    def get_supported_platforms(cls) -> list:
        """Get list of supported platforms."""
        return ['arduino', 'arm_cortex_m', 'esp32', 'm5stack', 'teensy', 'seeed_xiao',
                'generic_c', 'generic_cpp', 'esp_idf', 'micropython', 'zephyr']

    @classmethod
    def register_generator(cls, model_type: str, generator_class):
        """Register a custom generator class (stored in _generators for inspection)."""
        cls._generators[model_type] = generator_class


def generate_deployment_code(model_type: str, model_data: Dict[str, Any],
                             platform: str = 'arduino', optimization: str = 'balanced',
                             overlap: float = 0.5, quantization: str = 'none',
                             deployment_approach: str = 'direct',
                             confidence_threshold: float = 0.6,
                             smoothing_window: int = 1) -> Dict[str, str]:
    """
    Convenience function to generate deployment code with organized naming.

    Args:
        model_type: Type of model to generate code for
        model_data: Model parameters and data
        platform: Target platform
        optimization: Optimization strategy ('accuracy', 'speed', 'power', 'balanced')
        overlap: Window overlap percentage (0.0 to 0.99)
        quantization: Weight quantization mode ('none', 'int8', 'int16', 'float16')
        deployment_approach: Deployment approach ('direct', 'tflite_micro', 'onnx_runtime')

    Returns:
        Dictionary with descriptive filename as key and code content as value

    Raises:
        ValidationError: If input parameters are invalid
        ModelDataError: If model_data is invalid
        RuntimeError: If code generation fails
    """
    try:
        # Extract additional model information if available
        if 'model_object' in model_data:
            # If we have the actual model object, extract real parameters
            model_data = extract_real_model_parameters(model_data)

        # ----------------------------------------------------------------
        # v2 clean architecture for supported direct deployments
        # ----------------------------------------------------------------

        # ONNX Runtime requires a full OS (Linux/Windows) and the ONNX Runtime
        # shared library.  It cannot be compiled for bare-metal microcontrollers.
        _MICROCONTROLLER_PLATFORMS = frozenset([
            'arduino', 'nano_33', 'mkr_imu', 'esp32',
            'm5stack', 'm5stick', 'm5stickc', 'seeed_xiao', 'teensy', 'arm_cortex_m',
        ])
        if deployment_approach == 'onnx_runtime' and platform in _MICROCONTROLLER_PLATFORMS:
            raise ValueError(
                f"ONNX Runtime is not supported on microcontroller target '{platform}'. "
                "ONNX Runtime requires a full OS with the runtime library installed "
                "(Raspberry Pi, Jetson Nano, Linux/Windows PC). "
                "Use 'direct' or 'tflite_micro' deployment for microcontrollers."
            )

        _v2_deployments = {'direct', 'tflite_micro', 'onnx_runtime'}

        if (deployment_approach in _v2_deployments
                and model_type in _V2_SUPPORTED_MODELS):
            # Build sketch/folder name matching create_output_folder_structure()
            # so the .ino filename equals the folder name (Arduino IDE requirement)
            num_features = len(model_data.get('feature_names', []))
            num_classes = len(model_data.get('classes', []))
            approach_prefix = {'tflite_micro': 'tflite_', 'onnx_runtime': 'onnx_'}.get(
                deployment_approach, '')
            base_name = (
                f"har_{approach_prefix}{model_type}_{platform}"
                f"_f{num_features}_c{num_classes}_{optimization}"
            )
            if quantization and quantization != 'none':
                base_name += f"_{quantization}"
            # Don't mutate the caller's dict
            model_data = dict(model_data)
            model_data['model_name'] = base_name
            gen = HARCodeGeneratorV2(
                model_data=model_data,
                platform=platform,
                optimization=optimization,
                overlap=overlap,
                confidence_threshold=confidence_threshold,
                smoothing_window=smoothing_window,
                quantization=quantization,
                deployment_approach=deployment_approach,
            )
            return gen.generate_files()
        # ----------------------------------------------------------------

        generator = CodeGeneratorFactory.create_generator(
            model_type, model_data, platform, optimization, overlap, quantization,
            deployment_approach, confidence_threshold=confidence_threshold,
            smoothing_window=smoothing_window)

        # Create organized filenames
        # For alternative deployment approaches, generate files differently
        if deployment_approach == 'tflite_micro':
            return _generate_tflite_files(generator, model_type, platform, model_data, optimization, quantization)
        elif deployment_approach == 'onnx_runtime':
            return _generate_onnx_files(generator, model_type, platform, model_data, optimization, quantization)

        if platform == 'arm_cortex_m':
            cortex_filename = create_organized_filename(
                model_type, platform, 'cortex_source', model_data, optimization, quantization)
            return {
                cortex_filename: generator.generate_implementation(
                    cortex_filename)
            }
        elif platform == 'micropython':
            # MicroPython: module (.py) + example (.py)
            source_filename = create_organized_filename(
                model_type, platform, 'source', model_data, optimization, quantization)
            sketch_filename = create_organized_filename(
                model_type, platform, 'sketch', model_data, optimization, quantization)
            return {
                source_filename: generator.generate_implementation(source_filename),
                sketch_filename: generator.generate_example_sketch(source_filename),
            }
        else:
            # All other platforms: header + source + example (extensions vary by platform)
            header_filename = create_organized_filename(
                model_type, platform, 'header', model_data, optimization, quantization)
            source_filename = create_organized_filename(
                model_type, platform, 'source', model_data, optimization, quantization)
            sketch_filename = create_organized_filename(
                model_type, platform, 'sketch', model_data, optimization, quantization)

            return {
                header_filename: generator.generate_header(),
                source_filename: generator.generate_implementation(header_filename),
                sketch_filename: generator.generate_example_sketch(
                    header_filename)
            }

    except (ValidationError, ModelDataError, OptimizationError) as e:
        # Re-raise validation errors with context
        raise type(e)(f"Code generation failed: {str(e)}") from e
    except Exception as e:
        # Wrap unexpected errors
        raise RuntimeError(
            f"Unexpected error generating code for {model_type}: {str(e)}") from e


def _generate_tflite_files(generator, model_type: str, platform: str,
                           model_data: Dict[str, Any],
                           optimization: str,
                           quantization: str = 'none') -> Dict[str, str]:
    """Generate file set for TFLite Micro deployment."""
    # Build descriptive base name matching direct approach convention
    num_features = len(model_data.get(
        'feature_names', [])) if model_data else 0
    num_classes = len(model_data.get('classes', [])) if model_data else 0

    base_name = f"har_tflite_{model_type}_{platform}"
    if num_features > 0:
        base_name += f"_f{num_features}"
    if num_classes > 0:
        base_name += f"_c{num_classes}"
    base_name += f"_{optimization}"
    if quantization and quantization != 'none':
        base_name += f"_{quantization}"

    header_filename = f"{base_name}.h"
    source_filename = f"{base_name}.cpp"

    if platform in ('arduino', 'seeed_xiao', 'esp32', 'm5stack', 'teensy'):
        sketch_filename = f"{base_name}.ino"
    elif platform == 'generic_c':
        sketch_filename = f"{base_name}_main.c"
    else:
        sketch_filename = f"{base_name}_main.cpp"

    files = {
        header_filename: generator.generate_header(),
        source_filename: generator.generate_implementation(header_filename),
        sketch_filename: generator.generate_example_sketch(header_filename),
    }

    # Arduino IDE build options: work around Arduino_TensorFlowLite
    # stl_emulation.h const-member assignment bug with newer GCC versions.
    # build_opt.h is automatically picked up by Arduino IDE 1.8.13+.
    if platform in ('arduino', 'seeed_xiao', 'esp32', 'm5stack', 'teensy'):
        # build_opt.h: each line is passed verbatim as a compiler flag.
        # NO comments allowed — they would be treated as file paths.
        files['build_opt.h'] = '-fpermissive\n'

    # Also save the raw .tflite model file if available
    if hasattr(generator, '_tflite_bytes') and generator._tflite_bytes:
        files[f"{base_name}.tflite"] = generator._tflite_bytes

    return files


def _generate_onnx_files(generator, model_type: str, platform: str,
                         model_data: Dict[str, Any],
                         optimization: str,
                         quantization: str = 'none') -> Dict[str, str]:
    """Generate file set for ONNX Runtime deployment."""
    # Build descriptive base name matching direct approach convention
    num_features = len(model_data.get(
        'feature_names', [])) if model_data else 0
    num_classes = len(model_data.get('classes', [])) if model_data else 0

    base_name = f"har_onnx_{model_type}_{platform}"
    if num_features > 0:
        base_name += f"_f{num_features}"
    if num_classes > 0:
        base_name += f"_c{num_classes}"
    base_name += f"_{optimization}"
    if quantization and quantization != 'none':
        base_name += f"_{quantization}"

    header_filename = f"{base_name}.h"
    source_filename = f"{base_name}.cpp"

    if platform in ('arduino', 'seeed_xiao', 'esp32', 'm5stack', 'teensy'):
        sketch_filename = f"{base_name}.ino"
    elif platform in ('generic_c', 'esp_idf', 'zephyr'):
        sketch_filename = f"{base_name}_main.c"
    else:
        sketch_filename = f"{base_name}_main.cpp"

    files = {
        header_filename: generator.generate_header(),
        source_filename: generator.generate_implementation(header_filename),
        sketch_filename: generator.generate_example_sketch(header_filename),
    }

    # Also save the raw .onnx model file if available
    if hasattr(generator, 'get_onnx_bytes'):
        onnx_bytes = generator.get_onnx_bytes()
        if onnx_bytes:
            files[f"{base_name}.onnx"] = onnx_bytes

    return files


# Legacy function for compatibility - returns tuple format
def generate_deployment_code_files(model_type: str, model_data: Dict[str, Any],
                                   platform: str = 'arduino') -> tuple:
    """
    Generate deployment code returning header and source as tuple.

    Args:
        model_type: Type of model to generate code for
        model_data: Model parameters and data
        platform: Target platform

    Returns:
        Tuple of (header_content, source_content)
    """
    generator = CodeGeneratorFactory.create_generator(
        model_type, model_data, platform)

    header_content = generator.generate_header()
    # For legacy compatibility, use generic header name
    source_content = generator.generate_implementation("har_model.h")

    return header_content, source_content


def generate_and_save_deployment_code(model_type: str, model_data: Dict[str, Any],
                                      platform: str = 'arduino',
                                      output_dir: str = 'generated_code',
                                      optimization: str = 'balanced',
                                      overlap: float = 0.5,
                                      quantization: str = 'none',
                                      deployment_approach: str = 'direct',
                                      confidence_threshold: float = 0.6,
                                      smoothing_window: int = 1) -> Dict[str, str]:
    """
    Generate deployment code and save to organized folder structure.

    Args:
        model_type: Type of model to generate code for
        model_data: Model parameters and data
        platform: Target platform
        output_dir: Base output directory for generated files
        optimization: Optimization strategy ('accuracy', 'speed', 'power', 'balanced')
        overlap: Window overlap percentage (0.0 to 0.99)
        quantization: Weight quantization mode ('none', 'int8', 'int16', 'float16')
        deployment_approach: Deployment approach ('direct', 'tflite_micro', 'onnx_runtime')

    Returns:
        Dictionary with full file paths as keys and success messages as values
    """
    # NOTE: Do NOT call extract_real_model_parameters() here!
    # It is already called inside generate_deployment_code() below.
    # Calling it twice causes a bug: the 1st call reorders feature_names to C++ order,
    # but the 2nd call re-extracts fresh scaler values (alphabetical order) and then
    # skips reordering because feature_names already appear to match C++ order.

    # Create organized folder structure
    folder_path = create_output_folder_structure(
        output_dir, model_type, platform, model_data, optimization,
        deployment_approach, quantization)

    # Generate code with organized naming and optimization
    generated_code = generate_deployment_code(
        model_type, model_data, platform, optimization, overlap, quantization,
        deployment_approach, confidence_threshold=confidence_threshold,
        smoothing_window=smoothing_window)

    # Save files and return file paths
    saved_files = {}

    for filename, content in generated_code.items():
        full_path = os.path.join(folder_path, filename)

        # Handle binary content (e.g., .onnx, .tflite files)
        if isinstance(content, bytes):
            try:
                with open(full_path, 'wb') as f:
                    f.write(content)
                saved_files[full_path] = f"✅ Successfully saved {len(content)} bytes (binary)"
            except Exception as e:
                saved_files[full_path] = f"❌ Error saving file: {str(e)}"
        else:
            try:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                saved_files[full_path] = f"✅ Successfully saved {len(content)} characters"
            except Exception as e:
                saved_files[full_path] = f"❌ Error saving file: {str(e)}"

    return saved_files


def get_deployment_info(model_type: str, model_data: Dict[str, Any],
                        platform: str = 'arduino', optimization: str = 'balanced') -> Dict[str, Any]:
    """
    Get information about what files would be generated without actually generating them.

    Args:
        model_type: Type of model
        model_data: Model parameters and data
        platform: Target platform
        optimization: Optimization level

    Returns:
        Dictionary with file information
    """
    num_features = len(model_data.get('feature_names', []))
    num_classes = len(model_data.get('classes', []))

    info = {
        'model_type': model_type,
        'platform': platform,
        'optimization': optimization,
        'features': num_features,
        'classes': num_classes,
        'folder_structure': f"{model_type}_models/{platform}/",
        'files': []
    }

    if platform == 'arm_cortex_m':
        cortex_filename = create_organized_filename(
            model_type, platform, 'cortex_source', model_data, optimization)
        info['files'].append({
            'filename': cortex_filename,
            'type': 'ARM Cortex-M Source File',
            'description': f'Optimized C code for {model_type} model on ARM Cortex-M ({optimization} optimization)'
        })
    else:
        header_filename = create_organized_filename(
            model_type, platform, 'header', model_data, optimization)
        source_filename = create_organized_filename(
            model_type, platform, 'source', model_data, optimization)
        sketch_filename = create_organized_filename(
            model_type, platform, 'sketch', model_data, optimization)

        info['files'].extend([
            {
                'filename': header_filename,
                'type': 'Header File',
                'description': f'Header declarations for {model_type} model on {platform} ({optimization} optimization)'
            },
            {
                'filename': source_filename,
                'type': 'Source File',
                'description': f'Implementation of {model_type} model for {platform} ({optimization} optimization)'
            },
            {
                'filename': sketch_filename,
                'type': 'Arduino Sketch',
                'description': f'Example Arduino sketch demonstrating {model_type} model usage ({optimization} optimization)'
            }
        ])

    return info


def analyze_resource_requirements(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze memory and computational requirements for edge deployment.

    Args:
        model_data: Trained model data

    Returns:
        Resource analysis results
    """

    num_features = len(model_data.get('feature_names', []))
    num_classes = len(model_data.get('classes', []))
    model_type = model_data.get('model_type', 'unknown')

    # Estimate memory requirements
    feature_memory = num_features * 4  # 4 bytes per float feature
    # Sensor buffer (100 samples, 6 axes, 4 bytes each)
    buffer_memory = 100 * 6 * 4

    # Model-specific memory estimation
    if model_type == 'random_forest':
        # Rough estimate: trees * nodes * parameters
        model_memory = 1000 * 50 * 8  # Conservative estimate
    elif model_type == 'svm':
        # Support vectors * features * parameters
        model_memory = 100 * num_features * 8
    elif model_type == 'neural_network':
        # Weights and biases estimation
        model_memory = num_features * 100 * 4 + 100 * num_classes * 4
    else:
        model_memory = 1000  # Default estimate

    total_memory = feature_memory + buffer_memory + model_memory

    # Computational complexity estimation
    operations_per_prediction = num_features * 10  # Rough estimate

    return {
        'memory_requirements': {
            'feature_memory_bytes': feature_memory,
            'buffer_memory_bytes': buffer_memory,
            'model_memory_bytes': model_memory,
            'total_memory_bytes': total_memory,
            'total_memory_kb': total_memory // 1024
        },
        'computational_requirements': {
            'operations_per_prediction': operations_per_prediction,
            'estimated_inference_time_ms': operations_per_prediction / 1000,  # Rough estimate
            'power_consumption_estimate': 'Low' if total_memory < 32768 else 'Medium'
        },
        'platform_compatibility': {
            'arduino_uno': total_memory < 2048,
            'arduino_nano_33': total_memory < 262144,
            'seeed_xiao_nrf52840': total_memory < 262144,
            'esp32': total_memory < 524288
        }
    }
