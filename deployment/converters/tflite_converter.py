"""
TensorFlow Lite Model Converter

Converts sklearn and PyTorch HAR models to TFLite format for deployment
with TensorFlow Lite Micro on microcontrollers.

Conversion paths:
- sklearn → ONNX → TF SavedModel → TFLite
- PyTorch → ONNX → TF SavedModel → TFLite
- sklearn MLP → Keras equivalent → TFLite (direct path)

Requires: tensorflow (or tflite-model-maker), onnx, onnx_tf (or onnx2tf)
"""

import logging
import numpy as np
import struct
import os
import tempfile
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def _get_keras():
    """Return the keras module, compatible with TF ≤2.15 (tf.keras) and TF 2.16+ (standalone keras)."""
    # Try standalone keras first (required for TF 2.16+; also works with TF 2.21+)
    try:
        import keras
        # Quick sanity check — keras must expose layers
        _ = keras.layers
        return keras
    except Exception:
        pass
    # Fallback: tf.keras for TF ≤ 2.15
    try:
        import tensorflow as tf
        keras = tf.keras
        _ = keras.layers
        return keras
    except Exception:
        pass
    raise ImportError(
        "Keras is not available. Install with: pip install tensorflow  "
        "or: pip install keras tensorflow"
    )


def check_tflite_dependencies() -> Dict[str, bool]:
    """Check which TFLite-related packages are available."""
    deps = {}
    try:
        import tensorflow as tf  # noqa: F401
        deps['tensorflow'] = True
        deps['tf_version'] = tf.__version__
    except ImportError:
        deps['tensorflow'] = False

    try:
        import onnx  # noqa: F401
        deps['onnx'] = True
    except ImportError:
        deps['onnx'] = False

    try:
        import onnx_tf  # noqa: F401
        deps['onnx_tf'] = True
    except ImportError:
        deps['onnx_tf'] = False

    try:
        import onnx2tf  # noqa: F401
        deps['onnx2tf'] = True
    except ImportError:
        deps['onnx2tf'] = False

    return deps


class TFLiteConverter:
    """
    Converts trained HAR models to TFLite format.

    Supports multiple conversion strategies:
    1. Direct Keras reconstruction (for MLP/NN models)
    2. ONNX → TF → TFLite pipeline (for all model types)
    3. Post-training quantization (INT8, FLOAT16)

    Usage:
        converter = TFLiteConverter(model_object, model_type, feature_names, classes)
        tflite_bytes = converter.convert(quantization='none')
        converter.save('model.tflite')
    """

    def __init__(self, model_object, model_type: str,
                 feature_names: List[str], classes: List[str],
                 model_params: Optional[Dict[str, Any]] = None):
        """
        Args:
            model_object: Trained EdgeMLModel (has .model, .scaler, .trainer)
            model_type: Model identifier string
            feature_names: Feature name list
            classes: Class label list
            model_params: Training parameters dict
        """
        self.model_object = model_object
        self.model_type = model_type
        self.feature_names = feature_names
        self.classes = classes
        self.model_params = model_params or {}
        self._tflite_bytes = None
        self._conversion_info = {}

    def convert(self, quantization: str = 'none',
                representative_data: Optional[np.ndarray] = None) -> bytes:
        """
        Convert model to TFLite format.

        Args:
            quantization: 'none', 'float16', 'int8', 'int16'
            representative_data: Calibration data for INT8 quantization.
                                 Shape: (num_samples, n_features) for MLP
                                        (num_samples, window_size, n_channels) for CNN

        Returns:
            TFLite model as bytes

        Raises:
            ImportError: If TensorFlow not installed
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "TFLite conversion requires TensorFlow. "
                "Install with: pip install tensorflow\n"
                "For lighter install: pip install tflite-runtime (inference only)"
            )

        # Strategy 1: Direct Keras reconstruction for MLP/NN
        if self.model_type in ('neural_network', 'pytorch_mlp'):
            keras_model = self._build_keras_mlp()
        elif self.model_type == 'pytorch_cnn':
            keras_model = self._build_keras_cnn()
        elif self.model_type in ('random_forest', 'svm'):
            # RF/SVM → TFLite conversion strategies:
            #
            # 1. ONNX pipeline (attempted first): sklearn → ONNX → onnx2tf → TFLite
            #    Known limitation: onnx2tf doesn't support ONNX-ML domain ops
            #    (TreeEnsembleClassifier for RF, SVMClassifier for SVM).
            #    This path will work if onnx2tf adds support in the future.
            #    See: https://pypi.org/project/onnx2tf/
            #
            # 2. Keras surrogate (fallback): Train a small Keras MLP to mimic
            #    the RF/SVM decision boundaries via knowledge distillation,
            #    then convert that MLP to TFLite. Only requires TensorFlow.
            #
            # For production deployment of RF/SVM, Direct C/C++ Code Generation
            # is recommended (faithful, no approximation, smaller binary).
            onnx_result = self._try_convert_via_onnx(quantization, representative_data)
            if onnx_result is not None:
                return onnx_result
            # Fallback: Keras surrogate via knowledge distillation.
            # NOTE: The resulting TFLite model is an approximation of the
            # original RF/SVM — it may have lower accuracy.  For production
            # use, Direct C/C++ Code Generation is recommended (faithful,
            # no approximation, smaller binary).
            keras_model = self._build_keras_surrogate()
            # Mark the conversion as surrogate so callers can surface the warning
            self._conversion_info['strategy'] = 'keras_surrogate'
            self._conversion_info['is_approximation'] = True
            self._conversion_info['approximation_warning'] = (
                f'{self.model_type.upper()} converted via Keras surrogate (knowledge distillation). '
                'This is an approximation — accuracy may differ from the original model. '
                'Use Direct C/C++ Code Generation for a faithful deployment.'
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Convert Keras model to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

        # Apply quantization
        self._apply_quantization(converter, quantization, representative_data)

        # Convert
        self._tflite_bytes = converter.convert()

        self._conversion_info = {
            'strategy': 'direct_keras',
            'quantization': quantization,
            'size_bytes': len(self._tflite_bytes),
            'size_kb': len(self._tflite_bytes) / 1024,
        }

        logger.info(
            f"Converted {self.model_type} to TFLite ({quantization}): "
            f"{len(self._tflite_bytes)} bytes"
        )
        return self._tflite_bytes

    def _build_keras_mlp(self):
        """Build a Keras MLP equivalent to the trained sklearn/PyTorch MLP."""
        keras = _get_keras()

        n_features = len(self.feature_names)
        n_classes = len(self.classes)

        # Extract weights from the model
        weights_dict = self._extract_mlp_weights()

        if weights_dict is None:
            raise ValueError("Could not extract MLP weights from model")

        coefs = weights_dict.get('coefs_', [])
        intercepts = weights_dict.get('intercepts_', [])

        if not coefs or not intercepts:
            raise ValueError("MLP weights (coefs_, intercepts_) not found")

        # Build Keras Sequential model
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(n_features,)))

        # NOTE: Do NOT embed the scaler here as a Normalization layer.
        # The C++ har_predict() wrapper applies StandardScaler before calling the model.
        # Embedding the scaler would cause double-scaling (C++ scales, then TFLite scales again).

        # Hidden layers: Dense + ReLU
        for i, (w, b) in enumerate(zip(coefs[:-1], intercepts[:-1])):
            layer = keras.layers.Dense(
                w.shape[1],
                activation='relu',
                name=f'hidden_{i}'
            )
            model.add(layer)

        # Output layer: Dense + Softmax
        output_layer = keras.layers.Dense(
            n_classes,
            activation='softmax',
            name='output'
        )
        model.add(output_layer)

        # Build the model by running a dummy input
        dummy = np.zeros((1, n_features), dtype=np.float32)
        model(dummy)

        # Now set weights to match the trained model
        layer_idx = 0
        for keras_layer in model.layers:
            if isinstance(keras_layer, keras.layers.Dense):
                w = coefs[layer_idx].astype(np.float32)
                b = intercepts[layer_idx].astype(np.float32)
                keras_layer.set_weights([w, b])
                layer_idx += 1

        logger.info(f"Built Keras MLP with {len(coefs)} layers, "
                     f"input={n_features}, output={n_classes}")
        return model

    def _build_keras_cnn(self):
        """Build a Keras 1D-CNN equivalent to the trained PyTorch CNN."""
        keras = _get_keras()

        n_classes = len(self.classes)
        window_size = self.model_params.get('window_size_samples', 150)
        n_channels = self.model_params.get('n_channels', 6)

        # Extract CNN weights
        cnn_weights = self._extract_cnn_weights()
        if cnn_weights is None:
            raise ValueError("Could not extract CNN weights from model")

        # Build Keras model matching HARCNN architecture:
        # Conv1d(n_ch, 32, k=5, pad=2) → ReLU
        # Conv1d(32, 64, k=5, pad=2) → ReLU → MaxPool1d(2)
        # Conv1d(64, 128, k=3, pad=1) → ReLU → Dropout
        # GlobalAvgPool → Dense(128, 64) → ReLU → Dense(64, n_classes)

        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(window_size, n_channels)),

            # Conv block 1
            keras.layers.Conv1D(32, kernel_size=5, padding='same',
                                activation='relu', name='conv1'),

            # Conv block 2
            keras.layers.Conv1D(64, kernel_size=5, padding='same',
                                activation='relu', name='conv2'),
            keras.layers.MaxPool1D(pool_size=2, name='maxpool'),

            # Conv block 3
            keras.layers.Conv1D(128, kernel_size=3, padding='same',
                                activation='relu', name='conv3'),

            # Global average pooling
            keras.layers.GlobalAveragePooling1D(name='global_avg_pool'),

            # Dense layers
            keras.layers.Dense(64, activation='relu', name='dense1'),
            keras.layers.Dense(n_classes, activation='softmax', name='output'),
        ])

        # Build model
        dummy = np.zeros((1, window_size, n_channels), dtype=np.float32)
        model(dummy)

        # Transfer PyTorch weights to Keras
        # PyTorch Conv1d weights: (out_ch, in_ch, kernel) → Keras: (kernel, in_ch, out_ch)
        self._transfer_cnn_weights(model, cnn_weights)

        logger.info(f"Built Keras CNN: input=({window_size}, {n_channels}), "
                     f"output={n_classes}")
        return model

    def _transfer_cnn_weights(self, keras_model, pytorch_weights: Dict):
        """Transfer PyTorch CNN weights to Keras model (handles axis transposition)."""
        keras = _get_keras()

        layer_map = {
            'conv1': 'conv1',
            'conv2': 'conv2',
            'conv3': 'conv3',
            'dense1': 'fc1',
            'output': 'fc2',
        }

        for keras_name, pytorch_name in layer_map.items():
            keras_layer = keras_model.get_layer(keras_name)
            weight_key = f'{pytorch_name}.weight'
            bias_key = f'{pytorch_name}.bias'

            if weight_key not in pytorch_weights or bias_key not in pytorch_weights:
                logger.warning(f"Weights for {pytorch_name} not found, skipping")
                continue

            w = pytorch_weights[weight_key]
            b = pytorch_weights[bias_key]

            if isinstance(keras_layer, keras.layers.Conv1D):
                # PyTorch: (out_ch, in_ch, kernel) → Keras: (kernel, in_ch, out_ch)
                w = np.transpose(w, (2, 1, 0)).astype(np.float32)
            elif isinstance(keras_layer, keras.layers.Dense):
                # PyTorch: (out, in) → Keras: (in, out)
                w = w.T.astype(np.float32)

            b = b.astype(np.float32)
            keras_layer.set_weights([w, b])

    def _extract_mlp_weights(self) -> Optional[Dict]:
        """Extract MLP weights from the model object."""
        model = self.model_object

        # PyTorch MLP: check trainer export
        if hasattr(model, 'trainer') and model.trainer is not None:
            if hasattr(model.trainer, 'export_mlp_weights'):
                weights = model.trainer.export_mlp_weights()
                if weights:
                    return weights

        # Check model_object for pytorch_weights
        if hasattr(model, 'pytorch_weights'):
            return model.pytorch_weights

        # sklearn MLP: extract from MLPClassifier
        sklearn_model = model.model
        if hasattr(sklearn_model, 'coefs_') and hasattr(sklearn_model, 'intercepts_'):
            return {
                'coefs_': [c.copy() for c in sklearn_model.coefs_],
                'intercepts_': [b.copy() for b in sklearn_model.intercepts_],
            }

        return None

    def _extract_cnn_weights(self) -> Optional[Dict[str, np.ndarray]]:
        """Extract CNN weights as numpy arrays from PyTorch model."""
        model = self.model_object

        state_dict = None
        if hasattr(model, 'trainer') and model.trainer is not None:
            state_dict = model.trainer.model.state_dict()
        elif hasattr(model, 'model'):
            import torch
            if isinstance(model.model, torch.nn.Module):
                state_dict = model.model.state_dict()

        if state_dict is None:
            return None

        return {k: v.cpu().numpy() for k, v in state_dict.items()}

    def _build_keras_surrogate(self):
        """Build a Keras MLP surrogate that mimics the RF/SVM predictions.

        Instead of the fragile sklearn → ONNX → TF → TFLite pipeline (which
        requires onnx-tf or onnx2tf with 10+ undeclared dependencies), we use
        knowledge distillation: generate synthetic data, get the RF/SVM's
        soft predictions, and train a small Keras MLP to reproduce them.

        The surrogate is trained on the RF/SVM's probability outputs, so it
        captures the decision boundaries. This approach:
        - Requires only TensorFlow (already a dep)
        - Works for any sklearn model with predict_proba()
        - Produces a clean TFLite model with standard ops
        """
        keras = _get_keras()
        import tensorflow as tf

        n_features = len(self.feature_names)
        n_classes = len(self.classes)

        # Get the sklearn model
        sklearn_model = self.model_object.model

        # Generate synthetic training data for distillation
        # Use the model's training data if available, otherwise create synthetic
        n_distill_samples = 10000
        rng = np.random.default_rng(42)

        # Try to get real feature ranges from the scaler
        if hasattr(self.model_object, 'scaler') and self.model_object.scaler is not None:
            scaler = self.model_object.scaler
            if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                # Generate data around the training distribution (scaled space)
                X_synth = rng.standard_normal((n_distill_samples, n_features)).astype(np.float32)
                # Also add uniform coverage of the feature space
                X_uniform = rng.uniform(-3, 3, (n_distill_samples, n_features)).astype(np.float32)
                X_synth = np.vstack([X_synth, X_uniform])
                # Unscale for model prediction
                X_for_pred = X_synth * scaler.scale_ + scaler.mean_
            else:
                X_synth = rng.standard_normal((n_distill_samples * 2, n_features)).astype(np.float32)
                X_for_pred = X_synth
        else:
            X_synth = rng.standard_normal((n_distill_samples * 2, n_features)).astype(np.float32)
            X_for_pred = X_synth

        # Get soft labels from the teacher model
        if hasattr(sklearn_model, 'predict_proba'):
            y_soft = sklearn_model.predict_proba(X_for_pred).astype(np.float32)
        else:
            # SVM with decision_function — convert to one-hot
            y_hard = sklearn_model.predict(X_for_pred)
            y_soft = np.zeros((len(y_hard), n_classes), dtype=np.float32)
            for i, label in enumerate(y_hard):
                idx = list(sklearn_model.classes_).index(label)
                y_soft[i, idx] = 1.0

        # Build a compact Keras MLP surrogate
        # Architecture: match complexity to the number of features/classes
        hidden_size = min(128, max(32, n_features * 2))
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(n_features,)),
            keras.layers.Dense(hidden_size, activation='relu'),
            keras.layers.Dense(hidden_size // 2, activation='relu'),
            keras.layers.Dense(n_classes, activation='softmax'),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train the surrogate (use scaled inputs so TFLite gets scaled data)
        logger.info(f"Training Keras surrogate for {self.model_type} "
                    f"({len(X_synth)} samples, {hidden_size} hidden units)...")
        model.fit(
            X_synth, y_soft,
            epochs=50,
            batch_size=256,
            verbose=0,
            validation_split=0.1,
            callbacks=[keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True)]
        )

        # Verify surrogate accuracy on the distillation data
        y_pred = model.predict(X_synth, verbose=0)
        teacher_classes = np.argmax(y_soft, axis=1)
        student_classes = np.argmax(y_pred, axis=1)
        agreement = np.mean(teacher_classes == student_classes)
        logger.info(f"Surrogate agreement with {self.model_type}: {agreement:.1%}")

        if agreement < 0.8:
            logger.warning(
                f"Low surrogate agreement ({agreement:.1%}). "
                f"TFLite model may have reduced accuracy compared to direct C++ generation."
            )

        self._conversion_info['surrogate_agreement'] = float(agreement)
        return model

    def _try_convert_via_onnx(self, quantization: str,
                              representative_data: Optional[np.ndarray]) -> Optional[bytes]:
        """Attempt ONNX → TFLite conversion; return None if deps unavailable.

        This is the preferred path for RF/SVM → TFLite as it faithfully
        preserves the original model's decision boundaries. Falls back to
        None if required packages (onnx2tf, skl2onnx, etc.) aren't installed.

        Required packages for full pipeline:
            pip install onnx skl2onnx onnx2tf sng4onnx sne4onnx \\
                        onnxsim ai-edge-litert flatbuffers ml-dtypes

        See: https://pypi.org/project/onnx2tf/ for onnx2tf docs.
        """
        try:
            import onnx  # noqa: F401
            import skl2onnx  # noqa: F401
        except ImportError:
            logger.debug("onnx/skl2onnx not available, skipping ONNX pipeline")
            return None

        # Check for onnx2tf or onnx-tf bridge
        has_onnx2tf = False
        has_onnx_tf = False
        try:
            from onnx2tf import convert as _  # noqa: F401
            has_onnx2tf = True
        except (ImportError, ModuleNotFoundError) as e:
            logger.debug(f"onnx2tf not available: {e}")
        try:
            from onnx_tf.backend import prepare as _  # noqa: F401
            has_onnx_tf = True
        except (ImportError, ModuleNotFoundError) as e:
            logger.debug(f"onnx-tf not available: {e}")

        if not has_onnx2tf and not has_onnx_tf:
            logger.debug("No ONNX→TF bridge available, skipping ONNX pipeline")
            return None

        # All deps available — attempt the conversion
        try:
            return self._convert_via_onnx(quantization, representative_data)
        except Exception as e:
            logger.warning(
                f"ONNX pipeline failed ({type(e).__name__}: {e}), "
                f"falling back to Keras surrogate"
            )
            return None

    def _convert_via_onnx(self, quantization: str,
                          representative_data: Optional[np.ndarray]) -> bytes:
        """Convert model via ONNX → TF → TFLite pipeline (for RF, SVM)."""
        import tensorflow as tf

        # Check which bridge is available
        try:
            import onnx2tf  # noqa: F401
            use_onnx2tf = True
        except ImportError:
            use_onnx2tf = False

        # First convert to ONNX
        # Note: onnx2tf doesn't support 'Scaler' ONNX op, so skip scaler
        # in the ONNX graph when using onnx2tf (model expects pre-scaled input)
        from .onnx_converter import ONNXConverter
        onnx_converter = ONNXConverter(
            self.model_object, self.model_type,
            self.feature_names, self.classes, self.model_params
        )
        onnx_bytes = onnx_converter.convert(include_scaler=not use_onnx2tf)

        if use_onnx2tf:
            # onnx2tf available — any failure here is a real conversion error
            return self._convert_onnx_via_onnx2tf(
                onnx_bytes, quantization, representative_data
            )

        # Fallback: onnx-tf (legacy, slower)
        try:
            import onnx
            from onnx_tf.backend import prepare
        except ImportError:
            raise ImportError(
                "Converting RF/SVM to TFLite via ONNX requires onnx2tf or onnx-tf. "
                "Install with: pip install onnx2tf sng4onnx sne4onnx onnxsim "
                "ai-edge-litert flatbuffers ml-dtypes"
            )

        onnx_model = onnx.load_from_string(onnx_bytes)
        tf_rep = prepare(onnx_model)

        # Save as SavedModel then convert
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_model_path = os.path.join(tmpdir, 'saved_model')
            tf_rep.export_graph(saved_model_path)

            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
            self._apply_quantization(converter, quantization, representative_data)
            self._tflite_bytes = converter.convert()

        self._conversion_info = {
            'strategy': 'onnx_pipeline',
            'quantization': quantization,
            'size_bytes': len(self._tflite_bytes),
            'size_kb': len(self._tflite_bytes) / 1024,
        }
        return self._tflite_bytes

    def _convert_onnx_via_onnx2tf(self, onnx_bytes: bytes,
                                   quantization: str,
                                   representative_data: Optional[np.ndarray]) -> bytes:
        """Convert ONNX → TFLite using onnx2tf library.

        onnx2tf v2.4.0+ uses flatbuffer_direct as the default backend,
        which is ~100x faster than the legacy tf_converter path and
        produces TFLite directly without a SavedModel intermediate.

        See: https://pypi.org/project/onnx2tf/
        Required deps: onnx2tf sng4onnx sne4onnx onnxsim ai-edge-litert
                       flatbuffers ml-dtypes psutil
        """
        import tensorflow as tf
        import onnx2tf

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, 'model.onnx')
            with open(onnx_path, 'wb') as f:
                f.write(onnx_bytes)

            output_path = os.path.join(tmpdir, 'output')
            onnx2tf.convert(
                input_onnx_file_path=onnx_path,
                output_folder_path=output_path,
                non_verbose=True,
            )

            # onnx2tf flatbuffer_direct outputs {model_name}_float32.tflite
            # Also try saved_model path as fallback
            tflite_candidates = [
                f for f in os.listdir(output_path)
                if f.endswith('.tflite') and 'float32' in f
            ]

            if tflite_candidates:
                # Direct TFLite output from flatbuffer_direct backend
                tflite_path = os.path.join(output_path, tflite_candidates[0])
                with open(tflite_path, 'rb') as f:
                    self._tflite_bytes = f.read()

                # If quantization requested, re-convert from SavedModel
                if quantization != 'none':
                    saved_model_path = output_path
                    if os.path.exists(os.path.join(output_path, 'saved_model.pb')):
                        converter = tf.lite.TFLiteConverter.from_saved_model(output_path)
                        self._apply_quantization(converter, quantization, representative_data)
                        self._tflite_bytes = converter.convert()
            else:
                # Legacy path: SavedModel exists, convert to TFLite
                converter = tf.lite.TFLiteConverter.from_saved_model(output_path)
                self._apply_quantization(converter, quantization, representative_data)
                self._tflite_bytes = converter.convert()

        self._conversion_info = {
            'strategy': 'onnx2tf_pipeline',
            'quantization': quantization,
            'size_bytes': len(self._tflite_bytes),
            'size_kb': len(self._tflite_bytes) / 1024,
        }
        return self._tflite_bytes

    def _apply_quantization(self, converter, quantization: str,
                            representative_data: Optional[np.ndarray] = None):
        """Apply quantization settings to a TFLite converter.

        TFLite Micro only supports two quantization modes:
          - float32  (no quantization)
          - Full INT8  (both weights AND activations in INT8, float32 I/O allowed)

        Dynamic-range / hybrid quantization (weights INT8, activations float32) is
        NOT supported by TFLite Micro and causes "Hybrid models are not supported"
        at AllocateTensors() time.  We therefore always apply full INT8 by generating
        synthetic representative data when real calibration data is unavailable.
        """
        import tensorflow as tf

        if quantization == 'none':
            return

        if quantization == 'float16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            return

        if quantization in ('int8', 'int16'):
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Determine input shape for representative data generation
            n_features = len(self.feature_names)
            window_size = self.model_params.get('window_size_samples', 150)
            n_channels = self.model_params.get('n_channels', 6)
            is_cnn = self.model_type == 'pytorch_cnn'

            if representative_data is None:
                # Generate synthetic representative data matching expected input ranges.
                n_samples = 200
                if is_cnn:
                    # CNN input is RAW IMU sensor data (NOT standardized):
                    #   Accelerometer: typically ±20 m/s² (gravity ~9.8 + dynamic)
                    #   Gyroscope: typically ±500 deg/s (most motion ±50)
                    # Using N(0,1) here would clip real accel values (~9.8) to quantization
                    # range, producing near-uniform (garbage) output probabilities.
                    representative_data = np.zeros(
                        (n_samples, window_size, n_channels), dtype=np.float32)
                    rng = np.random.default_rng(42)
                    for i in range(n_samples):
                        # Accelerometer channels (0,1,2): simulate real sensor range
                        # Gravity + dynamic acceleration → range roughly -20 to +20 m/s²
                        representative_data[i, :, 0] = rng.uniform(-2, 2, window_size)       # aX
                        representative_data[i, :, 1] = rng.uniform(7, 12, window_size)       # aY (gravity axis)
                        representative_data[i, :, 2] = rng.uniform(-4, 0, window_size)       # aZ
                        # Gyroscope channels (3,4,5): range roughly -200 to +200 deg/s
                        for ch in range(3, min(n_channels, 6)):
                            representative_data[i, :, ch] = rng.uniform(-50, 50, window_size)
                        # Fill remaining channels if > 6
                        for ch in range(6, n_channels):
                            representative_data[i, :, ch] = rng.uniform(-10, 10, window_size)
                else:
                    representative_data = np.random.normal(
                        0, 1, (n_samples, n_features)).astype(np.float32)
                logger.warning(
                    "No representative data for INT8 quantization — using synthetic "
                    "calibration data.  For better accuracy, provide real calibration data."
                )

            if quantization == 'int8':
                # Capture representative_data in closure (must be a stable reference)
                _rep_data = representative_data

                def representative_dataset():
                    for i in range(min(200, len(_rep_data))):
                        yield [_rep_data[i:i+1].astype(np.float32)]

                converter.representative_dataset = representative_dataset
                # Full INT8 for all ops — required by TFLite Micro.
                # experimental_new_quantizer forces calibration-based (not dynamic-range) path.
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                # Keep I/O as float32 so the sketch feeds plain floats.
                # TFLite Micro will insert dequantize/quantize ops at the boundary.
                converter.inference_input_type = tf.float32
                converter.inference_output_type = tf.float32
                # Force calibration-based full integer quantization (avoids hybrid fallback)
                try:
                    converter.experimental_new_quantizer = True
                except AttributeError:
                    pass  # Older TF versions — try anyway without it

            elif quantization == 'int16':
                _rep_data = representative_data

                def representative_dataset():
                    for i in range(min(200, len(_rep_data))):
                        yield [_rep_data[i:i+1].astype(np.float32)]

                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
                ]
                try:
                    converter.experimental_new_quantizer = True
                except AttributeError:
                    pass

    def save(self, filepath: str) -> str:
        """Save the TFLite model to a file."""
        if self._tflite_bytes is None:
            self.convert()

        with open(filepath, 'wb') as f:
            f.write(self._tflite_bytes)

        logger.info(f"Saved TFLite model to {filepath}")
        return filepath

    def get_model_size(self) -> int:
        """Get TFLite model size in bytes."""
        if self._tflite_bytes is None:
            self.convert()
        return len(self._tflite_bytes)

    def get_conversion_info(self) -> Dict[str, Any]:
        """Get conversion details."""
        return self._conversion_info.copy()

    def to_c_array(self, variable_name: str = 'g_model', platform: str = 'arduino') -> str:
        """
        Convert TFLite model bytes to a C byte array for embedding.

        This is the key output for TFLite Micro deployment — the model
        is embedded as a const unsigned char array in the firmware.

        On ESP32-based platforms the array is placed in flash via PROGMEM
        to avoid exhausting the limited ~320 KB DRAM.

        Args:
            variable_name: C variable name for the array
            platform: Target platform (affects storage qualifiers)

        Returns:
            C source code string with the model byte array
        """
        if self._tflite_bytes is None:
            self.convert()

        # ESP32/M5Stack: const alone does NOT guarantee flash placement on
        # Xtensa — PROGMEM (maps to __attribute__((section(".rodata")))) is
        # required to keep the model out of DRAM.
        esp32_platforms = ('esp32', 'm5stack', 'esp_idf')
        use_progmem = platform in esp32_platforms
        storage_attr = 'PROGMEM ' if use_progmem else ''

        data = self._tflite_bytes
        lines = []
        lines.append(f"// TFLite model - {len(data)} bytes")
        lines.append(f"// Generated by HAR Edge Deployment Framework")
        lines.append(f"// Model type: {self.model_type}")
        lines.append(f"// Classes: {', '.join(self.classes)}")
        if use_progmem:
            lines.append(f"// Storage: PROGMEM (flash) — avoids DRAM exhaustion on ESP32")
        lines.append(f"")
        lines.append(f"#include <cstdint>")
        if use_progmem:
            lines.append(f"#include <pgmspace.h>")
        lines.append(f"")
        lines.append(f"alignas(16) const unsigned char {variable_name}[] {storage_attr}= {{")

        # Format bytes in rows of 12
        for i in range(0, len(data), 12):
            chunk = data[i:i+12]
            hex_values = ', '.join(f'0x{b:02x}' for b in chunk)
            comma = ',' if i + 12 < len(data) else ''
            lines.append(f"    {hex_values}{comma}")

        lines.append(f"}};")
        lines.append(f"const unsigned int {variable_name}_len = {len(data)};")

        return '\n'.join(lines)

    def enumerate_ops(self) -> Optional[List[str]]:
        """
        Enumerate the unique TFLite builtin operators used in the model.

        Returns a list of op names matching the MicroMutableOpResolver
        method names, e.g. ['Quantize', 'Conv2D', 'Reshape', ...].

        Returns None when an unknown opcode is encountered, signalling that
        the caller should fall back to AllOpsResolver for safe compilation.

        These are needed to build a MicroMutableOpResolver<N> with exactly
        the ops the model requires.
        """
        if self._tflite_bytes is None:
            self.convert()

        data = self._tflite_bytes

        # Map BuiltinOperator enum values to MicroMutableOpResolver method suffixes
        BUILTIN_OP_NAMES = {
            0: 'Add',               # ADD
            1: 'AveragePool2D',     # AVERAGE_POOL_2D
            2: 'Concatenation',     # CONCATENATION
            3: 'Conv2D',            # CONV_2D
            4: 'DepthwiseConv2D',   # DEPTHWISE_CONV_2D
            6: 'Dequantize',        # DEQUANTIZE
            9: 'FullyConnected',    # FULLY_CONNECTED
            14: 'Logistic',         # LOGISTIC
            17: 'MaxPool2D',        # MAX_POOL_2D
            18: 'Mul',              # MUL
            22: 'Reshape',          # RESHAPE
            25: 'Softmax',          # SOFTMAX
            27: 'Sub',              # SUB
            29: 'Pad',              # PAD
            40: 'Mean',             # MEAN
            47: 'Maximum',          # MAXIMUM
            54: 'ArgMax',           # ARG_MAX
            56: 'TransposeConv',    # TRANSPOSE_CONV
            70: 'ExpandDims',       # EXPAND_DIMS
            80: 'Abs',              # ABS
            87: 'Tanh',             # TANH
            97: 'StridedSlice',     # STRIDED_SLICE
            114: 'Quantize',        # QUANTIZE
        }

        # Parse the flatbuffer to extract operator codes
        try:
            from tensorflow.lite.python import schema_py_generated as schema_fb
            buf = bytearray(data)
            model_fb = schema_fb.ModelT.InitFromPackedBuf(buf, 0)

            op_codes = model_fb.operatorCodes
            subgraph = model_fb.subgraphs[0]

            # Collect unique opcode indices used
            used_opcode_indices = set()
            for op in subgraph.operators:
                used_opcode_indices.add(op.opcodeIndex)

            # Map to builtin codes.
            # In the TFLite flatbuffers schema, `deprecated_builtin_code` is a byte
            # field (0-126, or 127 as PLACEHOLDER_FOR_GREATER_OP_CODES).
            # When deprecated_builtin_code == 127, the actual opcode lives in the
            # int32 `builtin_code` field (value >= 128).  For ops with codes < 127,
            # the int32 field defaults to 0 (ADD), so we must NOT use it as a
            # fallback indicator — ADD (0) is a perfectly valid opcode.
            # Reference: tensorflow/lite/schema/schema.fbs, OperatorCode table.
            PLACEHOLDER_FOR_GREATER_OP_CODES = 127
            unique_ops = set()
            for idx in used_opcode_indices:
                oc = op_codes[idx]
                if oc.deprecatedBuiltinCode == PLACEHOLDER_FOR_GREATER_OP_CODES:
                    # Extended opcode: read from the int32 field
                    code = oc.builtinCode
                else:
                    # Legacy byte field is authoritative
                    code = oc.deprecatedBuiltinCode
                if code in BUILTIN_OP_NAMES:
                    unique_ops.add(BUILTIN_OP_NAMES[code])
                else:
                    logger.warning(
                        f"Unknown TFLite op code {code} encountered. "
                        f"Falling back to AllOpsResolver for safe compilation. "
                        f"To use MicroMutableOpResolver (smaller binary), add op code {code} "
                        f"to BUILTIN_OP_NAMES in tflite_converter.py and re-generate."
                    )
                    # Return None so the generator falls back to AllOpsResolver.
                    # An incomplete MicroMutableOpResolver would compile but fail at runtime.
                    return None

            result = sorted(unique_ops)
            logger.info(f"TFLite model uses {len(result)} ops: {result}")
            return result

        except Exception as e:
            logger.warning(f"Could not enumerate TFLite ops: {e}")
            # Fallback: return common ops for the model type
            if self.model_type == 'pytorch_cnn':
                return ['Conv2D', 'Dequantize', 'ExpandDims', 'FullyConnected',
                        'MaxPool2D', 'Mean', 'Quantize', 'Reshape', 'Softmax']
            else:
                return ['Dequantize', 'FullyConnected', 'Quantize', 'Softmax']
