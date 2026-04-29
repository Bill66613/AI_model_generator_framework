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
            # For tree-based and SVM: go through ONNX → TF → TFLite
            return self._convert_via_onnx(quantization, representative_data)
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
        import tensorflow as tf

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
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(n_features,)))

        # NOTE: Do NOT embed the scaler here as a Normalization layer.
        # The C++ har_predict() wrapper applies StandardScaler before calling the model.
        # Embedding the scaler would cause double-scaling (C++ scales, then TFLite scales again).

        # Hidden layers: Dense + ReLU
        for i, (w, b) in enumerate(zip(coefs[:-1], intercepts[:-1])):
            layer = tf.keras.layers.Dense(
                w.shape[1],
                activation='relu',
                name=f'hidden_{i}'
            )
            model.add(layer)

        # Output layer: Dense + Softmax
        output_layer = tf.keras.layers.Dense(
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
            if isinstance(keras_layer, tf.keras.layers.Dense):
                w = coefs[layer_idx].astype(np.float32)
                b = intercepts[layer_idx].astype(np.float32)
                keras_layer.set_weights([w, b])
                layer_idx += 1

        logger.info(f"Built Keras MLP with {len(coefs)} layers, "
                     f"input={n_features}, output={n_classes}")
        return model

    def _build_keras_cnn(self):
        """Build a Keras 1D-CNN equivalent to the trained PyTorch CNN."""
        import tensorflow as tf

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

        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(window_size, n_channels)),

            # Conv block 1
            tf.keras.layers.Conv1D(32, kernel_size=5, padding='same',
                                   activation='relu', name='conv1'),

            # Conv block 2
            tf.keras.layers.Conv1D(64, kernel_size=5, padding='same',
                                   activation='relu', name='conv2'),
            tf.keras.layers.MaxPool1D(pool_size=2, name='maxpool'),

            # Conv block 3
            tf.keras.layers.Conv1D(128, kernel_size=3, padding='same',
                                   activation='relu', name='conv3'),

            # Global average pooling
            tf.keras.layers.GlobalAveragePooling1D(name='global_avg_pool'),

            # Dense layers
            tf.keras.layers.Dense(64, activation='relu', name='dense1'),
            tf.keras.layers.Dense(n_classes, activation='softmax', name='output'),
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
        import tensorflow as tf

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

            if isinstance(keras_layer, tf.keras.layers.Conv1D):
                # PyTorch: (out_ch, in_ch, kernel) → Keras: (kernel, in_ch, out_ch)
                w = np.transpose(w, (2, 1, 0)).astype(np.float32)
            elif isinstance(keras_layer, tf.keras.layers.Dense):
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

    def _convert_via_onnx(self, quantization: str,
                          representative_data: Optional[np.ndarray]) -> bytes:
        """Convert model via ONNX → TF → TFLite pipeline (for RF, SVM)."""
        import tensorflow as tf

        # First convert to ONNX
        from .onnx_converter import ONNXConverter
        onnx_converter = ONNXConverter(
            self.model_object, self.model_type,
            self.feature_names, self.classes, self.model_params
        )
        onnx_bytes = onnx_converter.convert()

        # Then ONNX → TF SavedModel
        try:
            import onnx
            from onnx_tf.backend import prepare
        except ImportError:
            try:
                import onnx2tf
                return self._convert_onnx_via_onnx2tf(
                    onnx_bytes, quantization, representative_data
                )
            except ImportError:
                raise ImportError(
                    "Converting RF/SVM to TFLite requires onnx-tf or onnx2tf. "
                    "Install with: pip install onnx-tf  OR  pip install onnx2tf"
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
        """Alternative ONNX → TFLite using onnx2tf library."""
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

            # Find the saved model and convert to TFLite
            saved_model_path = output_path
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
            self._apply_quantization(converter, quantization, representative_data)
            self._tflite_bytes = converter.convert()

        self._conversion_info = {
            'strategy': 'onnx2tf_pipeline',
            'quantization': quantization,
            'size_bytes': len(self._tflite_bytes),
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
                # Generate synthetic representative data centred around 0
                # (features are StandardScaler-normalised, so ~N(0,1) is realistic)
                n_samples = 200
                if is_cnn:
                    representative_data = np.random.normal(
                        0, 1, (n_samples, window_size, n_channels)
                    ).astype(np.float32)
                else:
                    representative_data = np.random.normal(
                        0, 1, (n_samples, n_features)
                    ).astype(np.float32)
                logger.warning(
                    "No representative data for INT8 quantization — using synthetic "
                    "N(0,1) samples.  For better accuracy, provide real calibration data."
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

    def to_c_array(self, variable_name: str = 'g_model') -> str:
        """
        Convert TFLite model bytes to a C byte array for embedding.

        This is the key output for TFLite Micro deployment — the model
        is embedded as a const unsigned char array in the firmware.

        Args:
            variable_name: C variable name for the array

        Returns:
            C source code string with the model byte array
        """
        if self._tflite_bytes is None:
            self.convert()

        data = self._tflite_bytes
        lines = []
        lines.append(f"// TFLite model - {len(data)} bytes")
        lines.append(f"// Generated by HAR Edge Deployment Framework")
        lines.append(f"// Model type: {self.model_type}")
        lines.append(f"// Classes: {', '.join(self.classes)}")
        lines.append(f"")
        lines.append(f"#include <cstdint>")
        lines.append(f"")
        lines.append(f"alignas(16) const unsigned char {variable_name}[] = {{")

        # Format bytes in rows of 12
        for i in range(0, len(data), 12):
            chunk = data[i:i+12]
            hex_values = ', '.join(f'0x{b:02x}' for b in chunk)
            comma = ',' if i + 12 < len(data) else ''
            lines.append(f"    {hex_values}{comma}")

        lines.append(f"}};")
        lines.append(f"const unsigned int {variable_name}_len = {len(data)};")

        return '\n'.join(lines)

    def enumerate_ops(self) -> List[str]:
        """
        Enumerate the unique TFLite builtin operators used in the model.

        Returns a list of op names matching the MicroMutableOpResolver
        method names, e.g. ['Quantize', 'Conv2D', 'Reshape', ...].

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

            # Map to builtin codes
            unique_ops = set()
            for idx in used_opcode_indices:
                oc = op_codes[idx]
                code = oc.builtinCode if oc.builtinCode != 0 else oc.deprecatedBuiltinCode
                if code in BUILTIN_OP_NAMES:
                    unique_ops.add(BUILTIN_OP_NAMES[code])
                else:
                    logger.warning(f"Unknown TFLite op code {code} — add it to the resolver manually")
                    unique_ops.add(f'UnknownOp{code}')

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
