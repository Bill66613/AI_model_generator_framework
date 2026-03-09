"""
ONNX Model Converter

Converts sklearn and PyTorch HAR models to ONNX format for deployment
with ONNX Runtime on edge devices.

Requires: onnx, skl2onnx (for sklearn), torch (for PyTorch)
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


def check_onnx_dependencies() -> Dict[str, bool]:
    """Check which ONNX-related packages are available."""
    deps = {}
    try:
        import onnx  # noqa: F401
        deps['onnx'] = True
    except ImportError:
        deps['onnx'] = False

    try:
        import skl2onnx  # noqa: F401
        deps['skl2onnx'] = True
    except ImportError:
        deps['skl2onnx'] = False

    try:
        import torch  # noqa: F401
        deps['torch'] = True
    except ImportError:
        deps['torch'] = False

    try:
        import onnxruntime  # noqa: F401
        deps['onnxruntime'] = True
    except ImportError:
        deps['onnxruntime'] = False

    return deps


class ONNXConverter:
    """
    Converts trained HAR models (sklearn or PyTorch) to ONNX format.

    Usage:
        converter = ONNXConverter(model_object, model_type, feature_names, classes)
        onnx_bytes = converter.convert()
        converter.save('model.onnx')
    """

    def __init__(self, model_object, model_type: str,
                 feature_names: List[str], classes: List[str],
                 model_params: Optional[Dict[str, Any]] = None):
        """
        Args:
            model_object: The trained EdgeMLModel object (has .model, .scaler attributes)
            model_type: One of 'random_forest', 'neural_network', 'svm', 'pytorch_mlp', 'pytorch_cnn'
            feature_names: List of feature names
            classes: List of class labels
            model_params: Optional model parameters (sampling_rate, window_size_ms, etc.)
        """
        self.model_object = model_object
        self.model_type = model_type
        self.feature_names = feature_names
        self.classes = classes
        self.model_params = model_params or {}
        self._onnx_model = None
        self._onnx_bytes = None

    def convert(self, opset_version: int = 13) -> bytes:
        """
        Convert the model to ONNX format.

        Args:
            opset_version: ONNX opset version (default: 13, good compatibility)

        Returns:
            ONNX model as bytes

        Raises:
            ImportError: If required packages not installed
            ValueError: If model type not supported
        """
        if self.model_type in ('pytorch_mlp', 'pytorch_cnn'):
            return self._convert_pytorch(opset_version)
        elif self.model_type in ('random_forest', 'neural_network', 'svm'):
            return self._convert_sklearn(opset_version)
        else:
            raise ValueError(f"Unsupported model type for ONNX conversion: {self.model_type}")

    def _convert_sklearn(self, opset_version: int) -> bytes:
        """Convert sklearn model (RF, SVM, MLP) to ONNX via skl2onnx."""
        try:
            import onnx
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError as e:
            raise ImportError(
                f"ONNX conversion for sklearn models requires 'onnx' and 'skl2onnx'. "
                f"Install with: pip install onnx skl2onnx\n"
                f"Missing: {e}"
            )

        sklearn_model = self.model_object.model
        n_features = len(self.feature_names)

        # Build the sklearn pipeline (scaler + model) if scaler exists
        from sklearn.pipeline import Pipeline
        if hasattr(self.model_object, 'scaler') and self.model_object.scaler is not None:
            pipeline = Pipeline([
                ('scaler', self.model_object.scaler),
                ('model', sklearn_model)
            ])
        else:
            pipeline = Pipeline([('model', sklearn_model)])

        # Define input type
        initial_type = [('float_input', FloatTensorType([None, n_features]))]

        # Convert to ONNX
        onnx_model = convert_sklearn(
            pipeline,
            initial_types=initial_type,
            target_opset=opset_version,
            options={id(sklearn_model): {'zipmap': False}}  # Get raw probabilities
        )

        # Add metadata
        onnx_model.doc_string = (
            f"HAR Model: {self.model_type} | "
            f"Features: {n_features} | "
            f"Classes: {len(self.classes)} ({', '.join(self.classes)})"
        )

        # Validate
        onnx.checker.check_model(onnx_model)

        self._onnx_model = onnx_model
        self._onnx_bytes = onnx_model.SerializeToString()

        logger.info(
            f"Converted sklearn {self.model_type} to ONNX: "
            f"{len(self._onnx_bytes)} bytes, opset {opset_version}"
        )
        return self._onnx_bytes

    def _convert_pytorch(self, opset_version: int) -> bytes:
        """Convert PyTorch model (MLP, CNN) to ONNX via torch.onnx.export."""
        try:
            import torch
            import onnx
            import io
        except ImportError as e:
            raise ImportError(
                f"ONNX conversion for PyTorch models requires 'torch' and 'onnx'. "
                f"Install with: pip install torch onnx\n"
                f"Missing: {e}"
            )

        # Get the actual PyTorch model
        pytorch_model = None
        if hasattr(self.model_object, 'trainer') and self.model_object.trainer is not None:
            pytorch_model = self.model_object.trainer.model
        elif hasattr(self.model_object, 'model') and isinstance(self.model_object.model, torch.nn.Module):
            pytorch_model = self.model_object.model

        if pytorch_model is None:
            raise ValueError("Could not find PyTorch model in model_object")

        pytorch_model.eval()
        device = next(pytorch_model.parameters()).device

        # Build dummy input based on model type
        n_features = len(self.feature_names)
        if self.model_type == 'pytorch_cnn':
            # CNN expects (batch, window_size, n_channels)
            window_size = self.model_params.get('window_size_samples', 150)
            n_channels = self.model_params.get('n_channels', 6)
            dummy_input = torch.randn(1, window_size, n_channels, device=device)
            input_names = ['sensor_window']
            dynamic_axes = {
                'sensor_window': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        else:
            # MLP expects (batch, n_features)
            dummy_input = torch.randn(1, n_features, device=device)
            input_names = ['features']
            dynamic_axes = {
                'features': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }

        # Export to ONNX
        buffer = io.BytesIO()
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            buffer,
            opset_version=opset_version,
            input_names=input_names,
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

        buffer.seek(0)
        self._onnx_bytes = buffer.read()

        # Load and validate
        self._onnx_model = onnx.load_from_string(self._onnx_bytes)
        onnx.checker.check_model(self._onnx_model)

        logger.info(
            f"Converted PyTorch {self.model_type} to ONNX: "
            f"{len(self._onnx_bytes)} bytes, opset {opset_version}"
        )
        return self._onnx_bytes

    def save(self, filepath: str) -> str:
        """Save the ONNX model to a file."""
        if self._onnx_bytes is None:
            self.convert()

        with open(filepath, 'wb') as f:
            f.write(self._onnx_bytes)

        logger.info(f"Saved ONNX model to {filepath}")
        return filepath

    def get_model_size(self) -> int:
        """Get the ONNX model size in bytes."""
        if self._onnx_bytes is None:
            self.convert()
        return len(self._onnx_bytes)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the converted ONNX model."""
        if self._onnx_model is None:
            self.convert()

        try:
            import onnx
            graph = self._onnx_model.graph

            # Count parameters
            total_params = 0
            for initializer in graph.initializer:
                params = 1
                for dim in initializer.dims:
                    params *= dim
                total_params += params

            return {
                'format': 'ONNX',
                'opset_version': self._onnx_model.opset_import[0].version,
                'size_bytes': len(self._onnx_bytes),
                'size_kb': len(self._onnx_bytes) / 1024,
                'num_nodes': len(graph.node),
                'num_inputs': len(graph.input),
                'num_outputs': len(graph.output),
                'total_parameters': total_params,
                'model_type': self.model_type,
                'num_features': len(self.feature_names),
                'num_classes': len(self.classes),
            }
        except Exception as e:
            return {
                'format': 'ONNX',
                'size_bytes': len(self._onnx_bytes),
                'error': str(e),
            }

    def validate_output(self, test_input: Optional[np.ndarray] = None,
                        expected_output: Optional[np.ndarray] = None,
                        rtol: float = 1e-3) -> Dict[str, Any]:
        """
        Validate ONNX model output against original model.

        Args:
            test_input: Test input array. If None, generates random input.
            expected_output: Expected output from original model.
            rtol: Relative tolerance for comparison.

        Returns:
            Validation report dict with 'passed', 'max_diff', 'mean_diff'.
        """
        if self._onnx_bytes is None:
            self.convert()

        try:
            import onnxruntime as ort
        except ImportError:
            return {
                'passed': None,
                'error': "onnxruntime not installed. pip install onnxruntime"
            }

        # Create inference session
        session = ort.InferenceSession(self._onnx_bytes)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape

        # Generate test input if needed
        if test_input is None:
            n_features = len(self.feature_names)
            if self.model_type == 'pytorch_cnn':
                window_size = self.model_params.get('window_size_samples', 150)
                n_channels = self.model_params.get('n_channels', 6)
                test_input = np.random.randn(1, window_size, n_channels).astype(np.float32)
            else:
                test_input = np.random.randn(1, n_features).astype(np.float32)

        # Run ONNX inference
        onnx_output = session.run(None, {input_name: test_input})

        result = {
            'passed': True,
            'onnx_output_shape': [o.shape for o in onnx_output],
            'input_shape': test_input.shape,
        }

        # Compare with expected output if available
        if expected_output is not None:
            onnx_probs = onnx_output[-1] if len(onnx_output) > 1 else onnx_output[0]
            max_diff = float(np.max(np.abs(onnx_probs - expected_output)))
            mean_diff = float(np.mean(np.abs(onnx_probs - expected_output)))
            result['max_diff'] = max_diff
            result['mean_diff'] = mean_diff
            result['passed'] = max_diff < rtol

        return result
