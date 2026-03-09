"""
Model conversion utilities for alternative deployment approaches.

Provides converters from sklearn/PyTorch models to:
- ONNX format (for ONNX Runtime deployment)
- TFLite format (for TensorFlow Lite Micro deployment)

These converters require optional dependencies:
- ONNX export: skl2onnx, onnx, torch (for PyTorch models)
- TFLite export: tensorflow or onnx2tf
"""

from .onnx_converter import ONNXConverter, check_onnx_dependencies
from .tflite_converter import TFLiteConverter, check_tflite_dependencies

__all__ = [
    'ONNXConverter',
    'TFLiteConverter',
    'check_onnx_dependencies',
    'check_tflite_dependencies',
]
