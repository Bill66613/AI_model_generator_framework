"""
Deployment package initialization
Provides clean imports for the refactored code generators
"""

from .base_generator import BaseCodeGenerator
from .random_forest_generator import RandomForestCodeGenerator
from .neural_network_generator import NeuralNetworkCodeGenerator
from .svm_generator import SVMCodeGenerator
from .cnn_generator import CNNCodeGenerator
from .arm_cortex_generator import ARMCortexMCodeGenerator
from .micropython_generator import MicroPythonCodeGenerator
from .zephyr_generator import ZephyrCodeGenerator
from .code_generator_factory import (
    CodeGeneratorFactory, 
    generate_deployment_code, 
    analyze_resource_requirements,
    generate_and_save_deployment_code,
    get_deployment_info
)
from .validation import DeploymentValidator, validate_before_deployment
from .quantization import (
    quantize_tensor, quantize_symmetric_int8, quantize_symmetric_int16,
    quantize_float16, compute_quantization_error, QUANTIZATION_MODES,
    QuantizedTensor, QuantizationReport
)

__all__ = [
    'BaseCodeGenerator',
    'RandomForestCodeGenerator',
    'NeuralNetworkCodeGenerator',
    'SVMCodeGenerator',
    'CNNCodeGenerator',
    'ARMCortexMCodeGenerator',
    'MicroPythonCodeGenerator',
    'ZephyrCodeGenerator',
    'CodeGeneratorFactory',
    'generate_deployment_code',
    'analyze_resource_requirements',
    'generate_and_save_deployment_code',
    'get_deployment_info',
    'DeploymentValidator',
    'validate_before_deployment',
    'quantize_tensor',
    'QUANTIZATION_MODES',
    'QuantizedTensor',
    'QuantizationReport',
]
