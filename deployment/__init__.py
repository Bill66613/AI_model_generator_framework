"""
Deployment package — v2 clean architecture for HAR embedded code generation.

All generation is handled by deployment/v2/.  The factory functions below
provide the public API used by the application.
"""

from .code_generator_factory import (
    CodeGeneratorFactory,
    generate_deployment_code,
    analyze_resource_requirements,
    generate_and_save_deployment_code,
    get_deployment_info,
    DEPLOYMENT_APPROACHES,
    BaseCodeGenerator,        # backward-compat stub
    ValidationError,
    ModelDataError,
    OptimizationError,
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
    'TFLiteMicroCodeGenerator',
    'ONNXRuntimeCodeGenerator',
    'CodeGeneratorFactory',
    'generate_deployment_code',
    'analyze_resource_requirements',
    'generate_and_save_deployment_code',
    'get_deployment_info',
    'DEPLOYMENT_APPROACHES',
    'DeploymentValidator',
    'validate_before_deployment',
    'quantize_tensor',
    'QUANTIZATION_MODES',
    'QuantizedTensor',
    'QuantizationReport',
]
