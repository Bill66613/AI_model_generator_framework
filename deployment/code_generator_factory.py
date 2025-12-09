"""
Code Generator Factory
Factory pattern implementation for creating appropriate code generators
"""

import os
from datetime import datetime
from typing import Dict, Any, Type, List
from .base_generator import BaseCodeGenerator, ValidationError, ModelDataError, OptimizationError
from .random_forest_generator import RandomForestCodeGenerator
from .neural_network_generator import NeuralNetworkCodeGenerator
from .svm_generator import SVMCodeGenerator
from .arm_cortex_generator import ARMCortexMCodeGenerator


def extract_real_model_parameters(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract actual parameters from trained model object."""
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
            elif model_obj.model_type == 'svm':
                # Extract and merge SVM parameters
                svm_params = extract_svm_parameters(model_obj.model)
                enhanced_data.update(svm_params)

            # Extract label encoding
            if hasattr(model_obj, 'label_encoder') and model_obj.label_encoder:
                enhanced_data['label_mapping'] = {
                    i: label for i, label in enumerate(model_obj.label_encoder.classes_)
                }

        except Exception as e:
            print(f"Warning: Could not extract real model parameters: {e}")

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
    """Extract weights and biases from neural network model."""
    weights = {}
    try:
        if hasattr(nn_model, 'coefs_') and hasattr(nn_model, 'intercepts_'):
            weights['input_weights'] = nn_model.coefs_[0].tolist()
            weights['hidden_biases'] = nn_model.intercepts_[0].tolist()
            if len(nn_model.coefs_) > 1:
                weights['output_weights'] = nn_model.coefs_[1].tolist()
                weights['output_biases'] = nn_model.intercepts_[1].tolist()
            weights['hidden_size'] = len(nn_model.intercepts_[0])
    except Exception as e:
        print(f"Could not extract neural network weights: {e}")
    return weights


def extract_svm_parameters(svm_model) -> Dict[str, Any]:
    """Extract support vectors and parameters from SVM model."""
    params = {}
    try:
        if hasattr(svm_model, 'support_vectors_'):
            params['support_vectors'] = svm_model.support_vectors_.tolist()
        if hasattr(svm_model, 'dual_coef_'):
            params['dual_coefficients'] = svm_model.dual_coef_.tolist()
        if hasattr(svm_model, 'intercept_'):
            params['intercept'] = svm_model.intercept_.tolist()

        # Extract actual gamma value (not 'scale' or 'auto' string)
        if hasattr(svm_model, '_gamma'):
            # This is the actual computed gamma value used by the model
            params['gamma'] = float(svm_model._gamma)
        elif hasattr(svm_model, 'gamma'):
            gamma_val = svm_model.gamma
            # If gamma is a string ('scale' or 'auto'), we need the computed value
            if isinstance(gamma_val, str):
                # Fallback: use default scale formula if available
                if hasattr(svm_model, 'support_vectors_'):
                    n_features = svm_model.support_vectors_.shape[1]
                    if gamma_val == 'scale':
                        # gamma = 1 / (n_features * X.var())
                        # Use reasonable default since we don't have X.var()
                        params['gamma'] = 1.0 / n_features
                    else:  # 'auto'
                        # gamma = 1 / n_features
                        params['gamma'] = 1.0 / n_features
                else:
                    params['gamma'] = 0.1  # Reasonable default
            else:
                params['gamma'] = float(gamma_val)
        else:
            params['gamma'] = 0.1  # Default fallback
    except Exception as e:
        print(f"Could not extract SVM parameters: {e}")
    return params


def create_organized_filename(model_type: str, platform: str, file_type: str,
                              model_data: Dict[str, Any] = None, optimization: str = 'balanced') -> str:
    """
    Create organized filename with descriptive naming including optimization level.

    Args:
        model_type: Type of model (random_forest, neural_network, svm)
        platform: Target platform (arduino, arm_cortex_m, etc.)
        file_type: Type of file (header, source, sketch)
        model_data: Optional model data for additional info
        optimization: Optimization level (accuracy, speed, power, balanced)

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

    # Add file extension based on type
    if file_type == 'header':
        return f"{base_name}.h"
    elif file_type == 'source':
        return f"{base_name}.cpp"
    elif file_type == 'cortex_source':
        return f"{base_name}.c"
    elif file_type == 'sketch':
        # Arduino .ino file must match folder name - no _example suffix
        return f"{base_name}.ino"
    else:
        return f"{base_name}.{file_type}"


def create_output_folder_structure(base_output_dir: str, model_type: str,
                                   platform: str, model_data: Dict[str, Any] = None,
                                   optimization: str = 'balanced') -> str:
    """
    Create organized folder structure for generated code.
    For Arduino-based platforms, creates folder matching the .ino filename.

    Args:
        base_output_dir: Base directory for all generated code
        model_type: Type of model
        platform: Target platform
        model_data: Model data containing features and classes info
        optimization: Optimization strategy

    Returns:
        Full path to the specific model/platform folder
    """
    # For Arduino-based platforms, folder must match .ino filename
    if platform in ['arduino', 'seeed_xiao', 'esp32', 'teensy']:
        # Create the same base name as the .ino file (without _example.ino)
        num_features = len(model_data.get(
            'feature_names', [])) if model_data else 0
        num_classes = len(model_data.get('classes', [])) if model_data else 0

        folder_name = f"har_{model_type}_{platform}_f{num_features}_c{num_classes}_{optimization}"
        folder_path = os.path.join(
            base_output_dir, f"{model_type}_models", folder_name)
    else:
        # For non-Arduino platforms, use generic platform folder
        folder_path = os.path.join(
            base_output_dir, f"{model_type}_models", platform)

    # Create directories if they don't exist
    os.makedirs(folder_path, exist_ok=True)

    return folder_path


class CodeGeneratorFactory:
    """Factory for creating code generators based on model type and platform."""

    # Registry of available generators
    _generators: Dict[str, Type[BaseCodeGenerator]] = {
        'random_forest': RandomForestCodeGenerator,
        'neural_network': NeuralNetworkCodeGenerator,
        'svm': SVMCodeGenerator,
        'arm_cortex_m': ARMCortexMCodeGenerator,
    }

    @classmethod
    def create_generator(cls, model_type: str, model_data: Dict[str, Any],
                         platform: str = 'arduino', optimization: str = 'balanced') -> BaseCodeGenerator:
        """
        Create appropriate code generator based on model type and platform.

        Args:
            model_type: Type of model ('random_forest', 'neural_network', 'svm')
            model_data: Dictionary containing model parameters and data
            platform: Target platform ('arduino', 'arm_cortex_m', etc.)
            optimization: Optimization strategy ('accuracy', 'speed', 'power', 'balanced')

        Returns:
            Appropriate code generator instance

        Raises:
            ValueError: If model_type is not supported
            ValidationError: If input parameters are invalid
            ModelDataError: If model_data is invalid
        """
        try:
            # Validate model_type first
            if not isinstance(model_type, str):
                raise ValueError("model_type must be a string")

            # For ARM Cortex-M platform, use specialized generator
            if platform == 'arm_cortex_m':
                return cls._generators['arm_cortex_m'](model_data, platform, optimization)

            # For other platforms, use model-specific generators
            if model_type not in cls._generators:
                available_types = [
                    t for t in cls._generators.keys() if t != 'arm_cortex_m']
                raise ValueError(f"Unsupported model type: '{model_type}'. "
                                 f"Supported types: {available_types}")

            generator_class = cls._generators[model_type]
            return generator_class(model_data, platform, optimization)

        except (ValidationError, ModelDataError, OptimizationError) as e:
            # Re-raise validation errors with context
            raise type(e)(f"Generator creation failed: {str(e)}") from e
        except Exception as e:
            # Wrap unexpected errors
            raise RuntimeError(
                f"Unexpected error creating generator for {model_type}: {str(e)}") from e

    @classmethod
    def get_supported_models(cls) -> list:
        """Get list of supported model types."""
        return [model for model in cls._generators.keys() if model != 'arm_cortex_m']

    @classmethod
    def get_supported_platforms(cls) -> list:
        """Get list of supported platforms."""
        return ['arduino', 'arm_cortex_m', 'esp32', 'teensy']

    @classmethod
    def register_generator(cls, model_type: str, generator_class: Type[BaseCodeGenerator]):
        """
        Register a new generator class.

        Args:
            model_type: String identifier for the model type
            generator_class: Class that inherits from BaseCodeGenerator
        """
        if not issubclass(generator_class, BaseCodeGenerator):
            raise ValueError(
                "Generator class must inherit from BaseCodeGenerator")

        cls._generators[model_type] = generator_class


def generate_deployment_code(model_type: str, model_data: Dict[str, Any],
                             platform: str = 'arduino', optimization: str = 'balanced') -> Dict[str, str]:
    """
    Convenience function to generate deployment code with organized naming.

    Args:
        model_type: Type of model to generate code for
        model_data: Model parameters and data
        platform: Target platform
        optimization: Optimization strategy ('accuracy', 'speed', 'power', 'balanced')

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

        generator = CodeGeneratorFactory.create_generator(
            model_type, model_data, platform, optimization)

        # Create organized filenames
        if platform == 'arm_cortex_m':
            cortex_filename = create_organized_filename(
                model_type, platform, 'cortex_source', model_data, optimization)
            return {
                cortex_filename: generator.generate_implementation(
                    cortex_filename)
            }
        else:
            # Arduino-based platforms
            header_filename = create_organized_filename(
                model_type, platform, 'header', model_data, optimization)
            source_filename = create_organized_filename(
                model_type, platform, 'source', model_data, optimization)
            sketch_filename = create_organized_filename(
                model_type, platform, 'sketch', model_data, optimization)

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
                                      optimization: str = 'balanced') -> Dict[str, str]:
    """
    Generate deployment code and save to organized folder structure.

    Args:
        model_type: Type of model to generate code for
        model_data: Model parameters and data
        platform: Target platform
        output_dir: Base output directory for generated files
        optimization: Optimization strategy ('accuracy', 'speed', 'power', 'balanced')

    Returns:
        Dictionary with full file paths as keys and success messages as values
    """
    # Extract additional model information if available
    if 'model_object' in model_data:
        # If we have the actual model object, extract real parameters
        model_data = extract_real_model_parameters(model_data)

    # Create organized folder structure
    folder_path = create_output_folder_structure(
        output_dir, model_type, platform, model_data, optimization)

    # Generate code with organized naming and optimization
    generated_code = generate_deployment_code(
        model_type, model_data, platform, optimization)

    # Save files and return file paths
    saved_files = {}

    for filename, content in generated_code.items():
        full_path = os.path.join(folder_path, filename)

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
