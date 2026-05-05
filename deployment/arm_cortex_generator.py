"""
ARM Cortex-M Code Generator
Generates optimized C++ code for ARM Cortex-M microcontrollers.

This generator uses composition: it delegates the actual model prediction logic
to the appropriate model-specific generator (RF, NN, SVM) while adding
ARM Cortex-M specific optimizations (CMSIS-DSP, performance monitoring, etc.).
"""

from typing import Dict, Any
from .base_generator import BaseCodeGenerator
from .random_forest_generator import RandomForestCodeGenerator
from .neural_network_generator import NeuralNetworkCodeGenerator
from .svm_generator import SVMCodeGenerator
from .cnn_generator import CNNCodeGenerator


class ARMCortexMCodeGenerator(BaseCodeGenerator):
    """Code generator for ARM Cortex-M that delegates model logic to the
    appropriate model-specific generator and wraps it with ARM optimizations.
    """

    def __init__(self, model_data: Dict[str, Any], platform: str = 'arm_cortex_m',
                 optimization: str = 'balanced', overlap: float = 0.5,
                 quantization: str = 'none',
                 confidence_threshold: float = 0.6,
                 smoothing_window: int = 1,
                 enable_iir_filter: bool = False,
                 enable_kalman_filter: bool = False):
        # CNN models don't use traditional features — provide placeholders
        model_type = model_data.get('model_type', '')
        if model_type == 'pytorch_cnn' and not model_data.get('feature_names'):
            model_data = dict(model_data)
            n_ch = model_data.get('n_channels', 6)
            model_data['feature_names'] = [f'ch{i}' for i in range(n_ch)]

        super().__init__(model_data, platform, optimization, overlap, quantization,
                         confidence_threshold=confidence_threshold,
                         smoothing_window=smoothing_window,
                         enable_iir_filter=enable_iir_filter,
                         enable_kalman_filter=enable_kalman_filter)
        self.optimization_level = optimization

        # Create the inner model-specific generator to delegate prediction to
        model_type = model_data.get('model_type', '')
        if model_type == 'random_forest':
            self._inner = RandomForestCodeGenerator(model_data, platform, optimization, overlap, quantization)
        elif model_type in ('neural_network', 'pytorch_mlp'):
            self._inner = NeuralNetworkCodeGenerator(model_data, platform, optimization, overlap, quantization)
        elif model_type == 'svm':
            self._inner = SVMCodeGenerator(model_data, platform, optimization, overlap, quantization)
        elif model_type == 'pytorch_cnn':
            self._inner = CNNCodeGenerator(model_data, platform, optimization, overlap, quantization)
        else:
            # Fallback: no inner generator – will use parent's abstract stubs
            self._inner = None
            print(f"⚠ ARM Cortex-M generator: unknown model type '{model_type}', "
                  f"model-specific prediction will not be generated.")

    # ------------------------------------------------------------------ #
    #  For CNN models, bypass the composition pattern and delegate fully  #
    # ------------------------------------------------------------------ #

    def generate_header(self) -> str:
        """For CNN, use the inner CNN generator's header directly."""
        if isinstance(self._inner, CNNCodeGenerator):
            return self._inner.generate_header()
        return super().generate_header()

    def generate_implementation(self, header_filename: str = None) -> str:
        """For CNN, use the inner CNN generator's implementation directly."""
        if isinstance(self._inner, CNNCodeGenerator):
            return self._inner.generate_implementation(header_filename)
        return super().generate_implementation(header_filename)

    def generate_example_sketch(self, header_filename: str = None) -> str:
        """For CNN, use the inner CNN generator's sketch directly."""
        if isinstance(self._inner, CNNCodeGenerator):
            return self._inner.generate_example_sketch(header_filename)
        return super().generate_example_sketch(header_filename)

    # ------------------------------------------------------------------ #
    #  Model-specific sections – delegated to inner generator              #
    # ------------------------------------------------------------------ #

    def _get_model_specific_declarations(self) -> str:
        """Generate declarations: ARM-specific extras + inner model declarations."""
        arm_decls = """
// ARM Cortex-M optimizations
#ifdef __ARM_ARCH
    #include "arm_math.h"
    #define USE_ARM_DSP 1
#endif

// ARM Cortex-M utility functions
void print_system_info();
void start_prediction_timer();
void end_prediction_timer();
"""
        if self._inner:
            arm_decls += self._inner._get_model_specific_declarations()
        return arm_decls

    def _generate_model_specific_implementation(self) -> str:
        """Generate model weights/parameters from inner generator + ARM helpers."""
        impl = ""

        # Delegate model data (weights, trees, support vectors) to inner generator
        if self._inner:
            impl += self._inner._generate_model_specific_implementation()

        # Add ARM-specific helper implementations
        impl += """

// ---- ARM Cortex-M Optimized Helpers ----

#ifdef USE_ARM_DSP
// ARM CMSIS-DSP accelerated feature scaling
void arm_optimized_feature_scaling(float* features, float* scaled_features) {
    arm_sub_f32(features, (float*)feature_means, scaled_features, NUM_FEATURES);
    // Element-wise division (CMSIS-DSP does not have arm_div_f32 for vectors,
    // so we fall back to a loop with reciprocal multiplication)
    for (int i = 0; i < NUM_FEATURES; i++) {
        scaled_features[i] /= feature_stds[i];
    }
}
#endif
"""
        return impl

    def _generate_prediction_function(self) -> str:
        """Delegate prediction function to the inner model-specific generator.

        CRITICAL: har_predict_internal() receives ALREADY SCALED features from
        the har_predict() wrapper in base_generator.  Do NOT scale again here.
        """
        if self._inner:
            return self._inner._generate_prediction_function()

        # Safety fallback – should never happen if model_type is valid
        return """// ERROR: No model-specific prediction logic available.
// The ARM Cortex-M generator could not determine the model type.
int har_predict_internal(float features[NUM_FEATURES], float probs_out[NUM_CLASSES]) {
    // Features are ALREADY SCALED by har_predict()
    for (int i = 0; i < NUM_CLASSES; i++) probs_out[i] = 0.0f;
    return 0;  // Fallback – always predicts class 0
}"""

    def _generate_utility_functions(self) -> str:
        """Generate ARM Cortex-M specific utility functions."""
        utils = ""

        # Include inner model's utilities (e.g., print_network_outputs)
        if self._inner:
            utils += self._inner._generate_utility_functions()

        utils += """

// ---- ARM Cortex-M Platform Utilities ----

void print_system_info() {
    HAR_LOG("ARM Cortex-M HAR System");

#ifdef USE_ARM_DSP
    HAR_LOG("ARM DSP: Enabled");
#else
    HAR_LOG("ARM DSP: Disabled");
#endif

    HAR_LOG_FLOAT("Features", (float)NUM_FEATURES);
    HAR_LOG_FLOAT("Classes", (float)NUM_CLASSES);
}

// Performance monitoring
static uint32_t prediction_start_time = 0;
static uint32_t prediction_end_time = 0;

void start_prediction_timer() {
    prediction_start_time = micros();
}

void end_prediction_timer() {
    prediction_end_time = micros();
    HAR_LOG_FLOAT("Prediction time (us)", (float)(prediction_end_time - prediction_start_time));
}
"""
        return utils
