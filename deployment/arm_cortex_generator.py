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


class ARMCortexMCodeGenerator(BaseCodeGenerator):
    """Code generator for ARM Cortex-M that delegates model logic to the
    appropriate model-specific generator and wraps it with ARM optimizations.
    """

    def __init__(self, model_data: Dict[str, Any], platform: str = 'arm_cortex_m',
                 optimization: str = 'balanced', overlap: float = 0.5):
        super().__init__(model_data, platform, optimization, overlap)
        self.optimization_level = optimization

        # Create the inner model-specific generator to delegate prediction to
        model_type = model_data.get('model_type', '')
        if model_type == 'random_forest':
            self._inner = RandomForestCodeGenerator(model_data, platform, optimization, overlap)
        elif model_type == 'neural_network':
            self._inner = NeuralNetworkCodeGenerator(model_data, platform, optimization, overlap)
        elif model_type == 'svm':
            self._inner = SVMCodeGenerator(model_data, platform, optimization, overlap)
        else:
            # Fallback: no inner generator – will use parent's abstract stubs
            self._inner = None
            print(f"⚠ ARM Cortex-M generator: unknown model type '{model_type}', "
                  f"model-specific prediction will not be generated.")

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
int har_predict_internal(float features[NUM_FEATURES]) {
    // Features are ALREADY SCALED by har_predict()
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
    Serial.println("ARM Cortex-M HAR System");
    Serial.print("Optimization: ");
    Serial.println("ARM_CORTEX_M");

#ifdef USE_ARM_DSP
    Serial.println("ARM DSP: Enabled");
#else
    Serial.println("ARM DSP: Disabled");
#endif

    Serial.print("Features: ");
    Serial.println(NUM_FEATURES);
    Serial.print("Classes: ");
    Serial.println(NUM_CLASSES);
}

// Performance monitoring
static uint32_t prediction_start_time = 0;
static uint32_t prediction_end_time = 0;

void start_prediction_timer() {
    prediction_start_time = micros();
}

void end_prediction_timer() {
    prediction_end_time = micros();
    Serial.print("Prediction time (us): ");
    Serial.println(prediction_end_time - prediction_start_time);
}
"""
        return utils

    def generate_header_file(self) -> str:
        """Generate ARM Cortex-M specific header file."""
        header = super().generate_header_file()

        # Add ARM-specific includes and definitions
        arm_specific = """
#ifdef __ARM_ARCH
    #include "arm_math.h"
    #define ARM_CORTEX_M_OPTIMIZED 1
#endif

// Performance monitoring functions
void start_prediction_timer();
void end_prediction_timer();
"""

        # Insert ARM-specific content before the closing endif
        header = header.replace('#endif', arm_specific + '\n#endif')

        return header
