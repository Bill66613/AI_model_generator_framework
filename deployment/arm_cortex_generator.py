"""
ARM Cortex-M Code Generator
Generates optimized C++ code specifically for ARM Cortex-M microcontrollers
"""

from typing import Dict, Any
from .base_generator import BaseCodeGenerator


class ARMCortexMCodeGenerator(BaseCodeGenerator):
    """Code generator specifically optimized for ARM Cortex-M microcontrollers."""

    def __init__(self, model_data: Dict[str, Any], platform: str = 'arm_cortex_m', optimization: str = 'balanced', overlap: float = 0.5):
        super().__init__(model_data, platform, optimization, overlap)
        self.optimization_level = optimization  # Use the optimization parameter

    def _get_model_specific_declarations(self) -> str:
        """Generate ARM Cortex-M specific declarations."""
        return """
// ARM Cortex-M optimizations
#ifdef __ARM_ARCH
    #include "arm_math.h"
    #define USE_ARM_DSP 1
#endif

// Memory-optimized data structures
#define FIXED_POINT_PRECISION 16
#define SCALE_FACTOR (1 << FIXED_POINT_PRECISION)

// ARM Cortex-M specific optimizations
typedef int32_t fixed_point_t;

// ARM Cortex-M utility functions
fixed_point_t float_to_fixed(float f);
float fixed_to_float(fixed_point_t f);
void print_system_info();
"""

    def _generate_model_specific_implementation(self) -> str:
        """Generate ARM Cortex-M optimized implementation."""
        if self.optimization_level == 'memory':
            return self._generate_memory_optimized_implementation()
        else:
            return self._generate_speed_optimized_implementation()

    def _generate_memory_optimized_implementation(self) -> str:
        """Generate memory-optimized implementation for resource-constrained devices."""
        return """// Memory-optimized ARM Cortex-M implementation

// Fixed-point arithmetic for memory efficiency
fixed_point_t float_to_fixed(float f) {
    return (fixed_point_t)(f * SCALE_FACTOR);
}

float fixed_to_float(fixed_point_t f) {
    return (float)f / SCALE_FACTOR;
}

// Compressed feature scaling arrays (using fixed-point)
const fixed_point_t feature_means_fixed[NUM_FEATURES] = {
    // Converted feature means as fixed-point values
};

const fixed_point_t feature_stds_fixed[NUM_FEATURES] = {
    // Converted feature stds as fixed-point values
};"""

    def _generate_speed_optimized_implementation(self) -> str:
        """Generate speed-optimized implementation using ARM DSP instructions."""
        return """// Speed-optimized ARM Cortex-M implementation

#ifdef USE_ARM_DSP
// Use ARM CMSIS-DSP library for optimized operations
void arm_optimized_feature_scaling(float* features, float* scaled_features) {
    // Use ARM DSP functions for vector operations
    arm_sub_f32(features, feature_means, scaled_features, NUM_FEATURES);
    arm_div_f32(scaled_features, feature_stds, scaled_features, NUM_FEATURES);
}

void arm_optimized_matrix_mult(const float* matrix, const float* vector, 
                              float* result, uint16_t rows, uint16_t cols) {
    // Use ARM DSP matrix multiplication
    arm_matrix_instance_f32 mat_inst;
    arm_matrix_instance_f32 vec_inst;
    arm_matrix_instance_f32 res_inst;
    
    arm_mat_init_f32(&mat_inst, rows, cols, (float*)matrix);
    arm_mat_init_f32(&vec_inst, cols, 1, (float*)vector);
    arm_mat_init_f32(&res_inst, rows, 1, result);
    
    arm_mat_mult_f32(&mat_inst, &vec_inst, &res_inst);
}
#endif"""

    def _generate_prediction_function(self) -> str:
        """Generate ARM Cortex-M optimized prediction function."""
        return """int har_predict_internal(float features[NUM_FEATURES]) {
    float scaled_features[NUM_FEATURES];
    
#ifdef USE_ARM_DSP
    // Use ARM DSP optimized scaling
    arm_optimized_feature_scaling(features, scaled_features);
#else
    // Fallback to standard scaling
    for (int i = 0; i < NUM_FEATURES; i++) {
        scaled_features[i] = (features[i] - feature_means[i]) / feature_stds[i];
    }
#endif
    
    // Model-specific prediction logic would go here
    // This is a placeholder that would be replaced with actual model logic
    
    // Example: Simple classification based on feature thresholds
    int predicted_class = 0;
    float class_scores[NUM_CLASSES] = {0};
    
    // Calculate simple decision boundaries (placeholder logic)
    for (int i = 0; i < NUM_FEATURES && i < 10; i++) {
        if (scaled_features[i] > 0.5) class_scores[0] += 1;
        else if (scaled_features[i] < -0.5) class_scores[1] += 1;
        else class_scores[2] += 1;
    }
    
    // Find class with highest score
    float max_score = class_scores[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (class_scores[i] > max_score) {
            max_score = class_scores[i];
            predicted_class = i;
        }
    }
    
    return predicted_class;
}"""

    def _generate_utility_functions(self) -> str:
        """Generate ARM Cortex-M specific utility functions."""
        return """// ARM Cortex-M specific utilities

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
uint32_t prediction_start_time;
uint32_t prediction_end_time;

void start_prediction_timer() {
    prediction_start_time = micros();
}

void end_prediction_timer() {
    prediction_end_time = micros();
    Serial.print("Prediction time (μs): ");
    Serial.println(prediction_end_time - prediction_start_time);
}

// Memory usage monitoring
void print_memory_usage() {
    Serial.print("Free RAM: ");
    Serial.println(freeMemory());
}"""

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
void print_memory_usage();
"""

        # Insert ARM-specific content before the closing endif
        header = header.replace('#endif', arm_specific + '\n#endif')

        return header
