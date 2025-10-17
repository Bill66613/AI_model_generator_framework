"""
SVM Code Generator
Generates Arduino C++ code specifically for SVM models
"""

from typing import Dict, Any
from .base_generator import BaseCodeGenerator


class SVMCodeGenerator(BaseCodeGenerator):
    """Code generator specifically for SVM models."""

    def __init__(self, model_data: Dict[str, Any], platform: str = 'arduino', optimization: str = 'balanced'):
        super().__init__(model_data, platform, optimization)
        self.support_vectors = model_data.get('support_vectors', [])
        self.num_support_vectors = len(self.support_vectors)

    def _get_model_specific_declarations(self) -> str:
        """Generate SVM specific declarations."""
        return f"""
// SVM specific definitions
#define NUM_SUPPORT_VECTORS {self.num_support_vectors if self.num_support_vectors > 0 else 100}

// Kernel functions
float rbf_kernel(float* x1, float* x2, float gamma);
"""

    def _generate_model_specific_implementation(self) -> str:
        """Generate SVM implementation."""
        return f"""// SVM Model Implementation

// Support vectors (extracted from trained model)
const float support_vectors[NUM_SUPPORT_VECTORS][NUM_FEATURES] = {{
    // Placeholder - would contain actual support vectors
}};

const float support_vector_coeffs[NUM_SUPPORT_VECTORS] = {{
    // Placeholder - would contain actual coefficients
}};

const float svm_intercept = 0.0; // Placeholder - would contain actual intercept
const float svm_gamma = 0.1;     // Placeholder - would contain actual gamma

// RBF Kernel function
float rbf_kernel(float* x1, float* x2, float gamma) {{
    float sum = 0;
    for (int i = 0; i < NUM_FEATURES; i++) {{
        float diff = x1[i] - x2[i];
        sum += diff * diff;
    }}
    return exp(-gamma * sum);
}}"""

    def _generate_prediction_function(self) -> str:
        """Generate SVM prediction function."""
        return """int har_predict(float features[NUM_FEATURES]) {
    // Scale features using training parameters
    float scaled_features[NUM_FEATURES];
    for (int i = 0; i < NUM_FEATURES; i++) {
        scaled_features[i] = (features[i] - feature_means[i]) / feature_stds[i];
    }
    
    // SVM prediction using support vectors
    float decision_scores[NUM_CLASSES] = {0};
    
    // Calculate decision function for each class
    for (int sv = 0; sv < NUM_SUPPORT_VECTORS; sv++) {
        float kernel_value = rbf_kernel(scaled_features, (float*)support_vectors[sv], svm_gamma);
        
        // For multi-class SVM, distribute vote based on coefficient
        int target_class = sv % NUM_CLASSES; // Simplified class assignment
        decision_scores[target_class] += support_vector_coeffs[sv] * kernel_value;
    }
    
    // Add intercept
    for (int i = 0; i < NUM_CLASSES; i++) {
        decision_scores[i] += svm_intercept;
    }
    
    // Return class with highest decision score
    int predicted_class = 0;
    float max_score = decision_scores[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (decision_scores[i] > max_score) {
            max_score = decision_scores[i];
            predicted_class = i;
        }
    }
    
    return predicted_class;
}"""

    def _generate_utility_functions(self) -> str:
        """Generate SVM utility functions."""
        return """void print_svm_decision_scores(float features[]) {
    Serial.println("SVM Decision Scores:");
    
    // Calculate and print decision scores for each class
    float decision_scores[NUM_CLASSES] = {0};
    
    for (int sv = 0; sv < NUM_SUPPORT_VECTORS && sv < 10; sv++) {
        float kernel_value = rbf_kernel(features, (float*)support_vectors[sv], svm_gamma);
        Serial.print("SV");
        Serial.print(sv);
        Serial.print(" kernel: ");
        Serial.println(kernel_value, 4);
    }
}"""
