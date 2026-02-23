"""
SVM Code Generator
Generates Arduino C++ code specifically for SVM models
"""

from typing import Dict, Any
from .base_generator import BaseCodeGenerator


class SVMCodeGenerator(BaseCodeGenerator):
    """Code generator specifically for SVM models."""

    def __init__(self, model_data: Dict[str, Any], platform: str = 'arduino', optimization: str = 'balanced', overlap: float = 0.5):
        super().__init__(model_data, platform, optimization, overlap)
        self.support_vectors = model_data.get('support_vectors', [])
        self.num_support_vectors = len(self.support_vectors)

    def _get_model_specific_declarations(self) -> str:
        """Generate SVM specific declarations."""
        return f"""
// SVM specific definitions
#define NUM_SUPPORT_VECTORS {self.num_support_vectors if self.num_support_vectors > 0 else 1}

// SVM utility functions
float rbf_kernel(float* x1, float* x2, float gamma);
void print_svm_decision_scores(float features[]);
"""

    def _generate_model_specific_implementation(self) -> str:
        """Generate SVM implementation."""
        # Extract SVM parameters from model data
        support_vectors = self.model_data.get('support_vectors', [])
        dual_coef = self.model_data.get('dual_coefficients', [])
        intercepts = self.model_data.get('intercept', [])
        gamma_value = self.model_data.get('gamma', 0.1)

        num_classes = len(self.classes)
        num_sv = len(support_vectors) if support_vectors else 0

        # Format support vectors array
        sv_formatted = ""
        if support_vectors and num_sv > 0:
            for i, sv in enumerate(support_vectors):
                sv_formatted += "    {"
                sv_formatted += ", ".join([f"{val:.6f}f" for val in sv])
                sv_formatted += "}"
                if i < num_sv - 1:
                    sv_formatted += ",\n"
        else:
            # Fallback if no support vectors (should not happen in production)
            sv_formatted = "    {" + \
                ", ".join(["0.0f"] * len(self.feature_names)) + "}"

        # Format dual coefficients (organized by class for OvR)
        dual_coef_formatted = ""
        if dual_coef and len(dual_coef) > 0:
            for cls_idx, coefs in enumerate(dual_coef):
                dual_coef_formatted += "    {"
                dual_coef_formatted += ", ".join([f"{c:.6f}f" for c in coefs])
                dual_coef_formatted += "}"
                if cls_idx < len(dual_coef) - 1:
                    dual_coef_formatted += ",\n"
        else:
            # Fallback
            dual_coef_formatted = "    {" + \
                ", ".join(["0.0f"] * max(num_sv, 1)) + "}"

        # Format intercepts
        intercepts_formatted = ""
        if intercepts and len(intercepts) > 0:
            intercepts_formatted = ", ".join(
                [f"{ic:.6f}f" for ic in intercepts])
        else:
            intercepts_formatted = ", ".join(["0.0f"] * num_classes)

        return f"""// SVM Model Implementation
// One-vs-Rest (OvR) Multi-class SVM with RBF Kernel

// Support vectors (extracted from trained model)
// Shape: [NUM_SUPPORT_VECTORS][NUM_FEATURES]
const float support_vectors[NUM_SUPPORT_VECTORS][NUM_FEATURES] = {{
{sv_formatted}
}};

// Dual coefficients for each class (OvR strategy)
// Shape: [NUM_CLASSES][NUM_SUPPORT_VECTORS]
const float dual_coef[NUM_CLASSES][NUM_SUPPORT_VECTORS] = {{
{dual_coef_formatted}
}};

// Intercepts for each class
const float intercepts[NUM_CLASSES] = {{
    {intercepts_formatted}
}};

// RBF kernel gamma parameter
const float svm_gamma = {gamma_value:.6f}f;

// RBF Kernel function: K(x1, x2) = exp(-gamma * ||x1 - x2||^2)
float rbf_kernel(float* x1, float* x2, float gamma) {{
    float sum = 0.0f;
    for (int i = 0; i < NUM_FEATURES; i++) {{
        float diff = x1[i] - x2[i];
        sum += diff * diff;
    }}
    // Clamp exponent to prevent overflow/underflow on constrained devices
    float exponent = -gamma * sum;
    if (exponent < -80.0f) return 0.0f;   // exp(-80) ≈ 0
    if (exponent > 80.0f) exponent = 80.0f;
    return expf(exponent);
}}"""

    def _generate_prediction_function(self) -> str:
        """Generate SVM prediction function."""
        return """int har_predict_internal(float features[NUM_FEATURES]) {
    // NOTE: Features are already scaled by har_predict() wrapper function
    // Do NOT scale again here

    // SVM prediction using support vectors and One-vs-Rest strategy
    float decision_scores[NUM_CLASSES];
    
    // Initialize decision scores
    for (int i = 0; i < NUM_CLASSES; i++) {
        decision_scores[i] = 0.0f;
    }

    // Calculate decision function for each class using kernel trick
    // For each support vector, compute kernel and accumulate weighted sum
    for (int sv = 0; sv < NUM_SUPPORT_VECTORS; sv++) {
        float kernel_value = rbf_kernel(features, (float*)support_vectors[sv], svm_gamma);
        
        // Multi-class SVM: each support vector contributes to all class decisions
        // Using dual coefficients organized by class
        for (int cls = 0; cls < NUM_CLASSES; cls++) {
            decision_scores[cls] += dual_coef[cls][sv] * kernel_value;
        }
    }

    // Add intercepts for each class
    for (int i = 0; i < NUM_CLASSES; i++) {
        decision_scores[i] += intercepts[i];
    }

    // Return class with highest decision score (argmax)
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
        """Generate SVM utility functions (platform-portable)."""
        return """void print_svm_decision_scores(float features[]) {
    HAR_LOG("SVM Decision Scores:");

    // Calculate decision scores for each class
    float decision_scores[NUM_CLASSES];
    
    for (int cls = 0; cls < NUM_CLASSES; cls++) {
        decision_scores[cls] = 0.0f;
    }
    
    // Compute kernel values and accumulate weighted scores
    for (int sv = 0; sv < NUM_SUPPORT_VECTORS && sv < 10; sv++) {
        float kernel_value = rbf_kernel(features, (float*)support_vectors[sv], svm_gamma);
        HAR_LOG_FLOAT("SV kernel", kernel_value);
        
        for (int cls = 0; cls < NUM_CLASSES; cls++) {
            float contrib = dual_coef[cls][sv] * kernel_value;
            decision_scores[cls] += contrib;
        }
    }
    
    // Add intercepts
    for (int cls = 0; cls < NUM_CLASSES; cls++) {
        decision_scores[cls] += intercepts[cls];
    }
    
    // Print final scores
    HAR_LOG("Final decision scores:");
    for (int cls = 0; cls < NUM_CLASSES; cls++) {
        HAR_LOG_FLOAT(get_activity_name(cls), decision_scores[cls]);
    }
}"""
