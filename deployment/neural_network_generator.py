"""
Neural Network Code Generator
Generates Arduino C++ code specifically for Neural Network models
"""

from typing import Dict, Any
from .base_generator import BaseCodeGenerator


class NeuralNetworkCodeGenerator(BaseCodeGenerator):
    """Code generator specifically for Neural Network models."""

    def __init__(self, model_data: Dict[str, Any], platform: str = 'arduino', optimization: str = 'balanced'):
        super().__init__(model_data, platform, optimization)
        self.weights = model_data.get('weights', {})
        self.hidden_size = self.weights.get('hidden_size', 50)

        # Extract real model weights if available
        self._extract_real_weights()

    def _extract_real_weights(self):
        """Extract actual weights from trained model object or direct weights."""
        # First try to extract from model object (scikit-learn MLPClassifier)
        if 'model_object' in self.model_data:
            model_obj = self.model_data['model_object']
            try:
                if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'coefs_'):
                    # Extract weights from scikit-learn MLPClassifier
                    coefs = model_obj.model.coefs_
                    intercepts = model_obj.model.intercepts_

                    if len(coefs) >= 1:
                        # Input to hidden weights
                        self.input_weights = coefs[0].tolist()
                        # Hidden layer biases
                        self.hidden_biases = intercepts[0].tolist()
                        self.hidden_size = len(intercepts[0])

                    if len(coefs) >= 2:
                        # Hidden to output weights
                        self.output_weights = coefs[1].tolist()
                        # Output layer biases
                        self.output_biases = intercepts[1].tolist()
                    else:
                        # Single layer network
                        self.output_weights = []
                        self.output_biases = []

                    print(
                        f"Extracted NN weights from model object: {len(self.input_weights)}x{len(self.input_weights[0]) if self.input_weights else 0} input weights")
                    return

            except Exception as e:
                print(
                    f"Warning: Could not extract neural network weights from model object: {e}")

        # Try to extract from direct weights parameter
        if self.weights:
            try:
                self.input_weights = self.weights.get('input_weights', [])
                # Note: input_bias is hidden layer bias
                self.hidden_biases = self.weights.get('input_bias', [])
                # Note: hidden_weights is output weights
                self.output_weights = self.weights.get('hidden_weights', [])
                self.output_biases = self.weights.get('output_bias', [])

                if self.input_weights and self.hidden_biases:
                    print(
                        f"Extracted NN weights from direct parameters: {len(self.input_weights)}x{len(self.input_weights[0]) if self.input_weights else 0} input weights")
                    return

            except Exception as e:
                print(
                    f"Warning: Could not extract neural network weights from direct parameters: {e}")

        # Fallback to placeholder weights
        print("Using placeholder weights (no real weights available)")
        self.input_weights = []
        self.hidden_biases = []
        self.output_weights = []
        self.output_biases = []

    def _get_model_specific_declarations(self) -> str:
        """Generate Neural Network specific declarations."""
        return f"""
// Neural Network specific definitions
#define HIDDEN_LAYER_SIZE {self.hidden_size}
#define INPUT_SIZE NUM_FEATURES
#define OUTPUT_SIZE NUM_CLASSES

// Neural Network utility functions
float sigmoid(float x);
float relu(float x);
void print_network_outputs(float features[]);
"""

    def _generate_model_specific_implementation(self) -> str:
        """Generate Neural Network implementation with real or placeholder weights."""

        # Generate input weights array
        if hasattr(self, 'input_weights') and self.input_weights:
            input_weights_str = self._format_2d_array(
                self.input_weights, "input_weights")
        else:
            input_weights_str = f"""// Placeholder input weights - {len(self.feature_names)} x {self.hidden_size}
const float input_weights[INPUT_SIZE][HIDDEN_LAYER_SIZE] = {{
    // Would contain actual weights from trained model
    // Initialize with small random values for compilation
}};"""

        # Generate hidden biases array
        if hasattr(self, 'hidden_biases') and self.hidden_biases:
            hidden_biases_str = self._format_1d_array(
                self.hidden_biases, "hidden_biases", self.feature_precision)
        else:
            hidden_biases_str = f"""// Placeholder hidden biases
const float hidden_biases[HIDDEN_LAYER_SIZE] = {{
    // Would contain actual biases from trained model
}};"""

        # Generate output weights array
        if hasattr(self, 'output_weights') and self.output_weights:
            output_weights_str = self._format_2d_array(
                self.output_weights, "output_weights")
        else:
            output_weights_str = f"""// Placeholder output weights
const float output_weights[HIDDEN_LAYER_SIZE][OUTPUT_SIZE] = {{
    // Would contain actual output weights from trained model
}};"""

        # Generate output biases array
        if hasattr(self, 'output_biases') and self.output_biases:
            output_biases_str = self._format_1d_array(
                self.output_biases, "output_biases", self.feature_precision)
        else:
            output_biases_str = f"""// Placeholder output biases
const float output_biases[OUTPUT_SIZE] = {{
    // Would contain actual output biases from trained model
}};"""

        has_real_weights = hasattr(
            self, 'input_weights') and self.input_weights
        return f"""// Neural Network Model Implementation
// Using real weights: {has_real_weights}

{input_weights_str}

{hidden_biases_str}

{output_weights_str}

{output_biases_str}

// Activation functions
float sigmoid(float x) {{
    return 1.0 / (1.0 + exp(-x));
}}

float relu(float x) {{
    return x > 0 ? x : 0;
}}"""

    def _format_1d_array(self, array, name, precision=4):
        """Format 1D array for C++ code with proper precision."""
        if not array:
            return f"const float {name}[] = {{}};"

        # Format values with proper line breaks (8 values per line)
        formatted_lines = []
        for i in range(0, len(array), 8):
            chunk = array[i:i+8]
            values_str = ", ".join(f"{val:.{precision}f}" for val in chunk)
            formatted_lines.append(f"    {values_str}")

        # Join lines with actual newlines
        all_values = ",\n".join(formatted_lines)
        return f"""const float {name}[{len(array)}] = {{
{all_values}
}};"""

    def _format_2d_array(self, array, name):
        """Format 2D array for C++ code with complete values."""
        if not array or not array[0]:
            return f"const float {name}[][] = {{}};"

        rows = len(array)
        cols = len(array[0]) if array else 0
        precision = self.feature_precision

        # Generate all values regardless of size (critical for compilation)
        formatted_rows = []
        for i, row in enumerate(array):
            # Format each row with proper values (8 values per line for readability)
            row_chunks = []
            for j in range(0, len(row), 8):
                chunk = row[j:j+8]
                chunk_str = ", ".join(f"{val:.{precision}f}" for val in chunk)
                row_chunks.append(f"        {chunk_str}")

            # Join chunks with line breaks
            row_content = ",\n".join(row_chunks)
            formatted_rows.append(f"    {{\n{row_content}\n    }}")

        # Join all rows
        rows_str = ",\n".join(formatted_rows)
        return f"""const float {name}[{rows}][{cols}] = {{
{rows_str}
}};"""

    def _generate_prediction_function(self) -> str:
        """Generate Neural Network prediction function."""
        return """// Internal neural network prediction function
// NOTE: This function expects ALREADY SCALED features from har_predict()
int har_predict_internal(float features[NUM_FEATURES]) {
    // Forward pass through hidden layer
    // Features are already scaled by har_predict() wrapper
    float hidden_outputs[HIDDEN_LAYER_SIZE];
    for (int h = 0; h < HIDDEN_LAYER_SIZE; h++) {
        float sum = hidden_biases[h];
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += features[i] * input_weights[i][h];
        }
        hidden_outputs[h] = relu(sum);  // ReLU activation
    }

    // Forward pass through output layer
    float output_scores[OUTPUT_SIZE];
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        float sum = output_biases[o];
        for (int h = 0; h < HIDDEN_LAYER_SIZE; h++) {
            sum += hidden_outputs[h] * output_weights[h][o];
        }
        output_scores[o] = sum;
    }

    // Find class with highest score
    int predicted_class = 0;
    float max_score = output_scores[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (output_scores[i] > max_score) {
            max_score = output_scores[i];
            predicted_class = i;
        }
    }

    return predicted_class;
}"""

    def _generate_utility_functions(self) -> str:
        """Generate Neural Network utility functions."""
        return """void print_network_outputs(float features[]) {
    Serial.println("Neural Network Layer Outputs:");

    // Show first few hidden layer outputs
    float hidden_outputs[HIDDEN_LAYER_SIZE];
    for (int h = 0; h < 5 && h < HIDDEN_LAYER_SIZE; h++) {
        float sum = hidden_biases[h];
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += features[i] * input_weights[i][h];
        }
        hidden_outputs[h] = relu(sum);
        Serial.print("Hidden[");
        Serial.print(h);
        Serial.print("]: ");
        Serial.println(hidden_outputs[h], 3);
    }
}"""
