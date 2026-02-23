"""
Neural Network Code Generator
Generates Arduino C++ code specifically for Neural Network models
"""

import numpy as np
from typing import Dict, Any
from .base_generator import BaseCodeGenerator


class NeuralNetworkCodeGenerator(BaseCodeGenerator):
    """Code generator specifically for Neural Network models."""

    def __init__(self, model_data: Dict[str, Any], platform: str = 'arduino', optimization: str = 'balanced', overlap: float = 0.5):
        super().__init__(model_data, platform, optimization, overlap)
        self.weights = model_data.get('weights', {})
        self.hidden_size = self.weights.get('hidden_size', 50)

        # Multi-layer network support
        self.num_hidden_layers = 1  # Default to 1 hidden layer
        self.hidden_layer_sizes = []  # Will store sizes of all hidden layers
        self.all_weights = []  # Store all weight matrices
        self.all_biases = []   # Store all bias vectors

        # Extract real model weights if available
        self._extract_real_weights()

    def _extract_real_weights(self):
        """Extract actual weights from trained model object or direct weights."""
        # First try to extract from model object (scikit-learn MLPClassifier)
        if 'model_object' in self.model_data:
            model_obj = self.model_data['model_object']
            try:
                # --- PyTorch MLP path (export_mlp_weights returns sklearn-compatible format) ---
                if hasattr(model_obj, '_pytorch_trainer'):
                    mlp_export = model_obj._pytorch_trainer.export_mlp_weights()
                    coefs = mlp_export['coefs_']
                    intercepts = mlp_export['intercepts_']
                    self.hidden_layer_sizes = list(mlp_export.get('hidden_layer_sizes', ()))
                    self._populate_from_coefs(coefs, intercepts)
                    return

                # --- scikit-learn MLPClassifier path ---
                if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'coefs_'):
                    # Extract weights from scikit-learn MLPClassifier
                    coefs = model_obj.model.coefs_
                    intercepts = model_obj.model.intercepts_

                    # Get hidden layer sizes from model
                    if hasattr(model_obj.model, 'hidden_layer_sizes'):
                        if isinstance(model_obj.model.hidden_layer_sizes, tuple):
                            self.hidden_layer_sizes = list(
                                model_obj.model.hidden_layer_sizes)
                        else:
                            self.hidden_layer_sizes = [
                                model_obj.model.hidden_layer_sizes]

                    self._populate_from_coefs(coefs, intercepts)
                    return

            except Exception as e:
                print(
                    f"Warning: Could not extract neural network weights from model object: {e}")

        # Try to extract from direct weights parameter
        if self.weights:
            try:
                self.input_weights = self.weights.get('input_weights', [])
                self.hidden_biases = self.weights.get('input_bias', [])
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

    def _populate_from_coefs(self, coefs, intercepts):
        """Populate weight attributes from coefs/intercepts arrays.

        Works identically for sklearn MLPClassifier weights and PyTorch
        export_mlp_weights() output (both use the same array shapes).
        """
        import numpy as np

        self.num_hidden_layers = len(
            self.hidden_layer_sizes) if self.hidden_layer_sizes else len(coefs) - 1

        # Ensure numpy arrays for indexing
        coefs = [np.asarray(c) for c in coefs]
        intercepts = [np.asarray(b) for b in intercepts]

        self.all_weights = [c.tolist() for c in coefs]
        self.all_biases = [b.tolist() for b in intercepts]

        # Apply feature reorder to input weights if needed
        reorder_indices = self.model_data.get('_feature_reorder_indices')
        if reorder_indices is not None:
            reordered_coef0 = coefs[0][reorder_indices, :]
            self.all_weights[0] = reordered_coef0.tolist()
            print(f"  Applied feature reorder to input weight matrix")
        else:
            reordered_coef0 = coefs[0]

        # For backwards compatibility, set common attributes
        if len(coefs) >= 1:
            self.input_weights = reordered_coef0.tolist()
            self.hidden_biases = intercepts[0].tolist()
            self.hidden_size = len(intercepts[0])

        if len(coefs) >= 2:
            self.output_weights = coefs[1].tolist()
            self.output_biases = intercepts[1].tolist()

        # For multi-hidden-layer networks
        if len(coefs) >= 3:
            self.hidden2_weights = coefs[1].tolist()
            self.hidden2_biases = intercepts[1].tolist()
            self.hidden2_size = len(intercepts[1])
            self.final_weights = coefs[2].tolist()
            self.final_biases = intercepts[2].tolist()
            print(f"✅ Extracted {len(coefs)}-layer NN: {[c.shape for c in coefs]}")
        else:
            print(f"✅ Extracted {len(coefs)}-layer NN: "
                  f"{coefs[0].shape[0]}→{coefs[0].shape[1]}→"
                  f"{coefs[1].shape[1] if len(coefs) > 1 else '?'}")

    def _get_model_specific_declarations(self) -> str:
        """Generate Neural Network specific declarations."""
        # Check if this is a multi-hidden-layer network
        declarations = f"""
// Neural Network specific definitions
#define HIDDEN_LAYER_SIZE {self.hidden_size}
#define INPUT_SIZE NUM_FEATURES
#define OUTPUT_SIZE NUM_CLASSES
"""

        # Add second hidden layer size if it exists
        if hasattr(self, 'hidden2_size'):
            declarations += f"#define HIDDEN2_LAYER_SIZE {self.hidden2_size}\n"

        declarations += """
// Neural Network utility functions
float sigmoid(float x);
float relu(float x);
void print_network_outputs(float features[]);
"""
        return declarations

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

        # Generate output biases array (or hidden2 biases for 3-layer networks)
        if hasattr(self, 'output_biases') and self.output_biases:
            # Check if this is actually hidden2 layer (for 3-layer networks)
            if hasattr(self, 'final_weights'):
                # This is a 3-layer network - rename output_biases to hidden2_biases
                output_biases_str = self._format_1d_array(
                    self.output_biases, "hidden2_biases", self.feature_precision)
            else:
                # This is a 2-layer network - keep as output_biases
                output_biases_str = self._format_1d_array(
                    self.output_biases, "output_biases", self.feature_precision)
        else:
            output_biases_str = f"""// Placeholder output biases
const float output_biases[OUTPUT_SIZE] = {{
    // Would contain actual output biases from trained model
}};"""

        # Generate final layer weights if this is a 3-layer network
        final_weights_str = ""
        final_biases_str = ""
        if hasattr(self, 'final_weights') and self.final_weights:
            final_weights_str = self._format_2d_array(
                self.final_weights, "final_weights")
            final_biases_str = self._format_1d_array(
                self.final_biases, "final_biases", self.feature_precision)

        has_real_weights = hasattr(
            self, 'input_weights') and self.input_weights
        is_multilayer = hasattr(self, 'final_weights')

        # For 3-layer networks, rename output_weights to hidden2_weights
        if is_multilayer:
            output_weights_str = output_weights_str.replace(
                "output_weights", "hidden2_weights")

        implementation = f"""// Neural Network Model Implementation
// Using real weights: {has_real_weights}
// Architecture: {'3-layer (Input→Hidden1→Hidden2→Output)' if is_multilayer else '2-layer (Input→Hidden→Output)'}

{input_weights_str}

{hidden_biases_str}

{output_weights_str}

{output_biases_str}"""

        # Add final layer weights for 3-layer networks
        if final_weights_str:
            implementation += f"""

{final_weights_str}

{final_biases_str}"""

        implementation += """

// Activation functions
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float relu(float x) {
    return x > 0 ? x : 0;
}"""
        return implementation

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
        # Check if this is a 3-layer network
        is_multilayer = hasattr(self, 'final_weights')

        if is_multilayer:
            # 3-layer network: Input → Hidden1 → Hidden2 → Output
            return """// Internal neural network prediction function (3-layer architecture)
// NOTE: This function expects ALREADY SCALED features from har_predict()
int har_predict_internal(float features[NUM_FEATURES]) {
    // Layer 1: Input → Hidden1 (ReLU activation)
    float hidden1_outputs[HIDDEN_LAYER_SIZE];
    for (int h = 0; h < HIDDEN_LAYER_SIZE; h++) {
        float sum = hidden_biases[h];
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += features[i] * input_weights[i][h];
        }
        hidden1_outputs[h] = relu(sum);
    }

    // Layer 2: Hidden1 → Hidden2 (ReLU activation)
    float hidden2_outputs[HIDDEN2_LAYER_SIZE];
    for (int h = 0; h < HIDDEN2_LAYER_SIZE; h++) {
        float sum = hidden2_biases[h];
        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
            sum += hidden1_outputs[i] * hidden2_weights[i][h];
        }
        hidden2_outputs[h] = relu(sum);
    }

    // Layer 3: Hidden2 → Output (Linear activation)
    float output_scores[OUTPUT_SIZE];
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        float sum = final_biases[o];
        for (int h = 0; h < HIDDEN2_LAYER_SIZE; h++) {
            sum += hidden2_outputs[h] * final_weights[h][o];
        }
        output_scores[o] = sum;  // Linear activation (no ReLU on output)
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
        else:
            # 2-layer network: Input → Hidden → Output
            return """// Internal neural network prediction function (2-layer architecture)
// NOTE: This function expects ALREADY SCALED features from har_predict()
int har_predict_internal(float features[NUM_FEATURES]) {
    // Layer 1: Input → Hidden (ReLU activation)
    float hidden_outputs[HIDDEN_LAYER_SIZE];
    for (int h = 0; h < HIDDEN_LAYER_SIZE; h++) {
        float sum = hidden_biases[h];
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += features[i] * input_weights[i][h];
        }
        hidden_outputs[h] = relu(sum);
    }

    // Layer 2: Hidden → Output (Linear activation)
    float output_scores[OUTPUT_SIZE];
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        float sum = output_biases[o];
        for (int h = 0; h < HIDDEN_LAYER_SIZE; h++) {
            sum += hidden_outputs[h] * output_weights[h][o];
        }
        output_scores[o] = sum;  // Linear activation
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
        """Generate Neural Network utility functions (platform-portable)."""
        return """void print_network_outputs(float features[]) {
    HAR_LOG("Neural Network Layer Outputs:");

    // Show first few hidden layer outputs
    float hidden_outputs[HIDDEN_LAYER_SIZE];
    for (int h = 0; h < 5 && h < HIDDEN_LAYER_SIZE; h++) {
        float sum = hidden_biases[h];
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += features[i] * input_weights[i][h];
        }
        hidden_outputs[h] = relu(sum);
        HAR_LOG_FLOAT("Hidden", hidden_outputs[h]);
    }
}"""
