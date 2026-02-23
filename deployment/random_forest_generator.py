"""
Random Forest Code Generator
Generates Arduino C++ code specifically for Random Forest models
"""

import numpy as np
from typing import Dict, List, Any
from .base_generator import BaseCodeGenerator


class RandomForestCodeGenerator(BaseCodeGenerator):
    """Code generator specifically for Random Forest models."""

    def __init__(self, model_data: Dict[str, Any], platform: str = 'arduino', optimization: str = 'balanced', overlap: float = 0.5):
        super().__init__(model_data, platform, optimization, overlap)
        self.trees = model_data.get('trees', [])
        self.num_trees = len(self.trees)

    def _get_model_specific_declarations(self) -> str:
        """Generate Random Forest specific declarations."""
        return f"""
// Random Forest specific definitions
#define NUM_TREES {self.num_trees if self.num_trees > 0 else 100}
#define MAX_TREE_DEPTH 20

typedef struct {{
    int feature_idx;
    float threshold;
    int left_child;
    int right_child;
    float value;
}} TreeNode;

// Random Forest utility functions
int predict_tree(const TreeNode* nodes, int tree_start, float* features);
void print_tree_prediction_debug(float features[]);
void print_feature_vector(float features[]);
"""

    def _generate_model_specific_implementation(self) -> str:
        """Generate Random Forest implementation with actual tree structure."""
        if self.trees:
            tree_code = self._generate_tree_structures()
        else:
            tree_code = """// Placeholder tree structure - actual trees would be generated here
const TreeNode tree_nodes[] = {
    {0, 0.5, 1, 2, -1},   // Internal node: feature 0, threshold 0.5
    {-1, 0.0, -1, -1, 0}, // Leaf node: class 0
    {-1, 0.0, -1, -1, 1}  // Leaf node: class 1
};

const int tree_starts[] = {0}; // Start indices for each tree"""

        return f"""// Random Forest Model Implementation
{tree_code}

int predict_tree(const TreeNode* nodes, int tree_start, float* features) {{
    int current_node = tree_start;

    while (nodes[current_node].feature_idx >= 0) {{
        int feature_idx = nodes[current_node].feature_idx;
        float threshold = nodes[current_node].threshold;

        if (features[feature_idx] <= threshold) {{
            current_node = nodes[current_node].left_child;
        }} else {{
            current_node = nodes[current_node].right_child;
        }}
    }}

    return (int)nodes[current_node].value;
}}"""

    def _generate_prediction_function(self) -> str:
        """Generate Random Forest prediction function."""
        return f"""// Internal prediction function - receives ALREADY SCALED features from har_predict()
int har_predict_internal(float features[NUM_FEATURES]) {{
    // Random Forest prediction using all trees
    int votes[NUM_CLASSES] = {{0}};

    // Predict with each tree and accumulate votes
    for (int tree = 0; tree < NUM_TREES && tree < {min(self.num_trees, 100)}; tree++) {{
        int tree_prediction = predict_tree(tree_nodes, tree_starts[tree], features);
        if (tree_prediction >= 0 && tree_prediction < NUM_CLASSES) {{
            votes[tree_prediction]++;
        }}
    }}

    // Return class with most votes
    int max_votes = 0;
    int predicted_class = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {{
        if (votes[i] > max_votes) {{
            max_votes = votes[i];
            predicted_class = i;
        }}
    }}

    return predicted_class;
}}"""

    def _generate_tree_structures(self) -> str:
        """Generate actual tree structures from extracted trees."""
        if not self.trees:
            return "// No trees extracted from model"

        tree_nodes = []
        tree_starts = []
        current_index = 0

        # Use all extracted trees for embedded deployment
        for i, tree in enumerate(self.trees):
            tree_starts.append(current_index)

            # Convert sklearn tree structure to our format
            feature_indices = tree.get('feature_indices', [0])
            thresholds = tree.get('thresholds', [0.5])
            left_children = tree.get('left_children', [-1])
            right_children = tree.get('right_children', [-1])
            values = tree.get('values', [[1, 0, 0, 0]])  # Default values

            for j in range(len(feature_indices)):
                if feature_indices[j] >= 0:  # Internal node
                    tree_nodes.append(
                        f"    {{{feature_indices[j]}, {thresholds[j]:.6f}, {left_children[j] + current_index}, {right_children[j] + current_index}, -1}}")
                else:  # Leaf node
                    # Find class with maximum value
                    class_values = values[j][0] if len(
                        values) > j else [1, 0, 0, 0]
                    predicted_class = np.argmax(class_values) if isinstance(
                        class_values, list) else 0
                    tree_nodes.append(
                        f"    {{-1, 0.0, -1, -1, {predicted_class}}}")

            current_index += len(feature_indices)

        nodes_str = ",\n".join(tree_nodes)
        starts_str = ", ".join(map(str, tree_starts))

        return f"""// Random Forest Trees ({len(tree_starts)} of {self.num_trees} trees)
const TreeNode tree_nodes[] = {{
{nodes_str}
}};

const int tree_starts[] = {{{starts_str}}};"""

    def _generate_utility_functions(self) -> str:
        """Generate Random Forest utility functions (platform-portable)."""
        return """void print_tree_prediction_debug(float features[]) {
    HAR_LOG("Random Forest Tree Predictions:");
    for (int tree = 0; tree < NUM_TREES && tree < 5; tree++) {
        int prediction = predict_tree(tree_nodes, tree_starts[tree], features);
        HAR_LOG_FLOAT("Tree", (float)prediction);
    }
}

void print_feature_vector(float features[]) {
    HAR_LOG("Feature Vector (first 10):");
    for (int i = 0; i < 10 && i < NUM_FEATURES; i++) {
        HAR_LOG_FLOAT("F", features[i]);
    }
}"""
