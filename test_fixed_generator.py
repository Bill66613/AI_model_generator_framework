"""
Test the fixed neural network generator with the actual model.
This should now correctly extract all 3 layers and generate proper code.
"""

import joblib
import os
from deployment import generate_and_save_deployment_code

# Load the trained model
model_path = 'persistent_data_new/models/neural_network_har_model_20251223_230846.joblib'
print(f"Loading model from: {model_path}")

if not os.path.exists(model_path):
    print(f"❌ Model file not found: {model_path}")
    exit(1)

model_data = joblib.load(model_path)

# Transform the model data structure to match what the deployment system expects
# The loaded model has: {'model': MLPClassifier, 'scaler': StandardScaler, ...}
# The deployment system expects: {'model_object': wrapper_with_model_attribute, ...}

# Create a wrapper object that matches the expected structure


class ModelWrapper:
    def __init__(self, model, scaler, label_encoder, model_type):
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.model_type = model_type


# Wrap the model
nn_model = model_data['model']

# Get feature names - if None, generate based on input size
if model_data.get('feature_names') is None:
    # Get input feature count from the model's first layer
    num_features = nn_model.coefs_[0].shape[0]
    feature_names = [f'feature_{i}' for i in range(num_features)]
else:
    feature_names = model_data['feature_names']

wrapped_data = {
    'model_object': ModelWrapper(
        model=nn_model,
        scaler=model_data['scaler'],
        label_encoder=model_data['label_encoder'],
        model_type='neural_network'
    ),
    'feature_names': feature_names,
    'classes': model_data['label_encoder'].classes_.tolist() if model_data['label_encoder'] else ['class_0', 'class_1', 'class_2', 'class_3', 'class_4'],
    'model_type': 'neural_network'
}

# Check model architecture
print(f"\n📊 Model Architecture:")
print(f"   Number of layers: {nn_model.n_layers_}")
print(f"   Hidden layer sizes: {nn_model.hidden_layer_sizes}")
print(f"   Weight matrices: {len(nn_model.coefs_)}")
for i, coef in enumerate(nn_model.coefs_):
    print(f"   Layer {i} → {i+1}: {coef.shape}")
for i, intercept in enumerate(nn_model.intercepts_):
    print(f"   Layer {i+1} bias: {intercept.shape}")

print(f"\n🔧 Generating deployment code...")

# Generate deployment code with the fixed generator
output_dir = "deployment_test_fixed"
saved_files = generate_and_save_deployment_code(
    model_type='neural_network',
    model_data=wrapped_data,  # Use the wrapped data
    platform='seeed_xiao',
    output_dir=output_dir,
    optimization='balanced'
)

print(f"\n✅ Generated files:")
for filepath, status in saved_files.items():
    print(f"   {filepath}")
    print(f"   {status}")

# Read and verify the generated .cpp file contains final_weights
cpp_files = [f for f in saved_files.keys() if f.endswith('.cpp')]
if cpp_files:
    cpp_file = cpp_files[0]
    print(f"\n🔍 Checking generated .cpp file: {cpp_file}")

    with open(cpp_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check for required components
    checks = {
        'input_weights': 'input_weights' in content,
        'hidden_biases': 'hidden_biases' in content,
        'hidden2_weights': 'hidden2_weights' in content,
        'hidden2_biases': 'hidden2_biases' in content,
        'final_weights': 'final_weights' in content,
        'final_biases': 'final_biases' in content,
        '3-layer comment': '3-layer' in content,
        'HIDDEN2_LAYER_SIZE': 'HIDDEN2_LAYER_SIZE' in content,
    }

    print("\n📋 Component Checklist:")
    for component, present in checks.items():
        status = "✅" if present else "❌"
        print(f"   {status} {component}")

    # Count layers in prediction function
    if 'hidden1_outputs' in content and 'hidden2_outputs' in content and 'output_scores' in content:
        print("\n✅ Prediction function implements full 3-layer architecture!")
    else:
        print("\n⚠️ Prediction function may not be correct")

print("\n✅ Code generation complete!")
print(f"📁 Output directory: {output_dir}")
