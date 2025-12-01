"""
Check how many features the retrained model expects
"""
import joblib
import numpy as np

# Load the retrained model
model_path = "persistent_data/neural_network_time_only_20251126_025232.joblib"
print(f"Loading model from: {model_path}")

model_data = joblib.load(model_path)

# Check the scaler dimensions
scaler = model_data['scaler']
print(f"\nScaler expects {len(scaler.mean_)} features")
print(f"Feature means shape: {scaler.mean_.shape}")
print(f"Feature stds shape: {scaler.scale_.shape}")

# Check the neural network input layer
nn_model = model_data['model']
print(f"\nNeural Network input layer: {nn_model.n_features_in_} features")
print(f"Neural Network hidden layers: {nn_model.hidden_layer_sizes}")
print(f"Neural Network output layer: {nn_model.n_outputs_} classes")

# Try to predict with 90 features (time-domain only)
print("\n" + "="*50)
print("Testing with 90 features (time-domain only):")
try:
    test_features_90 = np.random.randn(1, 90)
    scaled_90 = scaler.transform(test_features_90)
    print(f"❌ ERROR: Model still expects 138 features, not 90!")
except ValueError as e:
    print(f"✅ CONFIRMED: Model expects 138 features as shown by error:")
    print(f"   {e}")

print("\n" + "="*50)
print("ISSUE: prepare_training_data() always extracts all 138 features!")
print("SOLUTION: Need to manually extract only 90 time-domain features")
print("="*50)
