"""
Test what the Python model predicts for a stationary device.
This will tell us if the problem is in the model itself or the C++ implementation.
"""
import joblib
import numpy as np
import pandas as pd
from utils.model_training import create_feature_vector

# Load the latest model
model_path = r"d:\Workspaces\Master\ComputerScience\Thesis\GUI_app\persistent_data_new\models\neural_network_har_model_20260205_005815.joblib"
model_dict = joblib.load(model_path)

model = model_dict['model']
scaler = model_dict['scaler']
label_encoder = model_dict['label_encoder']

print("Model loaded:")
print(f"  Classes: {list(label_encoder.classes_)}")
print()

# Create synthetic "still" data matching what device sees:
# acc=9.46 m/s², gyro=2.50 deg/s
n_samples = 150  # 1.5s @ 100Hz

# Still device: gravity only, minimal noise
np.random.seed(42)
acc_x = np.random.normal(0.0, 0.05, n_samples)
acc_y = np.random.normal(0.0, 0.05, n_samples)
acc_z = np.random.normal(9.46, 0.05, n_samples)  # Matches device reading
gyro_x = np.random.normal(0.0, 0.5, n_samples)
gyro_y = np.random.normal(0.0, 0.5, n_samples)
gyro_z = np.random.normal(0.0, 0.5, n_samples)

sensor_df = pd.DataFrame({
    'aX': acc_x,
    'aY': acc_y,
    'aZ': acc_z,
    'gX': gyro_x,
    'gY': gyro_y,
    'gZ': gyro_z
})

# Calculate magnitudes like device does
acc_mag = np.sqrt(sensor_df['aX']**2 + sensor_df['aY']**2 + sensor_df['aZ']**2)
gyro_mag = np.sqrt(sensor_df['gX']**2 +
                   sensor_df['gY']**2 + sensor_df['gZ']**2)

print("Synthetic 'still' sensor data:")
print(f"  Acc magnitude:  mean={acc_mag.mean():.2f}, std={acc_mag.std():.2f}")
print(
    f"  Gyro magnitude: mean={gyro_mag.mean():.2f}, std={gyro_mag.std():.2f}")
print(f"  (Device sees:   acc=9.46, gyro=2.50)")
print()

# Extract features using Python
sensor_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
features_df = create_feature_vector(
    sensor_df, sensor_cols, sampling_rate=100,
    include_frequency=False,
    orientation_robust=True,
    include_per_axis=False
)

features = features_df.values.flatten()
print(f"Extracted features: {len(features)}")
print(f"  First 5: {features[:5]}")
print(f"  Last 5: {features[-5:]}")
print()

# Scale features
scaled_features = scaler.transform([features])[0]
print(f"Scaled features (first 5): {scaled_features[:5]}")
print()

# Predict
pred_idx = model.predict([scaled_features])[0]
pred_name = label_encoder.inverse_transform([pred_idx])[0]

# Get probabilities
if hasattr(model, 'predict_proba'):
    proba = model.predict_proba([scaled_features])[0]
    print("Python Model Prediction:")
    print(f"  Predicted: {pred_name} (class {pred_idx})")
    print(f"  Probabilities:")
    for class_name, prob in zip(label_encoder.classes_, proba):
        print(f"    {class_name:20s}: {prob:.4f}")
else:
    print(f"Python Model Prediction: {pred_name} (class {pred_idx})")

print()
print("=" * 60)
if pred_name == 'still':
    print("✅ Python model CORRECTLY predicts 'still'")
    print("   → Issue is in C++ implementation")
else:
    print(f"❌ Python model INCORRECTLY predicts '{pred_name}'")
    print("   → Issue is in the MODEL ITSELF (needs retraining)")
print("=" * 60)
