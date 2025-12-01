"""Test model predictions with known still vs running data"""
from utils.model_training import create_feature_vector
import joblib
import numpy as np
import pandas as pd

# Load model
model_file = 'persistent_data/neural_network_optimized_20251124_234953.joblib'
m = joblib.load(model_file)
model = m['model']
scaler = m['scaler']
label_encoder = m['label_encoder']

# Load example still and running data
still_df = pd.read_csv('persistent_data/dragged_window_0_still.csv')
running_df = pd.read_csv('persistent_data/dragged_window_0_running.csv')

# Extract features using the training function

still_features = create_feature_vector(
    still_df, ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ'], 100, True)
running_features = create_feature_vector(
    running_df, ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ'], 100, True)

# Scale features
still_scaled = scaler.transform(still_features)
running_scaled = scaler.transform(running_features)

# Get predictions
still_pred = model.predict(still_scaled)
running_pred = model.predict(running_scaled)

# Get probabilities
still_proba = model.predict_proba(still_scaled)
running_proba = model.predict_proba(running_scaled)

label_names = ['running', 'still', 'walking',
               'walking_downstairs', 'walking_upstairs']

print("="*70)
print("TESTING MODEL WITH KNOWN DATA")
print("="*70)
print("\nSTILL DATA (acc=9.47, gyro=2.37):")
print(f"  Predicted class: {still_pred[0]} ({label_names[still_pred[0]]})")
print(f"  Probabilities:")
for i, prob in enumerate(still_proba[0]):
    print(f"    {label_names[i]}: {prob:.4f}")

print("\nRUNNING DATA (acc=15.67, gyro=195.74):")
print(f"  Predicted class: {running_pred[0]} ({label_names[running_pred[0]]})")
print(f"  Probabilities:")
for i, prob in enumerate(running_proba[0]):
    print(f"    {label_names[i]}: {prob:.4f}")

print("\n" + "="*70)
print("If model predicts correctly, this proves feature extraction mismatch")
print("If model predicts incorrectly, model itself has issues")
print("="*70)
