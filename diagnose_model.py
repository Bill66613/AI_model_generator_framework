"""Diagnose why the model always predicts 'running' (class 0)"""
import joblib
import numpy as np
import pandas as pd
import glob

print("="*70)
print("NEURAL NETWORK MODEL DIAGNOSTIC")
print("="*70)

# Load the model
model_data = joblib.load(
    'persistent_data/neural_network_optimized_20251124_234953.joblib')
print(f"\n1. Model file structure:")
print(f"   Keys: {list(model_data.keys())}")

# Extract the actual model
model = model_data.get('model') or model_data.get(
    'best_estimator_') or model_data

print(f"\n2. Model information:")
print(f"   Type: {type(model).__name__}")
if hasattr(model, 'classes_'):
    print(f"   Classes: {model.classes_}")
    print(f"   Class mapping: {dict(enumerate(model.classes_))}")
if hasattr(model, 'n_features_in_'):
    print(f"   Input features: {model.n_features_in_}")
if hasattr(model, 'coefs_'):
    print(f"   Hidden layer shape: {model.coefs_[0].shape}")
    print(f"   Output layer shape: {model.coefs_[1].shape}")
    print(f"\n3. Weight statistics:")
    print(
        f"   Hidden weights: min={model.coefs_[0].min():.6f}, max={model.coefs_[0].max():.6f}")
    print(
        f"   Output weights: min={model.coefs_[1].min():.6f}, max={model.coefs_[1].max():.6f}")
    print(f"\n4. Bias statistics:")
    print(f"   Hidden bias: {model.intercepts_[0][:5]}")
    print(f"   Output bias: {model.intercepts_[1]}")

# Test with zero features
print(f"\n5. Test predictions:")
zero_features = np.zeros((1, 138))
pred_zero = model.predict(zero_features)
print(f"   All zeros → Predicted class: {pred_zero[0]}")

# Test with real data sample
print(f"\n6. Testing with real training data:")
window_file = glob.glob('persistent_data/dragged_window_0_*.csv')[0]
df = pd.read_csv(window_file)
print(f"   Using: {window_file}")
print(f"   Samples in window: {len(df)}")

# Extract first 75 samples for a window
if len(df) >= 75:
    window_data = df.iloc[:75][['AccX', 'AccY',
                                'AccZ', 'GyrX', 'GyrY', 'GyrZ']].values
    print(f"   Sensor data shape: {window_data.shape}")
    print(f"   Sensor data range:")
    print(
        f"     AccX: {window_data[:, 0].min():.3f} to {window_data[:, 0].max():.3f}")
    print(
        f"     AccY: {window_data[:, 1].min():.3f} to {window_data[:, 1].max():.3f}")
    print(
        f"     AccZ: {window_data[:, 2].min():.3f} to {window_data[:, 2].max():.3f}")

    # Extract features (simplified - just basic stats for diagnosis)
    from utils.feature_extraction import extract_features_from_dataframe

    # Create a proper test
    features = []
    for axis_idx in range(6):
        axis_data = window_data[:, axis_idx]
        features.extend([
            np.mean(axis_data),
            np.std(axis_data),
            np.min(axis_data),
            np.max(axis_data),
            np.max(axis_data) - np.min(axis_data),  # range
        ])

    # Pad with zeros to reach 138 features (simplified version)
    features = np.array(features + [0] * (138 - len(features))).reshape(1, -1)

    pred_real = model.predict(features)
    pred_proba = model.predict_proba(features)

    print(f"\n7. Prediction on real data:")
    print(f"   Predicted class: {pred_real[0]}")
    print(f"   Class probabilities:")
    for i, prob in enumerate(pred_proba[0]):
        class_name = model.classes_[i] if hasattr(
            model, 'classes_') else f"Class {i}"
        print(f"     {class_name:20s}: {prob:.4f}")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

if hasattr(model, 'intercepts_'):
    output_bias = model.intercepts_[1]
    max_bias_idx = np.argmax(output_bias)
    print(
        f"✓ Highest output bias: Class {max_bias_idx} ({model.classes_[max_bias_idx]})")
    print(f"  Bias value: {output_bias[max_bias_idx]:.4f}")
    print(f"\n  All output biases: {output_bias}")

    if max_bias_idx == 0:
        print(f"\n⚠️  Class 0 (running) has the highest bias!")
        print(f"   This explains why it's always predicted.")
        print(f"\n   POSSIBLE CAUSES:")
        print(f"   1. Training data had running as first class")
        print(f"   2. Model training converged poorly")
        print(f"   3. Feature extraction mismatch between training and inference")
        print(f"\n   SOLUTION: Retrain the model with balanced data")
