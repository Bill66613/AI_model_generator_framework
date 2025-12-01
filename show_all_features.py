"""Generate complete feature comparison for debugging"""
import pandas as pd
import numpy as np
from utils.model_training import create_feature_vector
import joblib

# Load model
m = joblib.load(
    'persistent_data/neural_network_optimized_20251124_234953.joblib')
feature_names = m['feature_names']

# Load still data
df = pd.read_csv('persistent_data/dragged_window_0_still.csv')

# Python feature extraction
python_features = create_feature_vector(
    df, ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ'], 100, True)

print("="*80)
print("ALL 138 FEATURES - Python Feature Extraction")
print("="*80)
print("\nTIME-DOMAIN FEATURES (first 90):")
for i in range(90):
    print(
        f"{i:3d}. {feature_names[i]:35s} = {python_features.iloc[0, i]:15.6f}")

print("\nFREQUENCY-DOMAIN FEATURES (last 48):")
for i in range(90, 138):
    print(
        f"{i:3d}. {feature_names[i]:35s} = {python_features.iloc[0, i]:15.6f}")

print("\n" + "="*80)
print("If Arduino frequency features don't match these, that's the problem!")
print("="*80)
