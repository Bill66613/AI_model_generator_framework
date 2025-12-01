"""Compare Python vs Arduino feature extraction"""
import pandas as pd
import numpy as np
from utils.model_training import create_feature_vector

# Load still data
df = pd.read_csv('persistent_data/dragged_window_0_still.csv')

# Python feature extraction
python_features = create_feature_vector(
    df, ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ'], 100, True)

print("="*70)
print("PYTHON FEATURE EXTRACTION (first 20 features):")
print("="*70)
feature_names = ['aX_mean', 'aX_std', 'aX_min', 'aX_max', 'aX_range', 'aX_median', 'aX_q25', 'aX_q75', 'aX_iqr',
                 'aX_skewness', 'aX_kurtosis', 'aX_rms', 'aX_energy', 'aX_zero_crossings', 'aX_mean_crossing_rate',
                 'aY_mean', 'aY_std', 'aY_min', 'aY_max', 'aY_range']

for i, name in enumerate(feature_names):
    print(f"{i:3d}. {name:25s} = {python_features.iloc[0, i]:12.4f}")

print("\n" + "="*70)
print("SIMULATED ARDUINO FEATURE EXTRACTION (first 15 features):")
print("="*70)

# Simulate Arduino feature extraction for aX axis
axis_data = df['aX'].values
samples = len(axis_data)

# Sort for median/quartiles
sorted_data = np.sort(axis_data)

# Basic stats
mean = np.mean(axis_data)
std = np.std(axis_data)
min_val = np.min(axis_data)
max_val = np.max(axis_data)
range_val = max_val - min_val

# Median and quartiles
mid = samples // 2
median = (sorted_data[mid-1] + sorted_data[mid]) / \
    2 if samples % 2 == 0 else sorted_data[mid]
q1_idx = samples // 4
q3_idx = (3 * samples) // 4
q25 = sorted_data[q1_idx]
q75 = sorted_data[q3_idx]
iqr = q75 - q25

# Skewness and kurtosis (using pandas-compatible formulas)
m2 = np.sum(((axis_data - mean) / (std + 0.001))**2) / samples
m3 = np.sum(((axis_data - mean) / (std + 0.001))**3) / samples
m4 = np.sum(((axis_data - mean) / (std + 0.001))**4) / samples

# Adjusted skewness (pandas formula with bias correction)
if samples >= 3 and m2 > 0:
    adjust = np.sqrt(samples * (samples - 1)) / (samples - 2)
    skewness = adjust * m3 / (m2**1.5)
else:
    skewness = 0

# Excess kurtosis (pandas formula with bias correction)
if samples >= 4 and m2 > 0:
    adjust1 = (samples * (samples + 1) * (samples - 1)) / \
        ((samples - 2) * (samples - 3))
    adjust2 = 3.0 * (samples - 1) * (samples - 1) / \
        ((samples - 2) * (samples - 3))
    kurtosis = adjust1 * (m4 / (m2 * m2)) - adjust2 - 3.0
else:
    kurtosis = -3.0

# RMS and energy
rms = np.sqrt(np.mean(axis_data**2))
energy = np.sum(axis_data**2)

# Zero crossings
zero_crossings = np.sum(np.diff(np.sign(axis_data)) != 0)

# Mean crossings
mean_crossings = np.sum(np.diff(np.sign(axis_data - mean)) != 0)

arduino_features = [mean, std, min_val, max_val, range_val, median, q25, q75, iqr,
                    skewness, kurtosis, rms, energy, zero_crossings, mean_crossings]

arduino_names = ['mean', 'std', 'min', 'max', 'range', 'median', 'q25', 'q75', 'iqr',
                 'skewness', 'kurtosis', 'rms', 'energy', 'zero_crossings', 'mean_crossings']

for i, (name, val) in enumerate(zip(arduino_names, arduino_features)):
    py_val = python_features.iloc[0, i]
    diff = abs(val - py_val)
    match = "✅" if diff < 0.01 else "❌"
    print(f"{i:3d}. {name:25s} = {val:12.4f}  (Python: {py_val:12.4f})  {match}")

print("\n" + "="*70)
