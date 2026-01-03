"""Compute the correct scaler parameters from raw window data"""
import pandas as pd
import numpy as np
import glob

print('='*70)
print('COMPUTING CORRECT SCALER FROM RAW WINDOW DATA')
print('='*70)

# Load all windows
window_files = glob.glob('persistent_data_new/windows/dragged_window_*.csv')
print(f'\nFound {len(window_files)} window files')

# Extract features from each window (mimicking the training process)
features_list = []

for wfile in window_files:
    df_window = pd.read_csv(wfile)

    # Compute orientation-invariant features (EXACTLY as in training)
    acc_mag = np.sqrt(df_window['aX']**2 +
                      df_window['aY']**2 + df_window['aZ']**2)
    gyro_mag = np.sqrt(df_window['gX']**2 +
                       df_window['gY']**2 + df_window['gZ']**2)
    acc_diff = df_window[['aX', 'aY', 'aZ']].diff().fillna(0.0)
    jerk_mag = np.sqrt(acc_diff['aX']**2 +
                       acc_diff['aY']**2 + acc_diff['aZ']**2)

    feats = [
        float(np.mean(acc_mag)),       # 0
        float(np.std(acc_mag)),        # 1
        float(np.sqrt(np.mean(acc_mag**2))),  # 2: rms
        float(np.sum(acc_mag**2)),     # 3: energy
        float(np.mean(gyro_mag)),      # 4
        float(np.std(gyro_mag)),       # 5
        float(np.sqrt(np.mean(gyro_mag**2))),  # 6: rms
        float(np.sum(gyro_mag**2)),    # 7: energy
        float(np.mean(jerk_mag)),      # 8
        float(np.std(jerk_mag)),       # 9
        float(np.sqrt(np.mean(jerk_mag**2))),  # 10: rms
        float(np.sum(jerk_mag**2)),    # 11: energy
        float(np.min(acc_mag)),        # 12
        float(np.max(acc_mag)),        # 13
        float(np.min(gyro_mag)),       # 14
        float(np.max(gyro_mag)),       # 15
        float(np.max(jerk_mag))        # 16
    ]
    features_list.append(feats)

# Convert to numpy array
X_raw = np.array(features_list)

print(f'Extracted features from {X_raw.shape[0]} windows')
print(f'Feature dimension: {X_raw.shape[1]}')

# Compute scaler parameters
feature_means = X_raw.mean(axis=0)
feature_stds = X_raw.std(axis=0)

print('\n' + '='*70)
print('CORRECT SCALER PARAMETERS (from raw data)')
print('='*70)

feature_names = [
    'acc_mag_mean', 'acc_mag_std', 'acc_mag_rms', 'acc_mag_energy',
    'gyro_mag_mean', 'gyro_mag_std', 'gyro_mag_rms', 'gyro_mag_energy',
    'jerk_mag_mean', 'jerk_mag_std', 'jerk_mag_rms', 'jerk_mag_energy',
    'acc_mag_min', 'acc_mag_max', 'gyro_mag_min', 'gyro_mag_max', 'jerk_mag_max'
]

print('\nfeature_means:')
for i, (name, mean) in enumerate(zip(feature_names, feature_means)):
    print(f'  [{i:2d}] {name:<20} = {mean:>12.4f}')

print('\nfeature_stds:')
for i, (name, std) in enumerate(zip(feature_names, feature_stds)):
    print(f'  [{i:2d}] {name:<20} = {std:>12.4f}')

# Generate C++ array
print('\n' + '='*70)
print('C++ ARRAYS FOR DEPLOYMENT CODE')
print('='*70)

print('\nconst float feature_means[NUM_FEATURES] = {')
for i in range(0, 17, 8):
    vals = ', '.join([f'{v:.3f}' for v in feature_means[i:min(i+8, 17)]])
    print(f'    {vals}{"," if i+8 < 17 else ""}')
print('};')

print('\nconst float feature_stds[NUM_FEATURES] = {')
for i in range(0, 17, 8):
    vals = ', '.join([f'{v:.3f}' for v in feature_stds[i:min(i+8, 17)]])
    print(f'    {vals}{"," if i+8 < 17 else ""}')
print('};')

print('\n' + '='*70)
print('VERIFICATION')
print('='*70)

# Test with the still device values from user's output
print('\nUser\'s still device output:')
print('  acc_mag_mean (raw) = 9.52')
print('  gyro_mag_mean (raw) = 2.50')

# Apply scaling
acc_scaled = (9.52 - feature_means[0]) / feature_stds[0]
gyro_scaled = (2.50 - feature_means[4]) / feature_stds[4]

print(f'\nAfter correct scaling:')
print(f'  acc_mag_mean (scaled) = {acc_scaled:.4f}')
print(f'  gyro_mag_mean (scaled) = {gyro_scaled:.4f}')

print('\nExpected scaled values from training data (still activity):')
train_df = pd.read_csv(
    'persistent_data_new/training/running_still_walking_and_2_more_train.csv')
still_sample = train_df[train_df['label'] == 'still'].iloc[0]
print(f'  acc_mag_mean (expected) = {still_sample["acc_mag_mean"]:.4f}')
print(f'  gyro_mag_mean (expected) = {still_sample["gyro_mag_mean"]:.4f}')

if abs(acc_scaled - still_sample["acc_mag_mean"]) < 0.5 and abs(gyro_scaled - still_sample["gyro_mag_mean"]) < 0.5:
    print('\n✅ SCALING IS CORRECT! Values match expected range.')
else:
    print('\n⚠️  WARNING: Scaled values still don\'t match expected range.')
    print('   There may be additional preprocessing steps.')
