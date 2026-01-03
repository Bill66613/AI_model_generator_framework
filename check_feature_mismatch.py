"""Check feature extraction mismatch"""
import pandas as pd
import numpy as np

# Load training data
df = pd.read_csv(
    'persistent_data_new/training/running_still_walking_and_2_more_train.csv')
feature_cols = [c for c in df.columns if c != 'label']

print('='*70)
print('TRAINING DATA FEATURES')
print('='*70)
print(f'Total features: {len(feature_cols)}')
print('\nFeature order in training data:')
for i, col in enumerate(feature_cols):
    print(f'  {i:2d}. {col}')

print('\n' + '='*70)
print('DEPLOYED CODE FEATURES (from .cpp)')
print('='*70)
deployed_features = [
    'acc_mag_mean', 'acc_mag_std', 'acc_mag_rms', 'acc_mag_energy', 'acc_mag_min', 'acc_mag_max',
    'gyro_mag_mean', 'gyro_mag_std', 'gyro_mag_rms', 'gyro_mag_energy', 'gyro_mag_min', 'gyro_mag_max',
    'jerk_mag_mean', 'jerk_mag_std', 'jerk_mag_rms', 'jerk_mag_energy', 'jerk_mag_max'
]
print(f'Total features: {len(deployed_features)}')
print('\nFeature order in deployed code:')
for i, feat in enumerate(deployed_features):
    print(f'  {i:2d}. {feat}')

print('\n' + '='*70)
print('COMPARISON')
print('='*70)

# Check if sets match
training_set = set(feature_cols)
deployed_set = set(deployed_features)

print(f'\nTraining has {len(feature_cols)} features')
print(f'Deployed code has {len(deployed_features)} features')

missing_in_deployment = training_set - deployed_set
extra_in_deployment = deployed_set - training_set

if missing_in_deployment:
    print(f'\n❌ Missing in deployment code: {missing_in_deployment}')
if extra_in_deployment:
    print(f'\n❌ Extra in deployment code: {extra_in_deployment}')

print('\n' + '='*70)
print('FEATURE ORDER COMPARISON')
print('='*70)
print(f"{'Index':<6} {'Training':<25} {'Deployed':<25} {'Match':<10}")
print('='*70)
max_len = max(len(feature_cols), len(deployed_features))
for i in range(max_len):
    train_feat = feature_cols[i] if i < len(feature_cols) else 'N/A'
    deploy_feat = deployed_features[i] if i < len(deployed_features) else 'N/A'
    match = '✅' if train_feat == deploy_feat else '❌'
    print(f'{i:<6} {train_feat:<25} {deploy_feat:<25} {match}')

print('\n' + '='*70)
print('SAMPLE VALUES')
print('='*70)
print('\nFirst row of training data:')
for col in feature_cols:
    print(f'  {col:<20} = {df[col].iloc[0]:>10.4f}')
