"""Analyze motion characteristics of training data"""
import pandas as pd
import numpy as np
import glob

files = glob.glob('persistent_data/dragged_window_*_*.csv')
stats = {}

for f in files:
    label = f.split('_')[-1].replace('.csv', '')
    if label not in stats:
        stats[label] = {'files': [], 'mean_acc': [],
                        'std_acc': [], 'mean_gyro': [], 'std_gyro': []}

    df = pd.read_csv(f)

    # Calculate acceleration magnitude
    acc_mag = np.sqrt(df['aX']**2 + df['aY']**2 + df['aZ']**2)
    stats[label]['mean_acc'].append(acc_mag.mean())
    stats[label]['std_acc'].append(acc_mag.std())

    # Calculate gyroscope magnitude
    gyro_mag = np.sqrt(df['gX']**2 + df['gY']**2 + df['gZ']**2)
    stats[label]['mean_gyro'].append(gyro_mag.mean())
    stats[label]['std_gyro'].append(gyro_mag.std())

print('\nActivity Motion Characteristics:')
print('='*70)
print(f"{'Activity':<20} {'Acc Mean':<12} {'Acc Std':<12} {'Gyro Mean':<12} {'Gyro Std':<12}")
print('='*70)

for label in sorted(stats.keys()):
    s = stats[label]
    acc_mean = np.mean(s['mean_acc'])
    acc_std = np.mean(s['std_acc'])
    gyro_mean = np.mean(s['mean_gyro'])
    gyro_std = np.mean(s['std_gyro'])

    print(f"{label:<20} {acc_mean:>10.2f}   {acc_std:>10.2f}   {gyro_mean:>10.2f}   {gyro_std:>10.2f}")

print('='*70)
print("\nExpected order (by motion intensity):")
print("1. still - lowest acceleration & gyro")
print("2. walking - moderate motion")
print("3. walking_upstairs/downstairs - higher motion")
print("4. running - highest motion")
