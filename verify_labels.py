"""Check if training data labels are correct"""
import pandas as pd
import numpy as np
import glob

print("="*70)
print("VERIFYING TRAINING DATA LABELS")
print("="*70)

# Check a few files from each class
for activity in ['still', 'running', 'walking']:
    files = glob.glob(f'persistent_data/dragged_window_*_{activity}.csv')[:3]
    print(f"\n{activity.upper()} files:")
    for f in files:
        df = pd.read_csv(f)
        acc_mag = np.sqrt(df['aX']**2 + df['aY']**2 + df['aZ']**2).mean()
        gyro_mag = np.sqrt(df['gX']**2 + df['gY']**2 + df['gZ']**2).mean()
        print(
            f"  {f.split('/')[-1]:40s} acc={acc_mag:6.2f}  gyro={gyro_mag:7.2f}")

print("\n" + "="*70)
print("Expected ranges:")
print("  still:   acc ~9.5,  gyro ~2-3")
print("  walking: acc ~11,   gyro ~90-130")
print("  running: acc ~15,   gyro ~190-200")
print("="*70)
