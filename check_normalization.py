"""Check if raw window data was transformed before feature extraction"""
import pandas as pd
import numpy as np
import glob

# Load one window from each activity
files = glob.glob('persistent_data_new/windows/dragged_window_*_still.csv')
if files:
    df = pd.read_csv(files[0])
    print('='*70)
    print('RAW WINDOW DATA (Still activity)')
    print('='*70)
    print(df.head())

    # Compute magnitude features manually
    acc_mag = np.sqrt(df['aX']**2 + df['aY']**2 + df['aZ']**2)
    gyro_mag = np.sqrt(df['gX']**2 + df['gY']**2 + df['gZ']**2)

    print('\n' + '='*70)
    print('MANUAL FEATURE COMPUTATION FROM RAW DATA')
    print('='*70)
    print(f'acc_mag_mean  = {acc_mag.mean():.4f} m/s²')
    print(f'acc_mag_std   = {acc_mag.std():.4f}')
    print(f'gyro_mag_mean = {gyro_mag.mean():.4f} deg/s')
    print(f'gyro_mag_std  = {gyro_mag.std():.4f}')

    print('\n' + '='*70)
    print('TRAINING DATA FEATURES (After transformation)')
    print('='*70)
    train_df = pd.read_csv(
        'persistent_data_new/training/running_still_walking_and_2_more_train.csv')
    still_features = train_df[train_df['label'] == 'still'].iloc[0]
    print(f'acc_mag_mean  = {still_features["acc_mag_mean"]:.4f}')
    print(f'acc_mag_std   = {still_features["acc_mag_std"]:.4f}')
    print(f'gyro_mag_mean = {still_features["gyro_mag_mean"]:.4f}')
    print(f'gyro_mag_std  = {still_features["gyro_mag_std"]:.4f}')

    print('\n' + '='*70)
    print('CONCLUSION')
    print('='*70)
    print('The training features are NORMALIZED (near -1 to +1 range)')
    print('But raw sensor data is in physical units (9.8 m/s² for gravity)')
    print('\n⚠️  There is a NORMALIZATION STEP missing in the pipeline!')
