"""
Convert UCI-HAR Dataset to framework-compatible CSV format.
Each activity will be saved as a separate CSV file with columns:
timestamp, ax, ay, az, gx, gy, gz
"""

import numpy as np
import pandas as pd
import os

# Dataset paths
dataset_path = "data/UCI-HAR Dataset"
output_path = "data/UCI-HAR_converted"

# Create output directory
os.makedirs(output_path, exist_ok=True)

# Activity labels mapping
activity_labels = {
    1: "walking",
    2: "walking_upstairs", 
    3: "walking_downstairs",
    4: "sitting",
    5: "standing",
    6: "laying"
}

def load_inertial_signals(dataset_type):
    """Load raw inertial signals from train or test set."""
    base_path = f"{dataset_path}/{dataset_type}/Inertial Signals"
    
    # Load accelerometer data (total acceleration in g units)
    acc_x = np.loadtxt(f"{base_path}/total_acc_x_{dataset_type}.txt")
    acc_y = np.loadtxt(f"{base_path}/total_acc_y_{dataset_type}.txt")
    acc_z = np.loadtxt(f"{base_path}/total_acc_z_{dataset_type}.txt")
    
    # Load gyroscope data (angular velocity in rad/s)
    gyro_x = np.loadtxt(f"{base_path}/body_gyro_x_{dataset_type}.txt")
    gyro_y = np.loadtxt(f"{base_path}/body_gyro_y_{dataset_type}.txt")
    gyro_z = np.loadtxt(f"{base_path}/body_gyro_z_{dataset_type}.txt")
    
    # Load labels
    labels = np.loadtxt(f"{dataset_path}/{dataset_type}/y_{dataset_type}.txt", dtype=int)
    
    return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, labels

def convert_to_csv():
    """Convert UCI-HAR data to per-activity CSV files."""
    
    # Process both train and test sets
    all_data = {activity: [] for activity in activity_labels.values()}
    
    for dataset_type in ['train', 'test']:
        print(f"Processing {dataset_type} set...")
        
        # Load all inertial signals
        acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, labels = load_inertial_signals(dataset_type)
        
        # Each row is a 128-sample window (2.56 seconds at 50Hz)
        n_windows = acc_x.shape[0]
        samples_per_window = acc_x.shape[1]
        
        print(f"  Found {n_windows} windows, {samples_per_window} samples each")
        
        # Group data by activity
        for window_idx in range(n_windows):
            activity_id = labels[window_idx]
            activity_name = activity_labels[activity_id]
            
            # Extract window data
            window_data = {
                'ax': acc_x[window_idx],
                'ay': acc_y[window_idx],
                'az': acc_z[window_idx],
                'gx': gyro_x[window_idx],
                'gy': gyro_y[window_idx],
                'gz': gyro_z[window_idx]
            }
            
            all_data[activity_name].append(window_data)
    
    # Save each activity to separate CSV file
    sampling_rate = 50  # Hz
    time_step = 1.0 / sampling_rate  # 0.02 seconds
    
    for activity_name, windows in all_data.items():
        if not windows:
            continue
            
        print(f"\nCreating {activity_name}.csv...")
        
        # Concatenate all windows for this activity
        activity_samples = []
        current_time = 0.0
        
        for window in windows:
            # Each window has 128 samples
            for i in range(len(window['ax'])):
                sample = {
                    'aX': window['ax'][i],
                    'aY': window['ay'][i],
                    'aZ': window['az'][i],
                    'gX': window['gx'][i],
                    'gY': window['gy'][i],
                    'gZ': window['gz'][i]
                }
                activity_samples.append(sample)
                current_time += time_step
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(activity_samples)
        output_file = f"{output_path}/{activity_name}.csv"
        df.to_csv(output_file, index=False, float_format='%.6f')
        
        print(f"  Saved {len(activity_samples)} samples ({len(windows)} windows)")
        print(f"  Duration: {current_time:.2f} seconds")
        print(f"  File: {output_file}")
    
    print(f"\n✓ Conversion complete! Files saved to {output_path}/")
    print(f"\nActivity files created:")
    for activity in activity_labels.values():
        filepath = f"{output_path}/{activity}.csv"
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / 1024 / 1024  # MB
            print(f"  - {activity}.csv ({file_size:.2f} MB)")

if __name__ == "__main__":
    print("=" * 70)
    print("UCI-HAR Dataset to Framework CSV Converter")
    print("=" * 70)
    convert_to_csv()
    print("\nYou can now upload these CSV files to your framework!")
