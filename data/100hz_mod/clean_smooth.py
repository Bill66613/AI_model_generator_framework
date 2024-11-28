import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Load the dataset
filename = 'walking_upstairs'
file_path = f'{filename}.csv'  # Replace with the correct file path
data = pd.read_csv(file_path)

# Specify the IMU columns
imu_columns = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']

# Function to visualize raw data
def visualize_data(data, title="IMU Data"):
    plt.figure(figsize=(16, 8))
    for i, col in enumerate(data.columns):
        plt.subplot(2, 3, i + 1)
        plt.plot(data[col], label=col, linewidth=0.8)
        plt.title(f"{col} - {title}")
        plt.xlabel("Samples")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
    plt.tight_layout()
    plt.show()

# Visualize raw data
visualize_data(data[imu_columns], title="Raw Data")

# Detect and remove outliers using Z-Score
def remove_outliers(data, threshold=3):
    z_scores = np.abs((data - data.mean()) / data.std())
    return data[(z_scores < threshold).all(axis=1)]

# Apply outlier removal
data_cleaned = remove_outliers(data[imu_columns])

# Function to apply a low-pass filter
def low_pass_filter(data, cutoff=5, fs=50, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return pd.DataFrame(filtfilt(b, a, data, axis=0), columns=data.columns)

# Apply low-pass filter
data_smoothed = low_pass_filter(data_cleaned, cutoff=5, fs=50)

# Visualize cleaned and smoothed data
visualize_data(data_cleaned, title="Cleaned Data")
visualize_data(data_smoothed, title="Smoothed Data")

# Save cleaned and smoothed data
output_cleaned_path = f'{filename}_cleaned.csv'
output_smoothed_path = f'{filename}_smoothed.csv'
data_cleaned.to_csv(output_cleaned_path, index=False)
data_smoothed.to_csv(output_smoothed_path, index=False)

print(f"Cleaned data saved to {output_cleaned_path}")
print(f"Smoothed data saved to {output_smoothed_path}")
