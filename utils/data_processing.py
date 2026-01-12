# utils/data_processing.py
import pandas as pd
import base64
import io
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt


def parse_csv(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume UTF-8 encoding
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return None, "Unsupported file format"
    except Exception as e:
        return None, str(e)
    return df, None


def clean_data(df, method):
    # Assume sensor columns: aX, aY, aZ, gX, gY, gZ
    required = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
    if method == 'remove_missing':
        df = df.dropna(subset=required)
    elif method == 'filter_outliers':
        for col in required:
            mean, std = df[col].mean(), df[col].std()
            df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
    return df

# Function to apply a low-pass filter


def low_pass_filter(data, cutoff=5, fs=50, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return pd.DataFrame(filtfilt(b, a, data, axis=0), columns=data.columns)


def split_data(df, split_ratio):
    train_df, test_df = train_test_split(
        df, test_size=1-split_ratio, random_state=42)
    return train_df, test_df
