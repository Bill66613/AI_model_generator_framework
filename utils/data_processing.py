# utils/data_processing.py
import pandas as pd
import base64
import io
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt

from config.config import SENSOR_COLUMNS, DEFAULT_SAMPLING_RATE


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


def detect_sensor_columns(df, default=None):
    """Detect which sensor columns are present in a DataFrame.

    Returns the intersection of the DataFrame's columns with the expected
    sensor columns (from *default* or the global ``SENSOR_COLUMNS``).
    """
    expected = default or SENSOR_COLUMNS
    return [c for c in expected if c in df.columns]


def clean_data(df, method, sensor_cols=None):
    """Clean data using the specified method.

    Args:
        df: Input DataFrame.
        method: ``'remove_missing'`` or ``'filter_outliers'``.
        sensor_cols: Columns to operate on.  Defaults to auto-detected
                     sensor columns present in *df*.
    """
    if sensor_cols is None:
        sensor_cols = detect_sensor_columns(df)
    if method == 'remove_missing':
        df = df.dropna(subset=sensor_cols)
    elif method == 'filter_outliers':
        for col in sensor_cols:
            mean, std = df[col].mean(), df[col].std()
            df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
    return df


def low_pass_filter(data, cutoff=5, fs=None, order=2):
    """Apply a Butterworth low-pass filter.

    Args:
        data: DataFrame of numeric columns.
        cutoff: Cutoff frequency in Hz.
        fs: Sampling frequency in Hz.  Defaults to ``DEFAULT_SAMPLING_RATE``.
        order: Filter order.
    """
    if fs is None:
        fs = DEFAULT_SAMPLING_RATE
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return pd.DataFrame(filtfilt(b, a, data, axis=0), columns=data.columns)


def kalman_filter(data, process_noise=1e-3, measurement_noise=1e-1, fs=None):
    """Apply a 1D Kalman filter independently to each sensor channel.

    Uses a constant-velocity state model [position, velocity] per channel.
    The filter is causal (forward-only), so training and deployment produce
    identical results — no parity gap unlike filtfilt-based IIR.

    Args:
        data: DataFrame of numeric columns.
        process_noise: Process noise covariance (Q diagonal).  Smaller = smoother.
        measurement_noise: Measurement noise covariance (R).  Larger = smoother.
        fs: Sampling frequency in Hz.  Defaults to ``DEFAULT_SAMPLING_RATE``.

    Returns:
        DataFrame with Kalman-filtered values, same shape as input.
    """
    if fs is None:
        fs = DEFAULT_SAMPLING_RATE
    import numpy as np

    dt = 1.0 / fs

    # State transition matrix: [pos, vel] -> [pos + vel*dt, vel]
    F = np.array([[1, dt],
                  [0, 1]], dtype=np.float64)
    # Measurement matrix: we observe position only
    H = np.array([[1, 0]], dtype=np.float64)
    # Process noise covariance
    Q = process_noise * np.array([[dt**3 / 3, dt**2 / 2],
                                  [dt**2 / 2, dt]], dtype=np.float64)
    # Measurement noise covariance
    R = np.array([[measurement_noise]], dtype=np.float64)

    result = pd.DataFrame(index=data.index, columns=data.columns, dtype=np.float64)

    for col in data.columns:
        z = data[col].values.astype(np.float64)
        n = len(z)

        # Initialize state with first measurement
        x = np.array([z[0], 0.0], dtype=np.float64)
        P = np.eye(2, dtype=np.float64) * measurement_noise

        filtered = np.empty(n, dtype=np.float64)

        for k in range(n):
            # Predict
            x = F @ x
            P = F @ P @ F.T + Q

            # Update
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            y = z[k] - (H @ x)[0]
            x = x + (K @ np.array([[y]])).flatten()
            P = (np.eye(2) - K @ H) @ P

            filtered[k] = x[0]

        result[col] = filtered

    return result


def split_data(df, split_ratio):
    train_df, test_df = train_test_split(
        df, test_size=1-split_ratio, random_state=42)
    return train_df, test_df
