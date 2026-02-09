"""
Validation Script: Compare Python Model vs Generated C++ Code

This script validates that the generated C++ code produces identical predictions
to the Python model by testing:
1. Feature extraction (Python vs C++ simulation)
2. Scaling/normalization 
3. Model prediction
4. Step-by-step comparison to find discrepancies
"""

from utils.model_training import create_feature_vector
import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class DeploymentValidator:
    """Validate generated C++ code against Python model."""

    def __init__(self, model_path, training_csv_path):
        """
        Initialize validator.

        Args:
            model_path: Path to trained .joblib model
            training_csv_path: Path to training CSV with features
        """
        self.model_path = model_path
        self.training_csv_path = training_csv_path

        # Load model
        print(f"Loading model: {model_path}")
        model_dict = joblib.load(model_path)

        self.model = model_dict['model']
        self.scaler = model_dict['scaler']
        self.label_encoder = model_dict['label_encoder']
        self.feature_names = model_dict.get('feature_names', [])
        self.model_type = model_dict['model_type']

        print(f"✓ Model type: {self.model_type}")
        print(
            f"✓ Features: {len(self.feature_names) if self.feature_names else 'Unknown'}")
        print(f"✓ Classes: {list(self.label_encoder.classes_)}")

        # Load training data
        print(f"\nLoading training data: {training_csv_path}")
        self.training_df = pd.read_csv(training_csv_path)
        print(f"✓ Training samples: {len(self.training_df)}")

    def extract_raw_sensor_window(self, activity_name, window_index=0):
        """
        Extract raw sensor data window from original data files.

        Args:
            activity_name: Activity to extract (e.g., 'still', 'walking', 'running')
            window_index: Which window to extract (0 = first window)

        Returns:
            DataFrame with columns: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        """
        # Look for raw data files
        data_dirs = [
            project_root / 'data' / 'persistent_data-fixed-windows-1500ms',
            project_root / 'persistent_data_new' / 'sensor_data',
            project_root / 'data'
        ]

        for data_dir in data_dirs:
            if not data_dir.exists():
                continue

            # Find activity files
            csv_files = list(data_dir.glob(f'**/{activity_name}*.csv'))
            if csv_files:
                csv_file = csv_files[0]
                print(f"Found raw data: {csv_file}")

                df = pd.read_csv(csv_file)

                # Ensure we have the right columns
                expected_cols = ['acc_x', 'acc_y',
                                 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
                if all(col in df.columns for col in expected_cols):
                    # Return one window worth of data (150 samples @ 100Hz = 1.5s)
                    window_size = 150
                    start_idx = window_index * window_size
                    end_idx = start_idx + window_size

                    if end_idx <= len(df):
                        return df[expected_cols].iloc[start_idx:end_idx].reset_index(drop=True)

        print(f"⚠ Could not find raw sensor data for '{activity_name}'")
        print(f"Will generate synthetic test data instead...")
        return self._generate_synthetic_data(activity_name)

    def _generate_synthetic_data(self, activity_name):
        """Generate realistic synthetic sensor data for testing."""
        np.random.seed(42)
        n_samples = 150  # 1.5s @ 100Hz

        if activity_name == 'still':
            # Stationary: mostly gravity, minimal gyro
            acc_x = np.random.normal(0.0, 0.05, n_samples)
            acc_y = np.random.normal(0.0, 0.05, n_samples)
            acc_z = np.random.normal(9.81, 0.05, n_samples)  # Gravity
            gyro_x = np.random.normal(0.0, 0.5, n_samples)
            gyro_y = np.random.normal(0.0, 0.5, n_samples)
            gyro_z = np.random.normal(0.0, 0.5, n_samples)

        elif activity_name == 'walking':
            # Walking: periodic motion
            t = np.linspace(0, 1.5, n_samples)
            freq = 2.0  # 2 steps per second

            acc_x = np.sin(2 * np.pi * freq * t) * 2.0 + \
                np.random.normal(0, 0.2, n_samples)
            acc_y = np.cos(2 * np.pi * freq * t) * 1.5 + \
                np.random.normal(0, 0.2, n_samples)
            acc_z = 9.81 + np.sin(2 * np.pi * freq * t) * \
                1.0 + np.random.normal(0, 0.2, n_samples)

            gyro_x = np.sin(2 * np.pi * freq * t) * 20.0 + \
                np.random.normal(0, 5, n_samples)
            gyro_y = np.cos(2 * np.pi * freq * t) * 15.0 + \
                np.random.normal(0, 5, n_samples)
            gyro_z = np.random.normal(0, 10, n_samples)

        elif activity_name == 'running':
            # Running: higher amplitude, faster frequency
            t = np.linspace(0, 1.5, n_samples)
            freq = 3.0  # 3 steps per second

            acc_x = np.sin(2 * np.pi * freq * t) * 5.0 + \
                np.random.normal(0, 0.5, n_samples)
            acc_y = np.cos(2 * np.pi * freq * t) * 4.0 + \
                np.random.normal(0, 0.5, n_samples)
            acc_z = 9.81 + np.sin(2 * np.pi * freq * t) * \
                3.0 + np.random.normal(0, 0.5, n_samples)

            gyro_x = np.sin(2 * np.pi * freq * t) * 50.0 + \
                np.random.normal(0, 10, n_samples)
            gyro_y = np.cos(2 * np.pi * freq * t) * 40.0 + \
                np.random.normal(0, 10, n_samples)
            gyro_z = np.random.normal(0, 20, n_samples)
        else:
            raise ValueError(f"Unknown activity: {activity_name}")

        return pd.DataFrame({
            'aX': acc_x,
            'aY': acc_y,
            'aZ': acc_z,
            'gX': gyro_x,
            'gY': gyro_y,
            'gZ': gyro_z
        })

    def simulate_cpp_feature_extraction(self, sensor_df):
        """
        Simulate C++ feature extraction in Python.

        This mimics the exact C++ logic to see if there are differences
        between Python feature extraction and C++ implementation.
        """
        sensor_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
        data = sensor_df[sensor_cols].values
        n_samples = len(data)

        # Calculate magnitudes (exactly as C++ does)
        acc_mag = np.sqrt(data[:, 0]**2 + data[:, 1]**2 + data[:, 2]**2)
        gyro_mag = np.sqrt(data[:, 3]**2 + data[:, 4]**2 + data[:, 5]**2)

        # Calculate jerk magnitude
        acc_diff = np.diff(data[:, :3], axis=0)
        jerk_mag = np.sqrt(np.sum(acc_diff**2, axis=1))

        features = []

        # Extract magnitude stats (15 features per magnitude)
        for mag_name, mag_data in [('acc_mag', acc_mag), ('gyro_mag', gyro_mag)]:
            features.extend(self._extract_magnitude_stats(mag_data))

        # Jerk features (3 features)
        features.extend([
            np.mean(jerk_mag),
            np.std(jerk_mag),
            np.max(jerk_mag)
        ])

        return np.array(features)

    def _extract_magnitude_stats(self, mag):
        """Extract 15 statistical features from magnitude array (C++ style)."""
        stats = [
            np.mean(mag),                      # 0: mean
            np.std(mag),                       # 1: std
            np.min(mag),                       # 2: min
            np.max(mag),                       # 3: max
            np.max(mag) - np.min(mag),         # 4: range
            np.median(mag),                    # 5: median
            np.percentile(mag, 25),            # 6: q25
            np.percentile(mag, 75),            # 7: q75
            np.percentile(mag, 75) - np.percentile(mag, 25),  # 8: iqr
            pd.Series(mag).skew(),             # 9: skewness
            pd.Series(mag).kurtosis(),         # 10: kurtosis
            np.sqrt(np.mean(mag**2)),          # 11: rms
            np.sum(mag**2),                    # 12: energy
            self._zero_crossings(mag),         # 13: zero_crossings
            self._mean_crossing_rate(mag)      # 14: mean_crossing_rate
        ]
        return stats

    def _zero_crossings(self, signal):
        """Count zero crossings."""
        return np.sum(signal[:-1] * signal[1:] < 0)

    def _mean_crossing_rate(self, signal):
        """Calculate mean crossing rate."""
        mean = np.mean(signal)
        crossings = np.sum((signal[:-1] - mean) * (signal[1:] - mean) < 0)
        return crossings / len(signal)

    def validate_window(self, sensor_df, expected_activity=None):
        """
        Validate one window of sensor data through entire pipeline.

        Args:
            sensor_df: DataFrame with raw sensor data
            expected_activity: Expected activity name (optional)

        Returns:
            Dictionary with validation results
        """
        print("\n" + "="*80)
        print("VALIDATION: Single Window Test")
        print("="*80)

        sensor_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']

        # Calculate magnitude for logging
        acc_mag = np.sqrt(sensor_df['aX']**2 +
                          sensor_df['aY']**2 + sensor_df['aZ']**2)
        gyro_mag = np.sqrt(sensor_df['gX']**2 +
                           sensor_df['gY']**2 + sensor_df['gZ']**2)

        print(f"\n📊 Input Sensor Data:")
        print(f"  Samples: {len(sensor_df)}")
        print(
            f"  Acc magnitude:  mean={acc_mag.mean():.2f}, std={acc_mag.std():.2f}, range=[{acc_mag.min():.2f}, {acc_mag.max():.2f}]")
        print(
            f"  Gyro magnitude: mean={gyro_mag.mean():.2f}, std={gyro_mag.std():.2f}, range=[{gyro_mag.min():.2f}, {gyro_mag.max():.2f}]")

        # Step 1: Python feature extraction
        print(f"\n1️⃣ Python Feature Extraction:")
        python_features_df = create_feature_vector(
            sensor_df, sensor_cols, sampling_rate=100,
            include_frequency=False,
            orientation_robust=True,
            include_per_axis=False
        )
        python_features = python_features_df.values.flatten()
        print(f"  ✓ Extracted {len(python_features)} features")
        print(f"  First 5: {python_features[:5]}")
        print(f"  Last 5: {python_features[-5:]}")

        # Step 2: C++ simulated feature extraction
        print(f"\n2️⃣ C++ Simulated Feature Extraction:")
        cpp_features = self.simulate_cpp_feature_extraction(sensor_df)
        print(f"  ✓ Extracted {len(cpp_features)} features")
        print(f"  First 5: {cpp_features[:5]}")
        print(f"  Last 5: {cpp_features[-5:]}")

        # Step 3: Compare feature extraction
        print(f"\n3️⃣ Feature Extraction Comparison:")
        feature_diff = np.abs(python_features - cpp_features)
        max_diff_idx = np.argmax(feature_diff)

        if np.allclose(python_features, cpp_features, rtol=1e-3, atol=1e-6):
            print(
                f"  ✅ Features match (max diff: {feature_diff.max():.6f} at index {max_diff_idx})")
        else:
            print(f"  ❌ Features differ!")
            print(
                f"  Max difference: {feature_diff.max():.6f} at index {max_diff_idx}")
            print(f"    Python: {python_features[max_diff_idx]:.6f}")
            print(f"    C++:    {cpp_features[max_diff_idx]:.6f}")

            # Show top 5 differences
            top_diff_indices = np.argsort(feature_diff)[-5:][::-1]
            print(f"\n  Top 5 differences:")
            feature_names_list = self.feature_names if self.feature_names else []
            for idx in top_diff_indices:
                feat_name = feature_names_list[idx] if idx < len(
                    feature_names_list) else f"Feature_{idx}"
                print(
                    f"    [{idx}] {feat_name}: Python={python_features[idx]:.6f}, C++={cpp_features[idx]:.6f}, diff={feature_diff[idx]:.6f}")

        # Step 4: Scaling
        print(f"\n4️⃣ Feature Scaling:")
        python_scaled = self.scaler.transform([python_features])[0]
        cpp_scaled = self.scaler.transform([cpp_features])[0]

        print(f"  Scaler mean (first 5): {self.scaler.mean_[:5]}")
        print(f"  Scaler scale (first 5): {self.scaler.scale_[:5]}")
        print(f"  Python scaled (first 5): {python_scaled[:5]}")
        print(f"  C++ scaled (first 5): {cpp_scaled[:5]}")

        # Step 5: Model prediction
        print(f"\n5️⃣ Model Prediction:")

        # Python prediction
        python_pred_idx = self.model.predict([python_scaled])[0]
        python_pred_name = self.label_encoder.inverse_transform([python_pred_idx])[
            0]

        # C++ prediction (using C++ features)
        cpp_pred_idx = self.model.predict([cpp_scaled])[0]
        cpp_pred_name = self.label_encoder.inverse_transform([cpp_pred_idx])[0]

        # Get prediction probabilities if available
        if hasattr(self.model, 'predict_proba'):
            python_proba = self.model.predict_proba([python_scaled])[0]
            cpp_proba = self.model.predict_proba([cpp_scaled])[0]

            print(
                f"  Python prediction: {python_pred_name} (class {python_pred_idx})")
            print(
                f"    Probabilities: {dict(zip(self.label_encoder.classes_, python_proba))}")

            print(f"  C++ prediction: {cpp_pred_name} (class {cpp_pred_idx})")
            print(
                f"    Probabilities: {dict(zip(self.label_encoder.classes_, cpp_proba))}")
        else:
            print(
                f"  Python prediction: {python_pred_name} (class {python_pred_idx})")
            print(f"  C++ prediction: {cpp_pred_name} (class {cpp_pred_idx})")

        # Step 6: Final result
        print(f"\n6️⃣ Validation Result:")

        if expected_activity:
            python_correct = python_pred_name == expected_activity
            cpp_correct = cpp_pred_name == expected_activity

            print(f"  Expected: {expected_activity}")
            print(
                f"  Python: {python_pred_name} {'✅' if python_correct else '❌'}")
            print(f"  C++:    {cpp_pred_name} {'✅' if cpp_correct else '❌'}")
        else:
            match = python_pred_name == cpp_pred_name
            print(
                f"  Python vs C++ predictions: {'✅ MATCH' if match else '❌ DIFFER'}")

        return {
            'sensor_data': sensor_df,
            'python_features': python_features,
            'cpp_features': cpp_features,
            'python_scaled': python_scaled,
            'cpp_scaled': cpp_scaled,
            'python_prediction': python_pred_name,
            'cpp_prediction': cpp_pred_name,
            'expected_activity': expected_activity
        }

    def test_training_samples(self, n_samples=10):
        """
        Test random samples from training data.

        This ensures the model can correctly predict its own training data.
        """
        print("\n" + "="*80)
        print("VALIDATION: Training Data Samples")
        print("="*80)

        # Get feature columns (exclude 'label' and 'activity')
        feature_cols = [
            col for col in self.training_df.columns if col not in ['label', 'activity']]

        # Check if we have label column
        has_labels = 'label' in self.training_df.columns

        # Sample random rows
        sample_indices = np.random.choice(len(self.training_df), min(
            n_samples, len(self.training_df)), replace=False)

        results = []
        for idx in sample_indices:
            row = self.training_df.iloc[idx]

            # Get pre-computed features
            features = row[feature_cols].values.astype(float)

            # Get true label (handle both numeric and string labels)
            if has_labels:
                label_val = row['label']
                if isinstance(label_val, (int, np.integer)):
                    true_label = int(label_val)
                elif isinstance(label_val, str):
                    # Convert string label to numeric
                    try:
                        true_label = list(
                            self.label_encoder.classes_).index(label_val)
                    except ValueError:
                        true_label = None
                else:
                    true_label = None
            else:
                true_label = None

            activity_name = row['activity'] if 'activity' in row and pd.notna(
                row['activity']) else f"Sample_{idx}"

            # Scale features
            scaled_features = self.scaler.transform([features])[0]

            # Predict
            pred_idx = self.model.predict([scaled_features])[0]
            pred_name = self.label_encoder.inverse_transform([pred_idx])[0]

            correct = (
                pred_idx == true_label) if true_label is not None else None

            results.append({
                'index': idx,
                'activity': activity_name,
                'true_label': true_label,
                'predicted_label': pred_idx,
                'predicted_name': pred_name,
                'correct': correct
            })

            status = '✅' if correct else '❌' if correct is not None else '?'
            print(
                f"  Sample {idx:4d}: True={activity_name:20s} | Pred={pred_name:20s} {status}")

        # Summary
        if any(r['correct'] is not None for r in results):
            accuracy = np.mean([r['correct']
                               for r in results if r['correct'] is not None])
            print(f"\n📊 Training Sample Accuracy: {accuracy*100:.1f}%")

        return results


def main():
    """Main validation routine."""

    print("="*80)
    print("DEPLOYMENT CODE VALIDATION")
    print("="*80)

    # Paths
    base_dir = Path(r"d:\Workspaces\Master\ComputerScience\Thesis\GUI_app")
    model_path = base_dir / "persistent_data_new" / "models" / \
        "neural_network_har_model_20260204_232342.joblib"
    training_csv = base_dir / "persistent_data_new" / \
        "training" / "running_still_walking_and_2_more_train.csv"

    # Create validator
    validator = DeploymentValidator(model_path, training_csv)

    # Test 1: Validate training samples (Python features are pre-computed correctly)
    print("\n" + "="*80)
    print("TEST 1: Training Data Validation")
    print("="*80)
    validator.test_training_samples(n_samples=20)

    # Test 2: Validate synthetic sensor data (test feature extraction)
    print("\n" + "="*80)
    print("TEST 2: Synthetic Sensor Data")
    print("="*80)

    for activity in ['still', 'walking', 'running']:
        print(f"\n{'='*80}")
        print(f"Testing Activity: {activity.upper()}")
        print('='*80)

        sensor_data = validator.extract_raw_sensor_window(
            activity, window_index=0)
        result = validator.validate_window(
            sensor_data, expected_activity=activity)

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)

    print("\n📋 Next Steps:")
    print("  1. If Python predictions are correct but C++ differs → Feature extraction bug")
    print("  2. If both Python and C++ are wrong → Model training issue")
    print("  3. If training samples are correct → Need real sensor data for validation")
    print("\n  Run this script to diagnose the issue systematically.")


if __name__ == "__main__":
    main()
