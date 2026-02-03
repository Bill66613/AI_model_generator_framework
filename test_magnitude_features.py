"""
Feature Parity Validation Test
Tests that Python and C++ magnitude feature extraction produce identical results
"""

import numpy as np
import pandas as pd
from utils.model_training import extract_orientation_invariant_features

def generate_test_data(samples=100, seed=42):
    """Generate synthetic sensor data for testing."""
    np.random.seed(seed)
    
    # Simulate walking activity
    t = np.linspace(0, 2, samples)  # 2 seconds at 50Hz
    
    # Accelerometer (walking pattern: ~2 Hz oscillation + gravity)
    aX = 0.5 * np.sin(2 * np.pi * 2 * t) + np.random.normal(0, 0.1, samples)
    aY = -0.3 * np.cos(2 * np.pi * 2 * t) + np.random.normal(0, 0.1, samples)
    aZ = 9.8 + 0.2 * np.sin(2 * np.pi * 2 * t) + np.random.normal(0, 0.1, samples)
    
    # Gyroscope (small rotation during walking)
    gX = 0.1 * np.sin(2 * np.pi * 2 * t) + np.random.normal(0, 0.05, samples)
    gY = 0.05 * np.cos(2 * np.pi * 2 * t) + np.random.normal(0, 0.05, samples)
    gZ = 0.02 * np.sin(2 * np.pi * 4 * t) + np.random.normal(0, 0.02, samples)
    
    df = pd.DataFrame({
        'aX': aX,
        'aY': aY,
        'aZ': aZ,
        'gX': gX,
        'gY': gY,
        'gZ': gZ
    })
    
    return df

def export_test_data_for_cpp(df, filename='test_data_for_cpp.txt'):
    """Export test data in format easy to copy into C++ code."""
    with open(filename, 'w') as f:
        f.write(f"// Test data: {len(df)} samples\n")
        f.write("float sensor_data[][6] = {\n")
        for idx, row in df.iterrows():
            f.write(f"    {{{row['aX']:.6f}f, {row['aY']:.6f}f, {row['aZ']:.6f}f, "
                   f"{row['gX']:.6f}f, {row['gY']:.6f}f, {row['gZ']:.6f}f}}")
            if idx < len(df) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")
        f.write(f"int num_samples = {len(df)};\n")
    print(f"Test data exported to {filename}")

def test_magnitude_features():
    """Test Python magnitude feature extraction."""
    print("=" * 80)
    print("MAGNITUDE FEATURE VALIDATION TEST")
    print("=" * 80)
    
    # Generate test data
    print("\n1. Generating test data...")
    df = generate_test_data(samples=100)
    print(f"   Generated {len(df)} samples")
    print(f"   Sample data (first 3 rows):")
    print(df.head(3))
    
    # Extract features
    print("\n2. Extracting Python magnitude features...")
    sensor_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
    features_df = extract_orientation_invariant_features(df, sensor_cols)
    
    # Convert DataFrame to numpy array
    features = features_df.values.flatten()
    
    print(f"   Extracted {len(features)} features")
    print(f"\n   Feature breakdown:")
    print(f"   - acc_mag features (0-14):")
    for i in range(15):
        print(f"      [{i}] {features_df.columns[i]:30s} = {features[i]:.6f}")
    print(f"\n   - gyro_mag features (15-29):")
    for i in range(15, 30):
        print(f"      [{i}] {features_df.columns[i]:30s} = {features[i]:.6f}")
    print(f"\n   - jerk_mag features (30-32):")
    for i in range(30, 33):
        print(f"      [{i}] {features_df.columns[i]:30s} = {features[i]:.6f}")
    
    # Export for C++ testing
    print("\n3. Exporting data for C++ validation...")
    export_test_data_for_cpp(df, 'test_data_for_cpp.txt')
    
    # Export expected features
    with open('expected_features_python.txt', 'w') as f:
        f.write("// Expected features from Python (33 total)\n")
        f.write("float expected_features[33] = {\n")
        for i, feat in enumerate(features):
            f.write(f"    {feat:.6f}f")
            if i < len(features) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n")
    print("   Expected features exported to expected_features_python.txt")
    
    # Test orientation invariance
    print("\n4. Testing orientation invariance...")
    
    # Rotate data 90 degrees (swap axes)
    df_rotated = df.copy()
    df_rotated['aX'] = -df['aY']
    df_rotated['aY'] = df['aX']
    # aZ stays same (rotation around Z axis)
    
    features_rotated_df = extract_orientation_invariant_features(df_rotated, sensor_cols)
    features_rotated = features_rotated_df.values.flatten()
    
    print(f"   Original acc_mag_mean:  {features[0]:.6f}")
    print(f"   Rotated acc_mag_mean:   {features_rotated[0]:.6f}")
    print(f"   Difference:             {abs(features[0] - features_rotated[0]):.6f}")
    
    # Check if features are similar (should be nearly identical)
    max_diff = np.max(np.abs(features - features_rotated))
    mean_diff = np.mean(np.abs(features - features_rotated))
    
    print(f"\n   Feature comparison (original vs rotated):")
    print(f"   - Max difference:  {max_diff:.6f}")
    print(f"   - Mean difference: {mean_diff:.6f}")
    
    if max_diff < 0.1:
        print(f"   ✅ PASS: Features are orientation-invariant! (max diff < 0.1)")
    else:
        print(f"   ❌ FAIL: Features changed with rotation (max diff = {max_diff:.6f})")
    
    # Generate C++ test code
    print("\n5. Generating C++ test skeleton...")
    with open('cpp_test_skeleton.ino', 'w', encoding='utf-8') as f:
        f.write("""// Arduino/ESP32 Feature Parity Test
// Compare C++ feature extraction with Python results

#include <Arduino.h>

// Copy from test_data_for_cpp.txt
// float sensor_data[][6] = {...};
// int num_samples = 100;

// Copy from expected_features_python.txt
// float expected_features[33] = {...};

float features[33];

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\\n========================================");
    Serial.println("C++ FEATURE EXTRACTION TEST");
    Serial.println("========================================\\n");
    
    // Extract features using C++ implementation
    extract_features(sensor_data, num_samples, features);
    
    // Compare with Python
    Serial.println("Feature Comparison (Python vs C++):\\n");
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    int errors = 0;
    
    for (int i = 0; i < 33; i++) {
        float diff = abs(features[i] - expected_features[i]);
        sum_diff += diff;
        if (diff > max_diff) max_diff = diff;
        
        Serial.print("Feature ");
        Serial.print(i);
        Serial.print(": Python=");
        Serial.print(expected_features[i], 6);
        Serial.print(", C++=");
        Serial.print(features[i], 6);
        Serial.print(", Diff=");
        Serial.println(diff, 6);
        
        if (diff > 0.01) {  // Tolerance
            errors++;
        }
    }
    
    Serial.println("\\n========================================");
    Serial.println("TEST RESULTS");
    Serial.println("========================================");
    Serial.print("Max difference:  "); Serial.println(max_diff, 6);
    Serial.print("Mean difference: "); Serial.println(sum_diff / 33.0f, 6);
    Serial.print("Errors (>0.01):  "); Serial.println(errors);
    
    if (errors == 0 && max_diff < 0.01) {
        Serial.println("\\n✅ PASS: C++ matches Python!");
    } else {
        Serial.println("\\n❌ FAIL: C++ differs from Python");
    }
}

void loop() {
    // Nothing
}

// TODO: Add your extract_features() and extract_magnitude_stats() functions here
// Copy from generated deployment code
""")
    print("   C++ test skeleton saved to cpp_test_skeleton.ino")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Copy test_data_for_cpp.txt into cpp_test_skeleton.ino")
    print("2. Copy expected_features_python.txt into cpp_test_skeleton.ino")
    print("3. Add extract_features() and extract_magnitude_stats() from base_generator.py")
    print("4. Upload to Arduino/ESP32 and check Serial Monitor")
    print("5. Verify all features match within ±0.01 tolerance")
    print("\n")

if __name__ == "__main__":
    test_magnitude_features()
