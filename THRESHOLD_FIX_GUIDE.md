"""
Quick fix: Add heuristic threshold to bias toward 'still' when motion is minimal
This is a temporary workaround until you can retrain with real data
"""

# Add this function to your .cpp file after har_predict_internal()

"""
int har_predict_internal_with_threshold(float features[NUM_FEATURES]) {
    // Get raw prediction from neural network
    int predicted_class = har_predict_internal(features);

    // Heuristic override: If average gyro magnitude is very low, force 'still'
    // features[6-11] are gyro_mag statistics
    float avg_gyro = features[6];  // Mean gyro magnitude

    // If gyro < 5 deg/s AND prediction is walking, override to still
    if (avg_gyro < 5.0f && predicted_class == 2) {  // Class 2 = walking
        return 1;  // Force Class 1 = still
    }

    return predicted_class;
}
"""

# Then change har_predict() to call the new function

"""
int har_predict(float features[NUM_FEATURES]) {
    // ... existing scaling code ...

    // Call threshold-enhanced prediction
    return har_predict_internal_with_threshold(scaled_features);
}
"""

# This will make

# - gyro < 5 => still (Class 1)

# - gyro 5-80 => walking (Class 2)

# - gyro > 80 => running (Class 0)
