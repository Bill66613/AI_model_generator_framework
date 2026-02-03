// Arduino/ESP32 Feature Parity Test
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
    
    Serial.println("\n========================================");
    Serial.println("C++ FEATURE EXTRACTION TEST");
    Serial.println("========================================\n");
    
    // Extract features using C++ implementation
    extract_features(sensor_data, num_samples, features);
    
    // Compare with Python
    Serial.println("Feature Comparison (Python vs C++):\n");
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
    
    Serial.println("\n========================================");
    Serial.println("TEST RESULTS");
    Serial.println("========================================");
    Serial.print("Max difference:  "); Serial.println(max_diff, 6);
    Serial.print("Mean difference: "); Serial.println(sum_diff / 33.0f, 6);
    Serial.print("Errors (>0.01):  "); Serial.println(errors);
    
    if (errors == 0 && max_diff < 0.01) {
        Serial.println("\n✅ PASS: C++ matches Python!");
    } else {
        Serial.println("\n❌ FAIL: C++ differs from Python");
    }
}

void loop() {
    // Nothing
}

// TODO: Add your extract_features() and extract_magnitude_stats() functions here
// Copy from generated deployment code
