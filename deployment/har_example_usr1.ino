/*
 * HAR Model Example Sketch
 * Demonstrates usage of the generated HAR model
 * Model Type: neural_network
 * Optimization: BALANCED
 * Balanced optimization - good trade-off between accuracy, speed, and power
 */

#include "har_neural_network_seeed_xiao_f138_c4_balanced.h"

#include <LSM6DS3.h>
// #include <ArduinoBLE.h>
#include <Wire.h>

// IMU sensor pins (adjust for your hardware)
#define IMU_SDA_PIN A4
#define IMU_SCL_PIN A5

// Create a instance of class LSM6DS3
LSM6DS3 myIMU(I2C_MODE, 0x6A);  // I2C device address 0x6A

/* Constant defines -------------------------------------------------------- */
#define CONVERT_G_TO_MS2 9.80665f
#define MAX_ACCEPTED_RANGE \
  2.0f  // starting 03/2022, models are generated setting range to +-2, but this example use Arudino library which set range to +-4g. If you are using an older model, ignore this value and use 4.0f instead

// Data collection variables
float sensor_buffer[WINDOW_SIZE][6];  // aX, aY, aZ, gX, gY, gZ
int buffer_index = 0;
float features[NUM_FEATURES];
unsigned long last_reading = 0;
const unsigned long READING_INTERVAL = 13;  // ms between readings

void setup() {
  Serial.begin(115200);
  while (!Serial)
    delay(10);

  // Print optimization info
  Serial.println("HAR Model - Balanced Optimization");
  Serial.println("========================================");
  Serial.println("Configuration:");
  Serial.print("  Sampling Rate: ");
  Serial.print(SAMPLING_RATE);
  Serial.println(" Hz");
  Serial.print("  Window Size: ");
  Serial.print(WINDOW_SIZE);
  Serial.println(" samples");
  Serial.print("  Features: ");
  Serial.println(NUM_FEATURES);
  Serial.print("  Optimization: ");
  Serial.println("Balanced");

  // Initialize HAR model
  har_init();

  // Initialize IMU (example - adapt for your sensor)
  Serial.println("Initializing IMU sensor...");
  // Your IMU initialization code here
  if (!myIMU.begin()) {
    Serial.println("Failed to initialize IMU!\r\n");
  } else {
    Serial.println("IMU initialized\r\n");
  }

  Serial.println("HAR Model Ready!");
  Serial.println("Collecting sensor data...");
}

void loop() {
  unsigned long current_time = millis();

  // Serial.print("Last time = ");
  // Serial.println(last_reading);
  // Serial.print("Current time = ");
  // Serial.println(current_time);

  // Check if it's time for a new reading
  if (current_time - last_reading >= READING_INTERVAL) {
    last_reading = current_time;

    // Read sensor data (example - replace with your IMU reading code)
    float aX = myIMU.readFloatAccelX();
    float aY = myIMU.readFloatAccelY();
    float aZ = myIMU.readFloatAccelZ();
    float gX = myIMU.readFloatGyroX();
    float gY = myIMU.readFloatGyroY();
    float gZ = myIMU.readFloatGyroZ();

    // Serial.print("aX: ");
    // Serial.println(aX);
    // Serial.print("aY: ");
    // Serial.println(aY);
    // Serial.print("aZ: ");
    // Serial.println(aZ);
    // Serial.print("gX: ");
    // Serial.println(gX);
    // Serial.print("gY: ");
    // Serial.println(gY);
    // Serial.print("gZ: ");
    // Serial.println(gZ);

    // Store in buffer
    sensor_buffer[buffer_index][0] = aX * CONVERT_G_TO_MS2;
    sensor_buffer[buffer_index][1] = aY * CONVERT_G_TO_MS2;
    sensor_buffer[buffer_index][2] = aZ * CONVERT_G_TO_MS2;
    sensor_buffer[buffer_index][3] = gX;
    sensor_buffer[buffer_index][4] = gY;
    sensor_buffer[buffer_index][5] = gZ;

    Serial.println("Sensor Data:");
    Serial.print("aX: ");
    Serial.println(sensor_buffer[buffer_index][0]);
    Serial.print("aY: ");
    Serial.println(sensor_buffer[buffer_index][1]);
    Serial.print("aZ: ");
    Serial.println(sensor_buffer[buffer_index][2]);
    Serial.print("gX: ");
    Serial.println(sensor_buffer[buffer_index][3]);
    Serial.print("gY: ");
    Serial.println(sensor_buffer[buffer_index][4]);
    Serial.print("gZ: ");
    Serial.println(sensor_buffer[buffer_index][5]);

    // Serial.print("Buffer Index: ");
    // Serial.println(buffer_index);

    buffer_index++;

    // When buffer is full, extract features and predict
    if (buffer_index >= WINDOW_SIZE) {
      buffer_index = 0;

      // Serial.println("Buffer full. Extracting features and predicting...");

      // Extract features
      extract_features(sensor_buffer, WINDOW_SIZE, features);

      // Make prediction
      int predicted_class = har_predict(features);
      const char* activity_name = get_activity_name(predicted_class);

      // Print result
      Serial.print("Predicted Activity: ");
      Serial.print(activity_name);
      Serial.print(" (Class ");
      Serial.print(predicted_class);
      Serial.println(")");
    } else {
      // Serial.println("Collecting still ...");
    }
  } else {
    // Serial.println("Dead");
  }

  delay(10);  // Optimization-specific delay
}