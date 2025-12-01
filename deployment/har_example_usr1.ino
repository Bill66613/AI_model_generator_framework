#include "har_neural_network_seeed_xiao_f138_c5_balanced.h"

#include <LSM6DS3.h>
#include <Wire.h>

//Create a instance of class LSM6DS3
LSM6DS3 myIMU(I2C_MODE, 0x6A);  //I2C device address 0x6A

/* Constant defines -------------------------------------------------------- */
#define CONVERT_G_TO_MS2 9.80665f
#define MAX_ACCEPTED_RANGE \
  2.0f  // starting 03/2022, models are generated setting range to +-2, but this example use Arudino library which set range to +-4g. If you are using an older model, ignore this value and use 4.0f instead

// Data collection variables
float sensor_buffer[WINDOW_SIZE][6];  // aX, aY, aZ, gX, gY, gZ
int buffer_index = 0;
float features[NUM_FEATURES];
unsigned long last_reading = 0;
const unsigned long READING_INTERVAL = 10;  // ms between readings

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

  // Initialize I2C bus first
  Wire.begin();
  delay(100);  // Give I2C time to stabilize

  // Initialize IMU sensor
  Serial.println("Initializing IMU sensor...");

  // LSM6DS3 begin() returns 0 on SUCCESS, non-zero on failure
  if (myIMU.begin() != 0) {
    Serial.println("❌ Failed to initialize IMU!");
    Serial.println("Trying alternate I2C address 0x6B...");

    // Try alternate address
    LSM6DS3 myIMU_alt(I2C_MODE, 0x6B);
    if (myIMU_alt.begin() != 0) {
      Serial.println("❌ IMU not found at 0x6A or 0x6B");
      Serial.println("Check I2C connections and power");
      while (1) {
        delay(100);  // Halt - cannot continue without IMU
      }
    } else {
      Serial.println("✅ IMU found at address 0x6B!");
      // Note: You'll need to update the global myIMU object address
    }
  } else {
    Serial.println("✅ IMU initialized successfully at 0x6A!");
  }

  // Print IMU settings
  Serial.print("Accelerometer range: ±");
  Serial.print(MAX_ACCEPTED_RANGE);
  Serial.println("g");

  Serial.println("HAR Model Ready!");
  Serial.println("Collecting sensor data...");
}

void loop() {
  unsigned long current_time = millis();

  // Check if it's time for a new reading
  if (current_time - last_reading >= READING_INTERVAL) {
    last_reading = current_time;

    // Read sensor data from IMU
    float aX = myIMU.readFloatAccelX() * CONVERT_G_TO_MS2;
    float aY = myIMU.readFloatAccelY() * CONVERT_G_TO_MS2;
    float aZ = myIMU.readFloatAccelZ() * CONVERT_G_TO_MS2;
    float gX = myIMU.readFloatGyroX();
    float gY = myIMU.readFloatGyroY();
    float gZ = myIMU.readFloatGyroZ();

    // Store in buffer (convert accelerometer from g to m/s²)
    sensor_buffer[buffer_index][0] = aX;
    sensor_buffer[buffer_index][1] = aY;
    sensor_buffer[buffer_index][2] = aZ;
    sensor_buffer[buffer_index][3] = gX;
    sensor_buffer[buffer_index][4] = gY;
    sensor_buffer[buffer_index][5] = gZ;

    buffer_index++;

    // When buffer is full, extract features and predict
    if (buffer_index >= WINDOW_SIZE) {
      buffer_index = 0;

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
    }
  }

  delay(10);  // Optimization-specific delay
}