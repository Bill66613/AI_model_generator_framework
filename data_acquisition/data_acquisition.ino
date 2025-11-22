/**
 * @file data.ino
 * @brief Data acquisition
 */

#include <LSM6DS3.h>
// #include <ArduinoBLE.h>
#include <Wire.h>

//Create a instance of class LSM6DS3
LSM6DS3 myIMU(I2C_MODE, 0x6A);  //I2C device address 0x6A

/* Constant defines -------------------------------------------------------- */
#define CONVERT_G_TO_MS2 9.80665f
#define MAX_ACCEPTED_RANGE \
  2.0f  // starting 03/2022, models are generated setting range to +-2, but this example use Arudino library which set range to +-4g. If you are using an older model, ignore this value and use 4.0f instead

#define FREQUENCY_HZ 100
#define INTERVAL_MS (1000 / (FREQUENCY_HZ + 1))

static unsigned long last_interval_ms = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial)
    delay(10);

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

  Serial.println("Collecting sensor data...");
  Serial.println("aX,aY,aZ,gX,gY,gZ,Time_seconds");
}

void loop() {
  if (millis() > last_interval_ms + INTERVAL_MS) {
    last_interval_ms = millis();

    // Read and output sensor data
    // Units: Accelerometer in m/s², Gyroscope in deg/s
    Serial.print(myIMU.readFloatAccelX() * CONVERT_G_TO_MS2, 4);  // aX (m/s²)
    Serial.print(',');
    Serial.print(myIMU.readFloatAccelY() * CONVERT_G_TO_MS2, 4);  // aY (m/s²)
    Serial.print(',');
    Serial.print(myIMU.readFloatAccelZ() * CONVERT_G_TO_MS2, 4);  // aZ (m/s²)
    Serial.print(',');
    Serial.print(myIMU.readFloatGyroX(), 4);  // gX (deg/s)
    Serial.print(',');
    Serial.print(myIMU.readFloatGyroY(), 4);  // gY (deg/s)
    Serial.print(',');
    Serial.print(myIMU.readFloatGyroZ(), 4);  // gZ (deg/s)
    Serial.print(',');
    Serial.println(last_interval_ms / 1000.0, 4);  // Time in seconds
  }
}