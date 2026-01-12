/*
 * Real-Time IMU Data Streaming for Device Testing
 * 
 * This sketch reads data from an LSM6DS3 6-axis IMU sensor and streams it
 * over serial at 100Hz for real-time visualization in the framework.
 * 
 * Hardware: Seeed XIAO nRF52840 Sense (or compatible board with LSM6DS3)
 * Connection: USB (Serial @ 115200 baud)
 * 
 * Data Format: aX,aY,aZ,gX,gY,gZ
 * Units: Accelerometer (m/s²), Gyroscope (deg/s)
 * 
 * Upload this sketch to your device, then use the Device Testing tab
 * in the framework to visualize and classify activities in real-time.
 */

#include <Wire.h>

// LSM6DS3 I2C address
#define LSM6DS3_ADDRESS 0x6A

// LSM6DS3 Register addresses
#define LSM6DS3_WHO_AM_I 0x0F
#define LSM6DS3_CTRL1_XL 0x10  // Accelerometer control
#define LSM6DS3_CTRL2_G  0x11  // Gyroscope control
#define LSM6DS3_OUTX_L_G 0x22  // Gyroscope output registers
#define LSM6DS3_OUTX_L_XL 0x28 // Accelerometer output registers

// Sampling rate: 100Hz
#define SAMPLE_INTERVAL_MS 10

// Conversion factors
#define ACCEL_SENSITIVITY 0.061  // mg/LSB for ±2g range
#define GYRO_SENSITIVITY 8.75    // mdps/LSB for ±245 dps range
#define GRAVITY 9.80665          // m/s²

unsigned long lastSampleTime = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }
  
  Wire.begin();
  delay(100);
  
  // Initialize LSM6DS3
  if (!initLSM6DS3()) {
    Serial.println("ERROR: LSM6DS3 not found!");
    while (1) {
      delay(1000);
    }
  }
  
  Serial.println("# LSM6DS3 initialized successfully");
  Serial.println("# Streaming format: aX,aY,aZ,gX,gY,gZ");
  Serial.println("# Units: m/s², deg/s");
  Serial.println("# Rate: 100Hz");
  delay(1000);
}

void loop() {
  unsigned long currentTime = millis();
  
  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = currentTime;
    
    // Read accelerometer data
    int16_t accelX, accelY, accelZ;
    readAccelerometer(&accelX, &accelY, &accelZ);
    
    // Read gyroscope data
    int16_t gyroX, gyroY, gyroZ;
    readGyroscope(&gyroX, &gyroY, &gyroZ);
    
    // Convert to physical units
    float aX = (accelX * ACCEL_SENSITIVITY / 1000.0) * GRAVITY;
    float aY = (accelY * ACCEL_SENSITIVITY / 1000.0) * GRAVITY;
    float aZ = (accelZ * ACCEL_SENSITIVITY / 1000.0) * GRAVITY;
    
    float gX = gyroX * GYRO_SENSITIVITY / 1000.0;
    float gY = gyroY * GYRO_SENSITIVITY / 1000.0;
    float gZ = gyroZ * GYRO_SENSITIVITY / 1000.0;
    
    // Send data in CSV format
    Serial.print(aX, 3);
    Serial.print(",");
    Serial.print(aY, 3);
    Serial.print(",");
    Serial.print(aZ, 3);
    Serial.print(",");
    Serial.print(gX, 2);
    Serial.print(",");
    Serial.print(gY, 2);
    Serial.print(",");
    Serial.println(gZ, 2);
  }
}

bool initLSM6DS3() {
  // Check WHO_AM_I register (should return 0x69)
  Wire.beginTransmission(LSM6DS3_ADDRESS);
  Wire.write(LSM6DS3_WHO_AM_I);
  Wire.endTransmission();
  Wire.requestFrom(LSM6DS3_ADDRESS, 1);
  
  if (Wire.available()) {
    uint8_t whoAmI = Wire.read();
    if (whoAmI != 0x69) {
      return false;
    }
  } else {
    return false;
  }
  
  // Configure accelerometer: 100Hz, ±2g
  writeRegister(LSM6DS3_CTRL1_XL, 0x50);
  
  // Configure gyroscope: 100Hz, ±245 dps
  writeRegister(LSM6DS3_CTRL2_G, 0x50);
  
  delay(100);
  return true;
}

void writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(LSM6DS3_ADDRESS);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

void readAccelerometer(int16_t* x, int16_t* y, int16_t* z) {
  Wire.beginTransmission(LSM6DS3_ADDRESS);
  Wire.write(LSM6DS3_OUTX_L_XL);
  Wire.endTransmission();
  Wire.requestFrom(LSM6DS3_ADDRESS, 6);
  
  if (Wire.available() >= 6) {
    uint8_t xLow = Wire.read();
    uint8_t xHigh = Wire.read();
    uint8_t yLow = Wire.read();
    uint8_t yHigh = Wire.read();
    uint8_t zLow = Wire.read();
    uint8_t zHigh = Wire.read();
    
    *x = (int16_t)(xHigh << 8 | xLow);
    *y = (int16_t)(yHigh << 8 | yLow);
    *z = (int16_t)(zHigh << 8 | zLow);
  }
}

void readGyroscope(int16_t* x, int16_t* y, int16_t* z) {
  Wire.beginTransmission(LSM6DS3_ADDRESS);
  Wire.write(LSM6DS3_OUTX_L_G);
  Wire.endTransmission();
  Wire.requestFrom(LSM6DS3_ADDRESS, 6);
  
  if (Wire.available() >= 6) {
    uint8_t xLow = Wire.read();
    uint8_t xHigh = Wire.read();
    uint8_t yLow = Wire.read();
    uint8_t yHigh = Wire.read();
    uint8_t zLow = Wire.read();
    uint8_t zHigh = Wire.read();
    
    *x = (int16_t)(xHigh << 8 | xLow);
    *y = (int16_t)(yHigh << 8 | yLow);
    *z = (int16_t)(zHigh << 8 | zLow);
  }
}
