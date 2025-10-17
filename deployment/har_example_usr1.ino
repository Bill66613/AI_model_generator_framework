/*
 * HAR Model Example Sketch
 * Demonstrates usage of the generated HAR model
 */

#include "har_model.h"
#include <LSM6DS3.h>
#include <ArduinoBLE.h>
#include <Wire.h>

/* Constant defines -------------------------------------------------------- */
#define CONVERT_G_TO_MS2 9.80665f
#define MAX_ACCEPTED_RANGE 2.0f // starting 03/2022, models are generated setting range to +-2, but this example use Arudino library which set range to +-4g. If you are using an older model, ignore this value and use 4.0f instead

// IMU sensor pins (adjust for your hardware)
#define IMU_SDA_PIN A4
#define IMU_SCL_PIN A5

/* Private variables ------------------------------------------------------- */
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
LSM6DS3 myIMU(I2C_MODE, 0x6A);
BLEService predictService("180D"); // Custom IMU service UUID
BLEByteCharacteristic predictChar("2A37", BLERead | BLENotify);

// Data collection variables
float sensor_buffer[WINDOW_SIZE][6];
int buffer_index = 0;
unsigned long last_reading = 0;
const unsigned long READING_INTERVAL = 1000 / SAMPLING_RATE; // milliseconds

void setup()
{
    Serial.begin(9600);
    while (!Serial)
        delay(10);

    Serial.println("HAR Model Demo");
    Serial.println("==============");

    // Initialize HAR model
    har_init();

    // Initialize IMU sensor (placeholder)
    // Add your IMU initialization code here
    Serial.println("IMU sensor initialized");

    Serial.println("Starting activity recognition...");
}

void loop()
{
    unsigned long current_time = millis();

    // Check if it's time for a new reading
    if (current_time - last_reading >= READING_INTERVAL)
    {
        last_reading = current_time;

        // Read IMU data (placeholder - replace with actual sensor reading)
        // float aX = random(-20, 20) / 10.0; // Simulated accelerometer
        // float aY = random(-20, 20) / 10.0;
        // float aZ = random(80, 120) / 10.0;
        // float gX = random(-50, 50) / 100.0; // Simulated gyroscope
        // float gY = random(-50, 50) / 100.0;
        // float gZ = random(-50, 50) / 100.0;

        float aX = myIMU.readFloatAccelX() * CONVERT_G_TO_MS2;
        float aY = myIMU.readFloatAccelY() * CONVERT_G_TO_MS2;
        float aZ = myIMU.readFloatAccelZ() * CONVERT_G_TO_MS2;
        float gX = myIMU.readFloatGyroX();
        float gY = myIMU.readFloatGyroY();
        float gZ = myIMU.readFloatGyroZ();

        // Store in buffer
        sensor_buffer[buffer_index][0] = aX;
        sensor_buffer[buffer_index][1] = aY;
        sensor_buffer[buffer_index][2] = aZ;
        sensor_buffer[buffer_index][3] = gX;
        sensor_buffer[buffer_index][4] = gY;
        sensor_buffer[buffer_index][5] = gZ;

        buffer_index++;

        // When buffer is full, make prediction
        if (buffer_index >= WINDOW_SIZE)
        {
            float features[NUM_FEATURES];

            // Extract features from the collected data
            extract_features(sensor_buffer, WINDOW_SIZE, features);

            // Make prediction
            int predicted_class = har_predict(features);
            const char *activity = get_activity_name(predicted_class);

            // Display result
            Serial.print("Predicted Activity: ");
            Serial.println(activity);

            // Reset buffer
            buffer_index = 0;
        }
    }

    // Small delay to prevent overwhelming the serial monitor
    delay(10);
}