/**
 * @file imu.ino
 * @author SeanKwok (shaoxiang@m5stack.com)
 * @brief M5StickCPlus2 get IMU data
 * @version 0.1
 * @date 2023-12-19
 *
 *
 * @Hardwares: M5StickCPlus2
 * @Platform Version: Arduino M5Stack Board Manager v2.0.9
 * @Dependent Library:
 * M5GFX: https://github.com/m5stack/M5GFX
 * M5Unified: https://github.com/m5stack/M5Unified
 * M5StickCPlus2: https://github.com/m5stack/M5StickCPlus2
 */

#include "M5StickCPlus2.h"
#define CONVERT_G_TO_MS2 9.80665f
#define FREQUENCY_HZ 100
#define INTERVAL_MS (1000 / (FREQUENCY_HZ + 1))

static unsigned long last_interval_ms = 0;

void setup() {
    auto cfg = M5.config();
    StickCP2.begin(cfg);
    StickCP2.Display.setRotation(1);
    StickCP2.Display.setTextColor(GREEN);
    StickCP2.Display.setTextDatum(middle_center);
    StickCP2.Display.setFont(&fonts::FreeSansBold9pt7b);
    StickCP2.Display.setTextSize(1);
    StickCP2.Display.setCursor(0, 40);
    StickCP2.Display.printf("Getting IMU from Serial\r\n");
    StickCP2.Display.printf("port with frequency of\r\n");
    StickCP2.Display.printf("%d Hz\r\n", FREQUENCY_HZ);
    Serial.begin(115200);
    while (!Serial);
}

void loop(void) {
    if (millis() > last_interval_ms + INTERVAL_MS) {
    if (StickCP2.Imu.update()) {
        // StickCP2.Display.setCursor(0, 40);
        // StickCP2.Display.clear();  // Delay 100ms 延迟100ms

        auto data = StickCP2.Imu.getImuData();

        // The data obtained by getImuData can be used as follows.
        // data.accel.x;      // accel x-axis value.
        // data.accel.y;      // accel y-axis value.
        // data.accel.z;      // accel z-axis value.
        // data.accel.value;  // accel 3values array [0]=x / [1]=y / [2]=z.

        // data.gyro.x;      // gyro x-axis value.
        // data.gyro.y;      // gyro y-axis value.
        // data.gyro.z;      // gyro z-axis value.
        // data.gyro.value;  // gyro 3values array [0]=x / [1]=y / [2]=z.

        // data.value;  // all sensor 9values array [0~2]=accel / [3~5]=gyro /
        //              // [6~8]=mag

        Serial.printf("%f,%f,%f,", data.accel.x*CONVERT_G_TO_MS2, data.accel.y*CONVERT_G_TO_MS2,
                      data.accel.z*CONVERT_G_TO_MS2);
        Serial.printf("%f,%f,%f\r\n", data.gyro.x, data.gyro.y,
                      data.gyro.z);

        // StickCP2.Display.printf("IMU:\r\n");
        // StickCP2.Display.printf("%0.2f %0.2f %0.2f\r\n", data.accel.x*CONVERT_G_TO_MS2,
        //                         data.accel.y*CONVERT_G_TO_MS2, data.accel.z*CONVERT_G_TO_MS2);
        // StickCP2.Display.printf("%0.2f %0.2f %0.2f\r\n", data.gyro.x,
        //                         data.gyro.y, data.gyro.z);
    }}
}