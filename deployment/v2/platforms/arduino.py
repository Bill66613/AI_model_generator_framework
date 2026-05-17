"""Arduino/ESP32/M5Stack platform sketch generator."""

from __future__ import annotations
from typing import List

from .base import PlatformSketch


class ArduinoSketch(PlatformSketch):
    """
    Generates a thin Arduino .ino sketch for HAR inference.

    Supported board variants (self.platform):
      arduino / nano_33 / mkr_imu  — Arduino Nano 33 BLE Sense (LSM9DS1)
      esp32                         — ESP32 + MPU6050 (I2C)
      m5stack / m5stick / m5stickc  — M5StickC Plus2 (IMU6886)
      seeed_xiao                    — Seeed XIAO BLE (LSM6DS3)
      generic                       — No IMU driver; user fills sensor read

    Serial output format (CSV, every sample, 115200 baud):
      aX,aY,aZ,gX,gY,gZ               — sensor data only (before first inference)
      aX,aY,aZ,gX,gY,gZ,activity_name — sensor data + last known activity
    Lines starting with '#' are informational and ignored by the Device Test parser.
    """

    # ------------------------------------------------------------------
    # IMU configuration per board
    # ------------------------------------------------------------------
    _BOARD_CONFIG = {
        "arduino": {
            "includes": '#include <Arduino_LSM9DS1.h>',
            "init": 'if (!IMU.begin()) { Serial.println("IMU init failed"); while(1); }',
            "read": (
                'float aX_raw, aY_raw, aZ_raw, gX_raw, gY_raw, gZ_raw;\n'
                '    if (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) return;\n'
                '    IMU.readAcceleration(aX_raw, aY_raw, aZ_raw);\n'
                '    IMU.readGyroscope(gX_raw, gY_raw, gZ_raw);\n'
                '    /* Convert: accel G→m/s², gyro already in deg/s */\n'
                '    aX_raw *= CONVERT_G_TO_MS2; aY_raw *= CONVERT_G_TO_MS2; aZ_raw *= CONVERT_G_TO_MS2;'
            ),
        },
        "nano_33": {
            "includes": '#include <Arduino_LSM9DS1.h>',
            "init": 'if (!IMU.begin()) { Serial.println("IMU init failed"); while(1); }',
            "read": (
                'float aX_raw, aY_raw, aZ_raw, gX_raw, gY_raw, gZ_raw;\n'
                '    if (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) return;\n'
                '    IMU.readAcceleration(aX_raw, aY_raw, aZ_raw);\n'
                '    IMU.readGyroscope(gX_raw, gY_raw, gZ_raw);\n'
                '    aX_raw *= CONVERT_G_TO_MS2; aY_raw *= CONVERT_G_TO_MS2; aZ_raw *= CONVERT_G_TO_MS2;'
            ),
        },
        "mkr_imu": {
            "includes": '#include <Arduino_LSM6DS3.h>',
            "init": 'if (!IMU.begin()) { Serial.println("IMU init failed"); while(1); }',
            "read": (
                'float aX_raw, aY_raw, aZ_raw, gX_raw, gY_raw, gZ_raw;\n'
                '    if (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) return;\n'
                '    IMU.readAcceleration(aX_raw, aY_raw, aZ_raw);\n'
                '    IMU.readGyroscope(gX_raw, gY_raw, gZ_raw);\n'
                '    aX_raw *= CONVERT_G_TO_MS2; aY_raw *= CONVERT_G_TO_MS2; aZ_raw *= CONVERT_G_TO_MS2;'
            ),
        },
        "esp32": {
            "includes": (
                '#include <Wire.h>\n'
                '#include <MPU6050.h>'
            ),
            "init": (
                'Wire.begin();\n'
                '  mpu.initialize();\n'
                '  if (!mpu.testConnection()) { Serial.println("MPU6050 init failed"); while(1); }'
            ),
            "read": (
                'int16_t ax16, ay16, az16, gx16, gy16, gz16;\n'
                '    mpu.getMotion6(&ax16, &ay16, &az16, &gx16, &gy16, &gz16);\n'
                '    /* MPU6050: accel ±2g → 16384 LSB/g, gyro ±250°/s → 131 LSB/°/s */\n'
                '    float aX_raw = ax16 / 16384.0f * CONVERT_G_TO_MS2;\n'
                '    float aY_raw = ay16 / 16384.0f * CONVERT_G_TO_MS2;\n'
                '    float aZ_raw = az16 / 16384.0f * CONVERT_G_TO_MS2;\n'
                '    float gX_raw = gx16 / 131.0f;\n'
                '    float gY_raw = gy16 / 131.0f;\n'
                '    float gZ_raw = gz16 / 131.0f;'
            ),
        },
        "m5stack": {
            "includes": '#include <M5StickCPlus2.h>',
            "init": (
                'M5.begin();\n'
                '  M5.IMU.init();'
            ),
            "read": (
                'float aX_raw, aY_raw, aZ_raw, gX_raw, gY_raw, gZ_raw;\n'
                '    M5.IMU.getAccelData(&aX_raw, &aY_raw, &aZ_raw);\n'
                '    M5.IMU.getGyroData(&gX_raw, &gY_raw, &gZ_raw);\n'
                '    /* M5StickC Plus2 IMU6886: accel in G, gyro in deg/s */\n'
                '    aX_raw *= CONVERT_G_TO_MS2; aY_raw *= CONVERT_G_TO_MS2; aZ_raw *= CONVERT_G_TO_MS2;'
            ),
        },
        "m5stick": {
            "includes": '#include <M5StickCPlus2.h>',
            "init": 'M5.begin();\n  M5.IMU.init();',
            "read": (
                'float aX_raw, aY_raw, aZ_raw, gX_raw, gY_raw, gZ_raw;\n'
                '    M5.IMU.getAccelData(&aX_raw, &aY_raw, &aZ_raw);\n'
                '    M5.IMU.getGyroData(&gX_raw, &gY_raw, &gZ_raw);\n'
                '    aX_raw *= CONVERT_G_TO_MS2; aY_raw *= CONVERT_G_TO_MS2; aZ_raw *= CONVERT_G_TO_MS2;'
            ),
        },
        "m5stickc": {
            "includes": '#include <M5StickCPlus2.h>',
            "init": 'M5.begin();\n  M5.IMU.init();',
            "read": (
                'float aX_raw, aY_raw, aZ_raw, gX_raw, gY_raw, gZ_raw;\n'
                '    M5.IMU.getAccelData(&aX_raw, &aY_raw, &aZ_raw);\n'
                '    M5.IMU.getGyroData(&gX_raw, &gY_raw, &gZ_raw);\n'
                '    aX_raw *= CONVERT_G_TO_MS2; aY_raw *= CONVERT_G_TO_MS2; aZ_raw *= CONVERT_G_TO_MS2;'
            ),
        },
        "seeed_xiao": {
            "includes": '#include <LSM6DS3.h>',
            "init": (
                'if (myIMU.begin() != 0) { Serial.println("IMU init failed"); while(1); }'
            ),
            "read": (
                'float aX_raw = myIMU.readFloatAccelX() * CONVERT_G_TO_MS2;\n'
                '    float aY_raw = myIMU.readFloatAccelY() * CONVERT_G_TO_MS2;\n'
                '    float aZ_raw = myIMU.readFloatAccelZ() * CONVERT_G_TO_MS2;\n'
                '    float gX_raw = myIMU.readFloatGyroX();\n'
                '    float gY_raw = myIMU.readFloatGyroY();\n'
                '    float gZ_raw = myIMU.readFloatGyroZ();'
            ),
        },
    }

    # Default / fallback
    _BOARD_CONFIG["generic"] = {
        "includes": (
            '/* TODO: include your IMU library here\n'
            ' * e.g. #include <Wire.h>\n'
            ' *      #include <MPU6050.h> */'
        ),
        "init": (
            '/* TODO: initialise your IMU here\n'
            '   e.g. Wire.begin(); mpu.initialize(); */'
        ),
        "read": (
            '/* TODO: read sensor values from your IMU.\n'
            '   Replace these dummy values with real sensor reads. */\n'
            '    float aX_raw = 0.0f, aY_raw = 0.0f, aZ_raw = CONVERT_G_TO_MS2;\n'
            '    float gX_raw = 0.0f, gY_raw = 0.0f, gZ_raw = 0.0f;'
        ),
    }
    _BOARD_CONFIG["arm_cortex_m"] = _BOARD_CONFIG["generic"]

    # ------------------------------------------------------------------

    def generate(self) -> str:
        cfg = self._BOARD_CONFIG.get(
            self.platform, self._BOARD_CONFIG["generic"])

        imu_global_decl = ""
        if self.platform == "esp32":
            imu_global_decl = "MPU6050 mpu;"
        elif self.platform in ("seeed_xiao",):
            imu_global_decl = 'LSM6DS3 myIMU(I2C_MODE, 0x6A);'

        # Display code runs inside the inference block (has access to `res`)
        display_code = ""
        if self.platform in ("m5stack", "m5stick", "m5stickc"):
            display_code = (
                '        M5.Lcd.setCursor(0, 0);\n'
                '        M5.Lcd.printf("%-12s\\n", last_activity != NULL ? last_activity : "---");\n'
                '        M5.Lcd.printf("%.2f%%     \\n", res.confidence * 100.0f);'
            )

        class_comment = " * Classes: " + ", ".join(
            f"{i}={c}" for i, c in enumerate(self.classes)
        )

        return f"""\
/*
 * {self.sketch_name}.ino — HAR inference sketch
 *
 * Generated by HAR Edge Deployment Framework v2
 * Platform: {self.platform}
 * Window: {self.window_size} samples @ {self.sampling_rate} Hz
 * Step:   {self.step_size} samples ({(1-self.overlap)*100:.0f}% new per inference)
 *
 * Serial output (CSV, 115200 baud, every sample):
 *   aX,aY,aZ,gX,gY,gZ                 — before first inference
 *   aX,aY,aZ,gX,gY,gZ,activity_name   — after first inference
 * Lines starting with '#' are informational (ignored by Device Test).
{class_comment}
 */

/* Gravity constant for accel G→m/s² conversion (matches training pipeline) */
#define CONVERT_G_TO_MS2 9.80665f

#include "har_classifier.h"
{cfg["includes"]}

{imu_global_decl}

/* ---- Rolling sensor window ---- */
static float sensor_window[HAR_WINDOW_SIZE][HAR_N_CHANNELS];
static int   sample_count  = 0;   /* samples collected since last inference */
static int   total_samples = 0;   /* total samples ever collected */
static const char* last_activity = NULL;  /* last inference result, NULL before first */

static unsigned long sample_interval_us = 1000000UL / HAR_SAMPLE_RATE;

void setup() {{
    Serial.begin(115200);
    while (!Serial) {{}}
    Serial.println("# HAR Edge Framework v2 ready");
    Serial.print("# Sampling at "); Serial.print(HAR_SAMPLE_RATE); Serial.println(" Hz");
    Serial.print("# Window: "); Serial.print(HAR_WINDOW_SIZE);
    Serial.print(" samples, step: "); Serial.print(HAR_STEP_SIZE); Serial.println(" samples");

    {cfg["init"]}

    har_reset_smoothing();
}}

/* Shift window left by one step, then add one new row at the end. */
static void _shift_and_add(float aX, float aY, float aZ,
                            float gX, float gY, float gZ) {{
    for (int i = 0; i < HAR_WINDOW_SIZE - 1; i++) {{
        sensor_window[i][0] = sensor_window[i + 1][0];
        sensor_window[i][1] = sensor_window[i + 1][1];
        sensor_window[i][2] = sensor_window[i + 1][2];
        sensor_window[i][3] = sensor_window[i + 1][3];
        sensor_window[i][4] = sensor_window[i + 1][4];
        sensor_window[i][5] = sensor_window[i + 1][5];
    }}
    int last = HAR_WINDOW_SIZE - 1;
    sensor_window[last][0] = aX;
    sensor_window[last][1] = aY;
    sensor_window[last][2] = aZ;
    sensor_window[last][3] = gX;
    sensor_window[last][4] = gY;
    sensor_window[last][5] = gZ;
}}

void loop() {{
    static unsigned long next_sample_us = 0;
    unsigned long now_us = micros();
    if (now_us < next_sample_us) return;
    next_sample_us = now_us + sample_interval_us;

    /* ---- Read IMU ---- */
    {cfg["read"]}

    /* ---- Add to rolling window ---- */
    _shift_and_add(aX_raw, aY_raw, aZ_raw, gX_raw, gY_raw, gZ_raw);
    total_samples++;
    sample_count++;

    /* ---- Inference (once window filled and step reached) ---- */
    if (total_samples >= HAR_WINDOW_SIZE && sample_count >= HAR_STEP_SIZE) {{
        sample_count = 0;
        har_result_t res;
        har_classify(sensor_window, &res);
        last_activity = har_get_class_name(res.predicted_class);

        /* Debug: print raw probabilities for each class (ignored by Device Test) */
        Serial.print("# PRED: class="); Serial.print(res.predicted_class);
        Serial.print(" conf="); Serial.print(res.confidence, 4);
        Serial.print(" probs=[");
        for (int i = 0; i < HAR_NUM_CLASSES; i++) {{
            if (i > 0) Serial.print(",");
            Serial.print(res.probabilities[i], 4);
        }}
        Serial.print("] name=");
        Serial.println(last_activity != NULL ? last_activity : "none");
{display_code}
    }}

    /* ---- Serial CSV output — every sample, Device Test tab format ---- */
    /* Format: aX,aY,aZ,gX,gY,gZ[,activity_name]  (no timestamp)         */
    Serial.print(aX_raw, 4); Serial.print(',');
    Serial.print(aY_raw, 4); Serial.print(',');
    Serial.print(aZ_raw, 4); Serial.print(',');
    Serial.print(gX_raw, 4); Serial.print(',');
    Serial.print(gY_raw, 4); Serial.print(',');
    Serial.print(gZ_raw, 4);
    if (last_activity != NULL) {{
        Serial.print(',');
        Serial.print(last_activity);
    }}
    Serial.println();
}}
"""
