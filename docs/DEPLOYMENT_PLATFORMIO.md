# Deploying HAR Models with PlatformIO

## Overview

PlatformIO is the recommended toolchain for compiling and flashing generated HAR code to embedded devices. It handles library dependencies, board configurations, and serial uploading automatically.

## Prerequisites

1. **PlatformIO CLI** — install via pip or the VS Code extension:

   ```bash
   pip install platformio
   # or install the PlatformIO IDE extension in VS Code
   ```

2. **USB drivers** for your board (CP2104 for ESP32, CH9102 for M5StickC Plus2, etc.)

3. A trained model (`.joblib`) in `persistent_data/models/`

## Step-by-Step Deployment

### 1. Generate Code from the GUI

1. Open the app (`python app.py`) and navigate to the **Code Generation** tab.
2. Select your trained model from the dropdown.
3. Choose the framework and target board:

   | Framework | Board | Generated Files |
   |-----------|-------|-----------------|
   | Arduino C++ | M5StickC Plus2 | `.h`, `.cpp`, `.ino` |
   | Arduino C++ | ESP32 DevKit | `.h`, `.cpp`, `.ino` |
   | Arduino C++ | XIAO nRF52840 Sense | `.h`, `.cpp`, `.ino` |
   | Generic C | Any | `.h`, `.c`, `_main.c` |

4. Select deployment approach (Direct, TFLite Micro, or ONNX Runtime).
5. Choose optimization level and click **Generate Code**.

Generated files are saved to:

```
persistent_data/generated/<model_type>_models/<model_name>/
```

### 2. Create a PlatformIO Project

```bash
# Create a new project directory
mkdir har_deploy && cd har_deploy

# Initialize PlatformIO project (example: M5StickC Plus2)
pio init --board m5stick-c --project-option "framework=arduino"
```

Or create `platformio.ini` manually. The framework auto-generates this for supported boards — check the generated folder for `platformio.ini`:

**M5StickC Plus2:**

```ini
[env:har_model]
platform = espressif32
board = m5stick-c
framework = arduino
monitor_speed = 115200
upload_speed = 921600
lib_deps =
    m5stack/M5StickCPlus2
```

**ESP32 DevKit:**

```ini
[env:har_model]
platform = espressif32
board = esp32dev
framework = arduino
monitor_speed = 115200
upload_speed = 921600
```

**XIAO nRF52840 Sense:**

```ini
[env:har_model]
platform = nordicnrf52
board = xiaonRF52840Sense
framework = arduino
monitor_speed = 115200
lib_deps =
    sparkfun/SparkFun LSM6DS3 Breakout
```

**STM32F4:**

```ini
[env:har_model]
platform = ststm32
board = genericSTM32F407VET6
framework = arduino
monitor_speed = 115200
```

### 3. Copy Generated Files

Copy the generated files into the PlatformIO project:

```bash
# For Arduino C++ output:
cp persistent_data/generated/<model>/*.h    har_deploy/src/
cp persistent_data/generated/<model>/*.cpp  har_deploy/src/
cp persistent_data/generated/<model>/*.ino  har_deploy/src/main.cpp
```

> **Note**: PlatformIO uses `main.cpp` instead of `.ino`. Rename the `.ino` file to `main.cpp`, or place it in `src/` with a `.cpp` extension.

For TFLite Micro deployment, also add the TFLite library:

```ini
lib_deps =
    tensorflow/TensorFlowLite_ESP32@^0.9.0
```

### 4. Build and Upload

```bash
# Compile
pio run

# Compile and upload
pio run --target upload

# Monitor serial output
pio device monitor --baud 115200
```

Or from the framework's GUI:

1. Select **PlatformIO** as the toolchain.
2. Select the serial port.
3. Click **Compile** or **Compile & Flash**.

### 5. Verify on Device

Open the serial monitor. You should see:

```
HAR Model - Balanced Optimization
========================================
Configuration:
  Sampling Rate: 100 Hz
  Window Size: 150 samples
  Features: 33
  Optimization: Balanced
✅ M5StickC Plus2 initialized!
HAR Model Ready!
Collecting sensor data...
```

After collecting one window of data (e.g. 1.5 seconds), predictions appear:

```
aX,aY,aZ,gX,gY,gZ,walking
-2.3456,8.1234,4.5678,10.23,-5.67,2.34,walking
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `pio` not found | Run `pip install platformio` or add to PATH |
| Upload fails | Check USB cable (data, not charge-only), install drivers |
| Board not recognized | Run `pio device list` to verify port detection |
| Library not found | Run `pio lib install <name>` or add to `platformio.ini` |
| Compilation errors on AVR | Model may be too large for 2KB RAM — use ESP32 or optimize |
| M5StickC display blank | Ensure `M5StickCPlus2` library version is compatible |

## Memory Considerations

| Board | RAM | Flash | Max Model Size (approx.) |
|-------|-----|-------|--------------------------|
| Arduino Uno | 2 KB | 32 KB | ≤5 trees, ≤10 features |
| ESP32 | 520 KB | 4 MB | Large RF, full NN |
| M5StickC Plus2 | 320 KB | 8 MB | Large RF, full NN |
| XIAO nRF52840 | 256 KB | 1 MB | Medium models |
| STM32F4 | 192 KB | 1 MB | Medium models |

Use the **Resource Analysis** button in the Code Generation tab to check if your model fits the target board before flashing.
