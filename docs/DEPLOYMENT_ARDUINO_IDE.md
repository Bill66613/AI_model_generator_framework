# Deploying HAR Models with Arduino IDE

## Overview

The Arduino IDE is a simpler alternative to PlatformIO for compiling and flashing generated HAR models. It works well for quick prototyping but requires manual library installation.

## Prerequisites

1. **Arduino IDE 2.x** — download from [arduino.cc](https://www.arduino.cc/en/software)
2. **Board support packages** installed via Board Manager
3. **Arduino CLI** (optional, for command-line compilation)

### Board Support Installation

Open Arduino IDE → **File → Preferences → Additional Board Manager URLs**, add:

| Board Family | Board Manager URL |
|-------------|-------------------|
| ESP32 / M5Stack | `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json` |
| Seeed nRF52 | `https://files.seeedstudio.com/arduino/package_seeeduino_boards_index.json` |
| STM32 | `https://github.com/stm32duino/BoardManagerFiles/raw/main/package_stmicroelectronics_index.json` |

Then go to **Tools → Board → Board Manager** and install the relevant package.

### Library Installation

Go to **Sketch → Include Library → Manage Libraries** and install:

| Board | Required Libraries |
|-------|--------------------|
| M5StickC Plus2 | `M5StickCPlus2` by M5Stack |
| XIAO nRF52840 Sense | `SparkFun LSM6DS3 Breakout` |
| TFLite deployment | `Arduino_TensorFlowLite` |

## Step-by-Step Deployment

### 1. Generate Code

Use the framework GUI (Code Generation tab) to generate code for your model. Select **Arduino C++** as the framework. Generated files are saved to:

```
persistent_data/generated/<model_type>_models/<model_name>/
```

Typically three files are generated:

- `har_<model>_<platform>.h` — model constants, weights, feature extraction declarations
- `har_<model>_<platform>.cpp` — feature extraction, scaling, prediction implementation
- `har_<model>_<platform>.ino` — Arduino sketch with IMU setup and inference loop

### 2. Create a Sketch Folder

Arduino IDE requires the `.ino` file to be in a folder with the same name:

```
har_model/
├── har_model.ino          ← renamed from generated .ino
├── har_model.h            ← renamed from generated .h
└── har_model.cpp          ← renamed from generated .cpp
```

> **Important**: The folder name must match the `.ino` filename exactly. Rename files if needed.

### 3. Open in Arduino IDE

1. Open Arduino IDE
2. **File → Open** → select the `.ino` file
3. All `.h` and `.cpp` files in the same folder are automatically included

### 4. Configure the Board

Go to **Tools** and set:

**M5StickC Plus2:**

| Setting | Value |
|---------|-------|
| Board | `M5StickC-Plus` (under ESP32 Arduino) |
| Upload Speed | `921600` |
| CPU Frequency | `240MHz` |
| Flash Size | `8MB` |
| Port | (your device's COM port) |

**ESP32 DevKit:**

| Setting | Value |
|---------|-------|
| Board | `ESP32 Dev Module` |
| Upload Speed | `921600` |
| Port | (your device's COM port) |

**XIAO nRF52840 Sense:**

| Setting | Value |
|---------|-------|
| Board | `Seeed XIAO nRF52840 Sense` |
| Port | (your device's COM port) |

### 5. Compile and Upload

1. Click **Verify** (✓) to compile
2. Click **Upload** (→) to flash to device
3. Open **Serial Monitor** (magnifying glass icon) at `115200` baud

### 6. Using Arduino CLI (Alternative)

If you prefer command-line workflow:

```bash
# Install Arduino CLI
# https://arduino.github.io/arduino-cli/installation/

# Install board cores
arduino-cli core install esp32:esp32
arduino-cli core install Seeeduino:nrf52

# Install libraries
arduino-cli lib install "M5StickCPlus2"
arduino-cli lib install "SparkFun LSM6DS3 Breakout"

# Compile
arduino-cli compile --fqbn m5stack:esp32:m5stick_c ./har_model/

# Upload
arduino-cli upload -p COM3 --fqbn m5stack:esp32:m5stick_c ./har_model/

# Monitor serial output
arduino-cli monitor -p COM3 --config baudrate=115200
```

The framework GUI can also invoke Arduino CLI directly — select **Arduino CLI** as the toolchain in the Code Generation tab.

## Generated Code Structure

The generated `.ino` sketch follows this structure:

```cpp
#include "har_model.h"       // Model constants, weights
#include "M5StickCPlus2.h"   // Board-specific IMU library

float sensor_buffer[WINDOW_SIZE][N_CHANNELS];
int buffer_index = 0;
float features[NUM_FEATURES];

void setup() {
    Serial.begin(115200);
    har_init();              // Initialize model
    // Board-specific IMU initialization
    StickCP2.begin(cfg);     // M5StickC Plus2 example
}

void loop() {
    // 1. Read IMU at configured sampling rate
    // 2. Store in circular buffer
    // 3. When buffer full:
    //    a. extract_features(buffer, window_size, features)
    //    b. har_predict(features, &confidence)
    //    c. Print prediction over Serial
    // 4. Slide window by overlap amount
}
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Board not found" | Install the board support package via Board Manager |
| "Library not found" | Install via Library Manager or add to `libraries/` folder |
| Compile error: "sketch too large" | Model too large for board — use ESP32 or reduce model |
| Upload error: "can't open port" | Close Serial Monitor before uploading; check driver |
| M5StickC not detected | Hold power button 2s to turn on; try different USB cable |
| Wrong predictions | Ensure same sampling rate and sensor units as training data |

## Comparison: Arduino IDE vs PlatformIO

| Feature | Arduino IDE | PlatformIO |
|---------|------------|------------|
| Setup complexity | Easy (GUI) | Moderate (config files) |
| Library management | Manual (Library Manager) | Automatic (`platformio.ini`) |
| Multi-file projects | Requires same-name folder | Standard `src/` layout |
| CI/CD integration | Limited | Excellent |
| Build speed | Slower | Faster (incremental) |
| Board support | Broad | Broad + custom boards |
| Recommended for | Quick testing | Production deployment |
