"""
Zephyr RTOS Code Generator
Generates C code targeting the Zephyr RTOS for HAR inference on
nRF52840, STM32, ESP32, and other Zephyr-supported SoCs.

Produces:
  - Header (.h)   — same as generic C header via base class
  - Source (.c)    — model implementation with Zephyr logging
  - Example (.c)   — app_main() with device-tree sensor bindings,
                     k_sleep, and LOG_MODULE_REGISTER
"""

from typing import Dict, Any
from .base_generator import BaseCodeGenerator
from .random_forest_generator import RandomForestCodeGenerator
from .neural_network_generator import NeuralNetworkCodeGenerator
from .svm_generator import SVMCodeGenerator


class ZephyrCodeGenerator(BaseCodeGenerator):
    """Generates Zephyr RTOS C code using composition.

    Like the ARM Cortex-M generator, this uses an *inner* model-specific
    generator for the prediction logic / model data, and wraps it with
    Zephyr-specific boilerplate (device-tree bindings, logging, etc.).
    """

    def __init__(self, model_data: Dict[str, Any], platform: str = 'zephyr',
                 optimization: str = 'balanced', overlap: float = 0.5):
        # CNN models don't use traditional features — provide placeholders
        model_type_name = model_data.get('model_type', '')
        if model_type_name == 'pytorch_cnn' and not model_data.get('feature_names'):
            model_data = dict(model_data)
            n_ch = model_data.get('n_channels', 6)
            model_data['feature_names'] = [f'ch{i}' for i in range(n_ch)]

        super().__init__(model_data, platform, optimization, overlap)
        self.model_type_name = model_data.get('model_type', '')

        # Create inner model-specific generator
        if self.model_type_name == 'random_forest':
            self._inner = RandomForestCodeGenerator(model_data, platform, optimization, overlap)
        elif self.model_type_name in ('neural_network', 'pytorch_mlp'):
            self._inner = NeuralNetworkCodeGenerator(model_data, platform, optimization, overlap)
        elif self.model_type_name == 'svm':
            self._inner = SVMCodeGenerator(model_data, platform, optimization, overlap)
        elif self.model_type_name == 'pytorch_cnn':
            from .cnn_generator import CNNCodeGenerator
            self._inner = CNNCodeGenerator(model_data, platform, optimization, overlap)
        else:
            self._inner = None

    # ------------------------------------------------------------------ #
    #  For CNN models, bypass the composition pattern and delegate fully  #
    # ------------------------------------------------------------------ #

    def generate_header(self) -> str:
        if self.model_type_name == 'pytorch_cnn' and self._inner is not None:
            return self._inner.generate_header()
        return super().generate_header()

    def generate_implementation(self, header_filename: str = None) -> str:
        if self.model_type_name == 'pytorch_cnn' and self._inner is not None:
            return self._inner.generate_implementation(header_filename)
        return super().generate_implementation(header_filename)

    # ------------------------------------------------------------------ #
    #  Delegated abstract methods                                         #
    # ------------------------------------------------------------------ #

    def _get_model_specific_declarations(self) -> str:
        """Zephyr-specific declarations + inner model declarations."""
        zephyr_decl = """
// Zephyr RTOS specifics
#ifdef CONFIG_ARM
    #include <zephyr/arch/arm/aarch32/cortex_m/cmsis.h>
#endif

// Zephyr utility functions
void har_print_system_info(void);
"""
        if self._inner:
            zephyr_decl += self._inner._get_model_specific_declarations()
        return zephyr_decl

    def _generate_model_specific_implementation(self) -> str:
        if self._inner:
            return self._inner._generate_model_specific_implementation()
        return "/* No model-specific implementation — unknown model type */\n"

    def _generate_prediction_function(self) -> str:
        if self._inner:
            return self._inner._generate_prediction_function()
        return (
            "int har_predict_internal(float features[NUM_FEATURES]) {\n"
            "    return 0;  /* fallback */\n"
            "}\n"
        )

    def _generate_utility_functions(self) -> str:
        utils = ""
        if self._inner:
            utils += self._inner._generate_utility_functions()
        utils += """

// ---- Zephyr Platform Utilities ----

void har_print_system_info(void) {
    HAR_LOG("Zephyr HAR System");
    HAR_LOG("Model type: %s", """" + self.model_type_name + """");
    HAR_LOG("Features: %d  Classes: %d", NUM_FEATURES, NUM_CLASSES);
    HAR_LOG("Sampling: %d Hz  Window: %d", SAMPLING_RATE, WINDOW_SIZE);
}
"""
        return utils

    # ------------------------------------------------------------------ #
    #  Example sketch — Zephyr main() with device-tree sensor binding    #
    # ------------------------------------------------------------------ #

    def generate_example_sketch(self, header_filename: str = None) -> str:
        """Generate a Zephyr ``main.c`` example.

        Uses the Zephyr sensor subsystem (``zephyr/drivers/sensor.h``) for
        IMU access and ``k_sleep`` for timing.
        For CNN models, delegates to the CNN generator.
        """
        if self.model_type_name == 'pytorch_cnn' and self._inner is not None:
            return self._inner.generate_example_sketch(header_filename)

        import os
        header_include = os.path.basename(header_filename) if header_filename else "har_model.h"

        delay_ms = int(1000 * self.window_size / max(self.sampling_rate, 1))

        return f"""/*
 * HAR Model — Zephyr RTOS Example
 * Model type  : {self.model_type_name.upper()}
 * Optimization: {self.optimization.upper()}
 *
 * Build:
 *   west build -b <your_board> -- -DOVERLAY_CONFIG=overlay.conf
 *   west flash
 *
 * prj.conf should include:
 *   CONFIG_LOG=y
 *   CONFIG_SENSOR=y
 *   CONFIG_I2C=y
 *   CONFIG_NEWLIB_LIBC=y
 *   CONFIG_FPU=y
 */

#include "{header_include}"
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/drivers/sensor.h>
#include <math.h>

LOG_MODULE_REGISTER(har_example, LOG_LEVEL_INF);

/* ---------- Device-tree sensor binding ---------- */

/* Adjust this label to match your board's device tree.            */
/* Common choices:                                                 */
/*   &lsm6ds3    — Seeed XIAO nRF52840 Sense                       */
/*   &mpu6050    — many breakout boards                             */
/*   &icm42688p  — Nordic Thingy:53                                 */
#define IMU_NODE DT_ALIAS(imu0)

#if DT_NODE_HAS_STATUS(IMU_NODE, okay)
static const struct device *imu_dev = DEVICE_DT_GET(IMU_NODE);
#else
#error "IMU device not found in device tree. Define an 'imu0' alias."
static const struct device *imu_dev;
#endif

/* ---------- Sensor reading helper ---------- */

static int read_sensor_window(float sensor_data[][6]) {{
    struct sensor_value accel[3];
    struct sensor_value gyro[3];

    for (int i = 0; i < WINDOW_SIZE; i++) {{
        if (sensor_fetch(imu_dev) < 0) {{
            LOG_ERR("Sensor fetch failed at sample %d", i);
            return -1;
        }}

        sensor_channel_get(imu_dev, SENSOR_CHAN_ACCEL_XYZ, accel);
        sensor_channel_get(imu_dev, SENSOR_CHAN_GYRO_XYZ, gyro);

        sensor_data[i][0] = (float)sensor_value_to_double(&accel[0]);
        sensor_data[i][1] = (float)sensor_value_to_double(&accel[1]);
        sensor_data[i][2] = (float)sensor_value_to_double(&accel[2]);
        sensor_data[i][3] = (float)sensor_value_to_double(&gyro[0]);
        sensor_data[i][4] = (float)sensor_value_to_double(&gyro[1]);
        sensor_data[i][5] = (float)sensor_value_to_double(&gyro[2]);

        k_sleep(K_MSEC(1000 / SAMPLING_RATE));
    }}
    return 0;
}}

/* ---------- Main thread ---------- */

void main(void) {{
    LOG_INF("HAR Model — {self.model_type_name.upper()} ({self.optimization.title()})");
    LOG_INF("Features: %d | Classes: %d | Window: %d @ %d Hz",
            NUM_FEATURES, NUM_CLASSES, WINDOW_SIZE, SAMPLING_RATE);

    if (!device_is_ready(imu_dev)) {{
        LOG_ERR("IMU device not ready");
        return;
    }}
    LOG_INF("IMU device ready");

    har_init();

    float sensor_data[WINDOW_SIZE][6];
    float features[NUM_FEATURES];
    int iteration = 0;

    while (1) {{
        if (read_sensor_window(sensor_data) != 0) {{
            LOG_ERR("Sensor read error — retrying in 1 s");
            k_sleep(K_SECONDS(1));
            continue;
        }}

        extract_features(sensor_data, WINDOW_SIZE, features);
        int predicted_class = har_predict(features);
        const char *activity = get_activity_name(predicted_class);

        LOG_INF("[%04d] Predicted: %s (class %d)", iteration, activity, predicted_class);
        iteration++;

        k_sleep(K_MSEC({delay_ms}));
    }}
}}
"""
