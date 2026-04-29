"""
ONNX Runtime Code Generator

Generates deployment code for ONNX Runtime inference on edge devices.
This approach exports the model as an ONNX file and generates code that
uses the ONNX Runtime C/C++ API for inference.

ONNX Runtime supports:
- Full ONNX Runtime (Linux/Windows edge devices, Raspberry Pi, Jetson)
- ONNX Runtime Mobile (Android/iOS)
- ONNX Runtime Micro (experimental, Cortex-M)

The feature extraction code is still generated using the same verified
C++ implementation from BaseCodeGenerator (maintaining training-deployment parity).

Requires: onnx, skl2onnx (sklearn), torch (PyTorch)
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List

from .base_generator import BaseCodeGenerator

logger = logging.getLogger(__name__)


class ONNXRuntimeCodeGenerator(BaseCodeGenerator):
    """
    Code generator for ONNX Runtime deployment.

    Generates:
    - har_model.h: Model declarations + feature extraction header
    - har_model.cpp: Feature extraction + ONNX Runtime inference wrapper
    - har_model.onnx: The ONNX model file (binary)
    - har_onnx.ino/.cpp: Example sketch/application

    For microcontrollers with limited resources, also generates a
    model_data.h with the ONNX model embedded as a C byte array.
    """

    def __init__(self, model_data: Dict[str, Any], platform: str = 'arduino',
                 optimization: str = 'balanced', overlap: float = 0.5,
                 quantization: str = 'none'):
        super().__init__(model_data, platform, optimization, overlap, quantization='none')
        self.onnx_quantization = quantization
        self._onnx_bytes = None
        self._onnx_info = None

    def convert_model(self) -> bytes:
        """
        Convert the model to ONNX format.

        Returns:
            ONNX model bytes
        """
        from .converters.onnx_converter import ONNXConverter

        model_object = self.model_data.get('model_object')
        if model_object is None:
            raise ValueError(
                "ONNX conversion requires the original model object. "
                "Make sure 'model_object' is present in model_data."
            )

        converter = ONNXConverter(
            model_object=model_object,
            model_type=self.model_type,
            feature_names=self.feature_names,
            classes=self.classes,
            model_params=self.model_data.get('model_params', {})
        )

        self._onnx_bytes = converter.convert()
        self._onnx_info = converter.get_model_info()

        return self._onnx_bytes

    # --- Abstract method implementations ---

    def _get_model_specific_declarations(self) -> str:
        """ONNX Runtime header declarations."""
        onnx_size = len(self._onnx_bytes) if self._onnx_bytes else 0

        lines = []
        lines.append("")
        lines.append("// ---- ONNX Runtime Model ----")
        lines.append(f"#define ONNX_MODEL_SIZE {onnx_size}")
        lines.append(f"#define DEPLOYMENT_APPROACH \"onnx_runtime\"")
        lines.append("")
        lines.append("// ONNX model embedded as byte array (for MCU deployment)")
        lines.append("extern const unsigned char g_onnx_model[];")
        lines.append("extern const unsigned int g_onnx_model_len;")
        lines.append("")
        lines.append("// ONNX Runtime inference functions")
        lines.append("bool onnx_init(const char* model_path);")
        lines.append("bool onnx_init_from_buffer(const unsigned char* buffer, unsigned int size);")
        lines.append(f"int onnx_predict(float features[NUM_FEATURES], float probabilities[NUM_CLASSES]);")
        lines.append("void onnx_cleanup();")
        lines.append("const char* onnx_get_class_name(int class_idx);")
        return '\n'.join(lines)

    def _generate_model_specific_implementation(self) -> str:
        """Generate the ONNX model as embedded C byte array."""
        if self._onnx_bytes is None:
            try:
                self.convert_model()
            except (ImportError, ValueError) as e:
                return self._generate_placeholder_model(str(e))

        return self._generate_onnx_c_array()

    def _generate_onnx_c_array(self) -> str:
        """Convert ONNX model to C byte array."""
        data = self._onnx_bytes
        lines = []
        lines.append(f"// ONNX model - {len(data)} bytes")
        lines.append(f"// Model type: {self.model_type}")
        lines.append(f"// Classes: {', '.join(self.classes)}")
        lines.append(f"// Features: {len(self.feature_names)}")
        if self._onnx_info:
            lines.append(f"// ONNX opset: {self._onnx_info.get('opset_version', 'unknown')}")
            lines.append(f"// Parameters: {self._onnx_info.get('total_parameters', 'unknown')}")
        lines.append(f"")
        lines.append(f"alignas(16) const unsigned char g_onnx_model[] = {{")

        for i in range(0, len(data), 12):
            chunk = data[i:i+12]
            hex_values = ', '.join(f'0x{b:02x}' for b in chunk)
            comma = ',' if i + 12 < len(data) else ''
            lines.append(f"    {hex_values}{comma}")

        lines.append(f"}};")
        lines.append(f"const unsigned int g_onnx_model_len = {len(data)};")

        return '\n'.join(lines)

    def _generate_prediction_function(self) -> str:
        """Generate ONNX Runtime inference wrapper."""
        lines = []
        lines.append("// ---- ONNX Runtime Inference ----")
        lines.append("//")
        lines.append("// Two deployment modes supported:")
        lines.append("// 1. File-based: Load .onnx from filesystem (Linux/RPi/Jetson)")
        lines.append("// 2. Buffer-based: Model embedded in firmware (MCU)")
        lines.append("//")
        lines.append("// For full ONNX Runtime (Linux/edge devices):")
        lines.append("//   #include \"onnxruntime_cxx_api.h\"")
        lines.append("// For ONNX Runtime Micro (Cortex-M, experimental):")
        lines.append("//   Contact Microsoft for ORT Micro SDK")
        lines.append("")

        # Generate platform-appropriate code
        if self.platform in ('arduino', 'seeed_xiao', 'esp32', 'teensy'):
            return self._generate_mcu_onnx_prediction(lines)
        else:
            return self._generate_full_onnx_prediction(lines)

    def _generate_mcu_onnx_prediction(self, lines: list) -> str:
        """Generate ONNX prediction for MCU (lightweight wrapper)."""
        lines.append("// NOTE: Full ONNX Runtime is too large for most MCUs.")
        lines.append("// For MCU deployment, consider:")
        lines.append("//   1. TFLite Micro approach (recommended for MCU)")
        lines.append("//   2. Direct code generation (current default)")
        lines.append("//   3. ONNX Runtime Micro (if available for your platform)")
        lines.append("//")
        lines.append("// This code provides the ONNX model as a byte array that can be")
        lines.append("// used with any compatible runtime on your target platform.")
        lines.append("")
        lines.append("// Placeholder inference — replace with your runtime's API")
        lines.append(f"static bool onnx_initialized = false;")
        lines.append("")
        lines.append("bool onnx_init_from_buffer(const unsigned char* buffer, unsigned int size) {")
        lines.append("    // Initialize your ONNX runtime here with the model buffer")
        lines.append("    // For ONNX Runtime Micro: OrtCreateSessionFromArray(buffer, size, ...)")
        lines.append("    onnx_initialized = (buffer != nullptr && size > 0);")
        lines.append("    return onnx_initialized;")
        lines.append("}")
        lines.append("")
        lines.append("bool onnx_init(const char* model_path) {")
        lines.append("    // Load model from filesystem (not typically available on MCU)")
        lines.append("    return onnx_init_from_buffer(g_onnx_model, g_onnx_model_len);")
        lines.append("}")
        lines.append("")
        lines.append(f"int onnx_predict(float features[NUM_FEATURES], float probabilities[NUM_CLASSES]) {{")
        lines.append("    if (!onnx_initialized) {")
        lines.append("        if (!onnx_init_from_buffer(g_onnx_model, g_onnx_model_len)) return -1;")
        lines.append("    }")
        lines.append("")
        lines.append("    // TODO: Replace with actual ONNX Runtime Micro inference call")
        lines.append("    // Example for ORT Micro:")
        lines.append("    // OrtValue* input = OrtCreateTensorWithData(features, NUM_FEATURES * sizeof(float));")
        lines.append("    // OrtValue* output = nullptr;")
        lines.append("    // OrtRun(session, &input, 1, &output, 1);")
        lines.append("    // memcpy(probabilities, OrtGetTensorData(output), NUM_CLASSES * sizeof(float));")
        lines.append("")
        lines.append("    // Find predicted class")
        lines.append("    float max_prob = -1.0f;")
        lines.append("    int predicted = 0;")
        lines.append("    for (int i = 0; i < NUM_CLASSES; i++) {")
        lines.append("        if (probabilities[i] > max_prob) {")
        lines.append("            max_prob = probabilities[i];")
        lines.append("            predicted = i;")
        lines.append("        }")
        lines.append("    }")
        lines.append("    return predicted;")
        lines.append("}")
        lines.append("")
        lines.append("void onnx_cleanup() {")
        lines.append("    onnx_initialized = false;")
        lines.append("}")

        return '\n'.join(lines)

    def _generate_full_onnx_prediction(self, lines: list) -> str:
        """Generate ONNX prediction using full ONNX Runtime C++ API."""
        lines.append("#ifdef USE_ONNX_RUNTIME")
        lines.append("#include \"onnxruntime_cxx_api.h\"")
        lines.append("")
        lines.append("static Ort::Env* ort_env = nullptr;")
        lines.append("static Ort::Session* ort_session = nullptr;")
        lines.append("static Ort::MemoryInfo* ort_memory_info = nullptr;")
        lines.append("")
        lines.append("bool onnx_init(const char* model_path) {")
        lines.append("    try {")
        lines.append("        ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, \"har_inference\");")
        lines.append("        Ort::SessionOptions session_options;")
        lines.append("        session_options.SetIntraOpNumThreads(1);")
        lines.append("        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);")
        lines.append("")
        lines.append("        ort_session = new Ort::Session(*ort_env, model_path, session_options);")
        lines.append("        ort_memory_info = new Ort::MemoryInfo(")
        lines.append("            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));")
        lines.append("")
        lines.append("        HAR_LOG(\"ONNX Runtime initialized: %s\", model_path);")
        lines.append("        return true;")
        lines.append("    } catch (const Ort::Exception& e) {")
        lines.append("        HAR_LOG(\"ONNX Runtime error: %s\", e.what());")
        lines.append("        return false;")
        lines.append("    }")
        lines.append("}")
        lines.append("")
        lines.append("bool onnx_init_from_buffer(const unsigned char* buffer, unsigned int size) {")
        lines.append("    try {")
        lines.append("        ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, \"har_inference\");")
        lines.append("        Ort::SessionOptions session_options;")
        lines.append("        session_options.SetIntraOpNumThreads(1);")
        lines.append("")
        lines.append("        ort_session = new Ort::Session(*ort_env, buffer, size, session_options);")
        lines.append("        ort_memory_info = new Ort::MemoryInfo(")
        lines.append("            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));")
        lines.append("")
        lines.append("        HAR_LOG(\"ONNX Runtime initialized from buffer (%u bytes)\", size);")
        lines.append("        return true;")
        lines.append("    } catch (const Ort::Exception& e) {")
        lines.append("        HAR_LOG(\"ONNX Runtime error: %s\", e.what());")
        lines.append("        return false;")
        lines.append("    }")
        lines.append("}")
        lines.append("")
        lines.append(f"int onnx_predict(float features[NUM_FEATURES], float probabilities[NUM_CLASSES]) {{")
        lines.append("    if (ort_session == nullptr) return -1;")
        lines.append("")
        lines.append("    try {")
        lines.append("        // Create input tensor")
        lines.append(f"        std::array<int64_t, 2> input_shape = {{1, NUM_FEATURES}};")
        lines.append("        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(")
        lines.append("            *ort_memory_info, features, NUM_FEATURES,")
        lines.append("            input_shape.data(), input_shape.size());")
        lines.append("")
        lines.append("        // Run inference")
        lines.append("        const char* input_names[] = {\"features\"};")
        lines.append("        const char* output_names[] = {\"output\"};")
        lines.append("")
        lines.append("        auto output_tensors = ort_session->Run(")
        lines.append("            Ort::RunOptions{nullptr},")
        lines.append("            input_names, &input_tensor, 1,")
        lines.append("            output_names, 1);")
        lines.append("")
        lines.append("        // Get output probabilities")
        lines.append("        float* output_data = output_tensors[0].GetTensorMutableData<float>();")
        lines.append("")
        lines.append("        float max_prob = -1.0f;")
        lines.append("        int predicted = 0;")
        lines.append("        for (int i = 0; i < NUM_CLASSES; i++) {")
        lines.append("            probabilities[i] = output_data[i];")
        lines.append("            if (probabilities[i] > max_prob) {")
        lines.append("                max_prob = probabilities[i];")
        lines.append("                predicted = i;")
        lines.append("            }")
        lines.append("        }")
        lines.append("        return predicted;")
        lines.append("")
        lines.append("    } catch (const Ort::Exception& e) {")
        lines.append("        HAR_LOG(\"Inference error: %s\", e.what());")
        lines.append("        return -1;")
        lines.append("    }")
        lines.append("}")
        lines.append("")
        lines.append("void onnx_cleanup() {")
        lines.append("    delete ort_session; ort_session = nullptr;")
        lines.append("    delete ort_memory_info; ort_memory_info = nullptr;")
        lines.append("    delete ort_env; ort_env = nullptr;")
        lines.append("}")
        lines.append("")
        lines.append("#else  // No ONNX Runtime — placeholder")
        lines.append("")
        lines.append("bool onnx_init(const char* model_path) {")
        lines.append("    HAR_LOG(\"ONNX Runtime not available. Define USE_ONNX_RUNTIME to enable.\");")
        lines.append("    return false;")
        lines.append("}")
        lines.append("")
        lines.append("bool onnx_init_from_buffer(const unsigned char* buffer, unsigned int size) {")
        lines.append("    return false;")
        lines.append("}")
        lines.append("")
        lines.append(f"int onnx_predict(float features[NUM_FEATURES], float probabilities[NUM_CLASSES]) {{")
        lines.append("    return -1;")
        lines.append("}")
        lines.append("")
        lines.append("void onnx_cleanup() {}")
        lines.append("")
        lines.append("#endif  // USE_ONNX_RUNTIME")

        return '\n'.join(lines)

    def _generate_utility_functions(self) -> str:
        """Generate ONNX utility functions."""
        lines = []
        lines.append("")
        lines.append("// ---- ONNX Utility Functions ----")
        lines.append("")
        lines.append("const char* onnx_get_class_name(int class_idx) {")
        lines.append("    if (class_idx < 0 || class_idx >= NUM_CLASSES) return \"unknown\";")
        lines.append("    return CLASS_NAMES[class_idx];")
        lines.append("}")
        lines.append("")
        lines.append("void onnx_print_info() {")
        lines.append(f"    HAR_LOG(\"Model: ONNX Runtime ({self.model_type})\");")
        lines.append(f"    HAR_LOG(\"Model size: %u bytes\", g_onnx_model_len);")
        lines.append(f"    HAR_LOG(\"Features: %d, Classes: %d\", NUM_FEATURES, NUM_CLASSES);")
        if self._onnx_info:
            lines.append(f"    HAR_LOG(\"ONNX opset: {self._onnx_info.get('opset_version', '?')}\");")
            lines.append(f"    HAR_LOG(\"Parameters: {self._onnx_info.get('total_parameters', '?')}\");")
        lines.append("}")

        return '\n'.join(lines)

    def generate_example_sketch(self, header_filename: str = None) -> str:
        """Generate example sketch for ONNX Runtime deployment."""
        if header_filename is None:
            header_filename = 'har_model.h'

        sketch = []
        sketch.append(f"/*")
        sketch.append(f" * HAR ONNX Runtime Deployment")
        sketch.append(f" * Model type: {self.model_type}")
        sketch.append(f" * Classes: {', '.join(self.classes)}")
        sketch.append(f" * Features: {len(self.feature_names)}")
        sketch.append(f" * Deployment approach: ONNX Runtime")
        sketch.append(f" *")
        sketch.append(f" * Generated by HAR Edge Deployment Framework")
        sketch.append(f" *")
        sketch.append(f" * For full ONNX Runtime (Linux/RPi/Jetson):")
        sketch.append(f" *   - Install onnxruntime: pip install onnxruntime")
        sketch.append(f" *   - Or build from source for your platform")
        sketch.append(f" *   - Define USE_ONNX_RUNTIME before including headers")
        sketch.append(f" *")
        sketch.append(f" * For MCU deployment:")
        sketch.append(f" *   - The ONNX model is embedded as a byte array in the code")
        sketch.append(f" *   - Use ONNX Runtime Micro SDK (contact Microsoft)")
        sketch.append(f" *   - Or convert to TFLite for TFLite Micro runtime")
        sketch.append(f" */")
        sketch.append(f"")

        if self.platform in ('generic_c', 'generic_cpp', 'esp_idf', 'zephyr'):
            return self._generate_linux_edge_sketch(sketch, header_filename)
        else:
            return self._generate_arduino_onnx_sketch(sketch, header_filename)

    def _generate_arduino_onnx_sketch(self, sketch: list, header_filename: str) -> str:
        """Generate Arduino sketch with embedded ONNX model.

        Reuses base class `_get_platform_specific_code()` for IMU init/read
        to ensure consistency with the direct deployment path (same defines,
        same sensor conversion, same I2C init sequence).
        """
        # Reuse the canonical platform-specific IMU code from base class
        platform_code = self._get_platform_specific_code()

        sketch.append(f"#include \"{header_filename}\"")
        sketch.append(f"")
        sketch.append(platform_code['includes'])
        sketch.append(f"")
        sketch.append(platform_code['defines'])
        sketch.append(f"")

        sketch.append(f"float sensor_buffer[WINDOW_SIZE][N_CHANNELS];")
        sketch.append(f"int sample_count = 0;")
        sketch.append(f"")

        # Sliding window
        sketch.append(platform_code['overlap_defines'])
        sketch.append(f"")

        sketch.append(f"// Timing — SAMPLING_RATE is defined in the header")
        sketch.append(f"const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLING_RATE;")
        sketch.append(f"unsigned long last_sample_time = 0;")
        sketch.append(f"")

        sketch.append(f"void setup() {{")
        sketch.append(f"    Serial.begin(115200);")
        sketch.append(f"    while (!Serial && millis() < 3000);")
        sketch.append(f"    Serial.println(\"HAR ONNX Runtime - Initializing...\");")
        sketch.append(f"")

        # IMU init from base (includes Wire.begin, address fallback, diagnostics)
        sketch.append(platform_code['imu_init'])
        sketch.append(f"")

        sketch.append(f"    if (!onnx_init_from_buffer(g_onnx_model, g_onnx_model_len)) {{")
        sketch.append(f"        Serial.println(\"ERROR: ONNX init failed!\");")
        sketch.append(f"        while (1) {{ delay(2000); Serial.println(\"ONNX init failed\"); }}")
        sketch.append(f"    }}")
        sketch.append(f"    onnx_print_info();")
        sketch.append(f"    Serial.println(\"Ready!\");")
        sketch.append(f"}}")
        sketch.append(f"")

        sketch.append(f"void loop() {{")
        sketch.append(f"    unsigned long now = micros();")
        sketch.append(f"    if (now - last_sample_time < SAMPLE_INTERVAL_US) return;")
        sketch.append(f"    last_sample_time = now;")
        sketch.append(f"")

        # Sensor read from base (includes CONVERT_G_TO_MS2 multiplication)
        sketch.append(platform_code['sensor_read'])
        sketch.append(f"")

        sketch.append(f"    // Store in buffer")
        sketch.append(f"    sensor_buffer[sample_count][0] = aX;")
        sketch.append(f"    sensor_buffer[sample_count][1] = aY;")
        sketch.append(f"    sensor_buffer[sample_count][2] = aZ;")
        sketch.append(f"    sensor_buffer[sample_count][3] = gX;")
        sketch.append(f"    sensor_buffer[sample_count][4] = gY;")
        sketch.append(f"    sensor_buffer[sample_count][5] = gZ;")
        sketch.append(f"    sample_count++;")
        sketch.append(f"")

        sketch.append(f"    static const char* last_activity = NULL;")
        sketch.append(f"    if (sample_count >= WINDOW_SIZE) {{")
        sketch.append(f"        float features[NUM_FEATURES];")
        sketch.append(f"        extract_features(sensor_buffer, WINDOW_SIZE, features);")
        sketch.append(f"        scale_features(features);")
        sketch.append(f"")
        sketch.append(f"        float probabilities[NUM_CLASSES];")
        sketch.append(f"        int predicted = onnx_predict(features, probabilities);")
        sketch.append(f"")
        sketch.append(f"        // Debug: print raw probabilities (lines starting with # are ignored by parser)")
        sketch.append(f"        Serial.print(\"# PRED: class=\"); Serial.print(predicted);")
        sketch.append(f"        Serial.print(\" probs=[\");")
        sketch.append(f"        for (int i = 0; i < NUM_CLASSES; i++) {{")
        sketch.append(f"            if (i > 0) Serial.print(\",\");")
        sketch.append(f"            Serial.print(probabilities[i], 4);")
        sketch.append(f"        }}")
        sketch.append(f"        Serial.print(\"] name=\");")
        sketch.append(f"        Serial.println(predicted >= 0 ? onnx_get_class_name(predicted) : \"none\");")
        sketch.append(f"")
        sketch.append(f"        if (predicted >= 0) {{")
        sketch.append(f"            last_activity = onnx_get_class_name(predicted);")
        sketch.append(f"        }}")
        sketch.append(f"")

        # Sliding window shift using base overlap defines
        sketch.append(f"        // Slide window")
        sketch.append(f"        int keep = WINDOW_SIZE - (int)buffer_index_shift;")
        sketch.append(f"        for (int i = 0; i < keep; i++)")
        sketch.append(f"            for (int j = 0; j < N_CHANNELS; j++)")
        sketch.append(f"                sensor_buffer[i][j] = sensor_buffer[i + (int)buffer_index_shift][j];")
        sketch.append(f"        sample_count = keep;")
        sketch.append(f"    }}")
        sketch.append(f"")

        # Sensor CSV output (for Device Test tab compatibility)
        sketch.append(f"    // Sensor CSV output (for Device Test tab live plotting)")
        sketch.append(f"    Serial.print(aX, 4); Serial.print(\",\");")
        sketch.append(f"    Serial.print(aY, 4); Serial.print(\",\");")
        sketch.append(f"    Serial.print(aZ, 4); Serial.print(\",\");")
        sketch.append(f"    Serial.print(gX, 4); Serial.print(\",\");")
        sketch.append(f"    Serial.print(gY, 4); Serial.print(\",\");")
        sketch.append(f"    Serial.print(gZ, 4);")
        sketch.append(f"    if (last_activity != NULL) {{")
        sketch.append(f"        Serial.print(\",\");")
        sketch.append(f"        Serial.print(last_activity);")
        sketch.append(f"    }}")
        sketch.append(f"    Serial.println();")
        sketch.append(f"}}")

        return '\n'.join(sketch)

    def _generate_linux_edge_sketch(self, sketch: list, header_filename: str) -> str:
        """Generate a C++ main() for Linux edge devices (RPi, Jetson, etc.)."""
        sketch.append(f"#define USE_ONNX_RUNTIME")
        sketch.append(f"#include \"{header_filename}\"")
        sketch.append(f"#include <stdio.h>")
        sketch.append(f"#include <stdlib.h>")
        sketch.append(f"#include <string.h>")
        sketch.append(f"#include <time.h>")
        sketch.append(f"")
        sketch.append(f"// For file-based ONNX model loading")
        sketch.append(f"#define ONNX_MODEL_PATH \"har_model.onnx\"")
        sketch.append(f"")
        sketch.append(f"int main(int argc, char* argv[]) {{")
        sketch.append(f"    const char* model_path = (argc > 1) ? argv[1] : ONNX_MODEL_PATH;")
        sketch.append(f"")
        sketch.append(f"    printf(\"HAR ONNX Runtime Inference\\n\");")
        sketch.append(f"    printf(\"Model: %s\\n\", model_path);")
        sketch.append(f"    printf(\"Features: %d, Classes: %d\\n\", NUM_FEATURES, NUM_CLASSES);")
        sketch.append(f"")
        sketch.append(f"    // Initialize ONNX Runtime")
        sketch.append(f"    if (!onnx_init(model_path)) {{")
        sketch.append(f"        fprintf(stderr, \"Failed to initialize ONNX Runtime\\n\");")
        sketch.append(f"        return 1;")
        sketch.append(f"    }}")
        sketch.append(f"")
        sketch.append(f"    // Example: read sensor data from stdin or file")
        sketch.append(f"    float sensor_buffer[WINDOW_SIZE][N_CHANNELS];")
        sketch.append(f"    // TODO: Fill sensor_buffer with real data from your IMU/sensor")
        sketch.append(f"")
        sketch.append(f"    // Extract features")
        sketch.append(f"    float features[NUM_FEATURES];")
        sketch.append(f"    extract_features(sensor_buffer, WINDOW_SIZE, features);")
        sketch.append(f"    scale_features(features);")
        sketch.append(f"")
        sketch.append(f"    // Run inference")
        sketch.append(f"    float probabilities[NUM_CLASSES];")
        sketch.append(f"    int predicted = onnx_predict(features, probabilities);")
        sketch.append(f"")
        sketch.append(f"    if (predicted >= 0) {{")
        sketch.append(f"        printf(\"Predicted: %s (confidence: %.4f)\\n\",")
        sketch.append(f"               onnx_get_class_name(predicted), probabilities[predicted]);")
        sketch.append(f"        for (int i = 0; i < NUM_CLASSES; i++) {{")
        sketch.append(f"            printf(\"  %s: %.4f\\n\", CLASS_NAMES[i], probabilities[i]);")
        sketch.append(f"        }}")
        sketch.append(f"    }}")
        sketch.append(f"")
        sketch.append(f"    onnx_cleanup();")
        sketch.append(f"    return 0;")
        sketch.append(f"}}")

        return '\n'.join(sketch)

    def _generate_placeholder_model(self, error_msg: str) -> str:
        """Generate placeholder when ONNX libraries are not installed."""
        lines = []
        lines.append(f"// ========================================")
        lines.append(f"// ONNX Model Placeholder")
        lines.append(f"// ========================================")
        lines.append(f"// ONNX conversion requires 'onnx' and 'skl2onnx' (sklearn) or 'torch' (PyTorch).")
        lines.append(f"// Install with: pip install onnx skl2onnx")
        lines.append(f"// Error: {error_msg}")
        lines.append(f"//")
        lines.append(f"// After installing, regenerate the code to embed the actual ONNX model.")
        lines.append(f"// ========================================")
        lines.append(f"")
        lines.append(f"alignas(16) const unsigned char g_onnx_model[] = {{0x00}};")
        lines.append(f"const unsigned int g_onnx_model_len = 0;")
        lines.append(f"")
        lines.append(f"#warning \"ONNX model not converted — install onnx/skl2onnx and regenerate\"")
        return '\n'.join(lines)

    def get_onnx_bytes(self) -> Optional[bytes]:
        """Get the raw ONNX model bytes for saving as .onnx file."""
        if self._onnx_bytes is None:
            try:
                self.convert_model()
            except Exception:
                return None
        return self._onnx_bytes
