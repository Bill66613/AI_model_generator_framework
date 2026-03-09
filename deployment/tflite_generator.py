"""
TensorFlow Lite Micro Code Generator

Generates deployment code for TFLite Micro runtime on microcontrollers.
Instead of generating standalone C++ prediction code, this generator:
1. Converts the model to .tflite format
2. Embeds it as a C byte array
3. Generates Arduino/C++ code using the TFLite Micro interpreter API

The feature extraction code is still generated using the same verified
C++ implementation from BaseCodeGenerator (maintaining training-deployment parity).

Requires: tensorflow (for model conversion)
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List

from .base_generator import BaseCodeGenerator

logger = logging.getLogger(__name__)


class TFLiteMicroCodeGenerator(BaseCodeGenerator):
    """
    Code generator for TFLite Micro deployment.

    Generates:
    - model_data.h: TFLite model as C byte array + feature extraction declarations
    - model_data.cpp: Feature extraction + scaling implementation
    - har_tflite.ino: Arduino sketch using TFLite Micro interpreter

    The feature extraction reuses BaseCodeGenerator's verified C++ code,
    ensuring training-deployment parity. Only the model inference step
    differs — using TFLite Micro interpreter instead of hand-coded prediction.
    """

    def __init__(self, model_data: Dict[str, Any], platform: str = 'arduino',
                 optimization: str = 'balanced', overlap: float = 0.5,
                 quantization: str = 'none'):
        # Force quantization to 'none' for base class — TFLite handles its own quantization
        super().__init__(model_data, platform, optimization, overlap, quantization='none')
        self.tflite_quantization = quantization  # Store for TFLite converter
        self._tflite_bytes = None
        self._tflite_c_array = None
        self._arena_size = None

    def convert_model(self) -> bytes:
        """
        Convert the model to TFLite format.

        Returns:
            TFLite model bytes

        Raises:
            ImportError: If tensorflow not installed
        """
        from .converters.tflite_converter import TFLiteConverter

        model_object = self.model_data.get('model_object')
        if model_object is None:
            raise ValueError(
                "TFLite conversion requires the original model object. "
                "Make sure 'model_object' is present in model_data."
            )

        converter = TFLiteConverter(
            model_object=model_object,
            model_type=self.model_type,
            feature_names=self.feature_names,
            classes=self.classes,
            model_params=self.model_data.get('model_params', {})
        )

        self._tflite_bytes = converter.convert(
            quantization=self.tflite_quantization
        )

        # Generate C array
        self._tflite_c_array = converter.to_c_array('g_har_model')

        # Estimate arena size
        self._arena_size = self._estimate_arena_size()

        return self._tflite_bytes

    def _estimate_arena_size(self) -> int:
        """
        Estimate TFLite Micro tensor arena size.

        The arena must hold all intermediate tensors during inference.
        We estimate conservatively based on model type and size.
        """
        n_features = len(self.feature_names)
        n_classes = len(self.classes)

        if self.model_type == 'pytorch_cnn':
            window_size = self.model_data.get('model_params', {}).get(
                'window_size_samples', 150)
            n_channels = self.model_data.get('model_params', {}).get(
                'n_channels', 6)
            # CNN needs more arena: conv buffers + pooling + dense
            # Rough estimate: input + largest intermediate conv output + dense
            input_size = window_size * n_channels * 4
            conv_buffer = window_size * 128 * 4  # Largest conv output (128 filters)
            dense_size = 128 * 4 + 64 * 4 + n_classes * 4
            arena = int((input_size + conv_buffer + dense_size) * 1.5)
        elif self.model_type in ('neural_network', 'pytorch_mlp'):
            # MLP: input features + hidden layers + output
            weights = self.model_data.get('weights', {})
            hidden_size = weights.get('hidden_size', 64) if weights else 64
            arena = (n_features + hidden_size * 2 + n_classes) * 4 * 2
        else:
            # RF/SVM: mostly the input + output tensors
            arena = (n_features + n_classes) * 4 * 4

        # Round up to nearest 1KB, minimum 4KB
        arena = max(arena, 4 * 1024)
        arena = ((arena + 1023) // 1024) * 1024

        return arena

    # --- Abstract method implementations ---

    def _get_model_specific_declarations(self) -> str:
        """TFLite-specific header declarations."""
        lines = []
        lines.append("")
        lines.append("// ---- TFLite Micro Model ----")
        lines.append(f"#define TENSOR_ARENA_SIZE {self._arena_size or 8192}")
        lines.append(f"#define TFLITE_MODEL_SIZE {len(self._tflite_bytes) if self._tflite_bytes else 0}")
        lines.append("")
        lines.append("extern const unsigned char g_har_model[];")
        lines.append("extern const unsigned int g_har_model_len;")
        lines.append("")
        lines.append("// TFLite Micro inference function")
        lines.append(f"int tflite_predict(float features[NUM_FEATURES], float probabilities[NUM_CLASSES]);")
        return '\n'.join(lines)

    def _generate_model_specific_implementation(self) -> str:
        """Generate the TFLite model C byte array."""
        if self._tflite_c_array is None:
            try:
                self.convert_model()
            except (ImportError, ValueError) as e:
                # Generate placeholder if TF not available or model_object missing
                return self._generate_placeholder_model(str(e))

        return self._tflite_c_array

    def _generate_prediction_function(self) -> str:
        """Generate TFLite Micro prediction wrapper."""
        lines = []
        lines.append("// ---- TFLite Micro Prediction ----")
        lines.append("// This function wraps the TFLite Micro interpreter")
        lines.append("// Feature extraction is handled by extract_features() from the base implementation")
        lines.append("")
        lines.append("#include <TensorFlowLite.h>")
        lines.append("#include \"tensorflow/lite/micro/all_ops_resolver.h\"")
        lines.append("#include \"tensorflow/lite/micro/micro_interpreter.h\"")
        lines.append("#include \"tensorflow/lite/schema/schema_generated.h\"")
        lines.append("")
        lines.append("// TFLite Micro globals")
        lines.append("static const tflite::Model* model = nullptr;")
        lines.append("static tflite::MicroInterpreter* interpreter = nullptr;")
        lines.append("static TfLiteTensor* input_tensor = nullptr;")
        lines.append("static TfLiteTensor* output_tensor = nullptr;")
        lines.append(f"static uint8_t tensor_arena[TENSOR_ARENA_SIZE];")
        lines.append("")
        lines.append("bool tflite_init() {")
        lines.append("    // Load model")
        lines.append("    model = tflite::GetModel(g_har_model);")
        lines.append("    if (model->version() != TFLITE_SCHEMA_VERSION) {")
        lines.append("        HAR_LOG(\"Model schema version mismatch!\");")
        lines.append("        return false;")
        lines.append("    }")
        lines.append("")
        lines.append("    // Set up resolver with all ops")
        lines.append("    static tflite::AllOpsResolver resolver;")
        lines.append("")
        lines.append("    // Build interpreter")
        lines.append("    static tflite::MicroInterpreter static_interpreter(")
        lines.append("        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);")
        lines.append("    interpreter = &static_interpreter;")
        lines.append("")
        lines.append("    // Allocate tensors")
        lines.append("    TfLiteStatus allocate_status = interpreter->AllocateTensors();")
        lines.append("    if (allocate_status != kTfLiteOk) {")
        lines.append("        HAR_LOG(\"AllocateTensors() failed\");")
        lines.append("        return false;")
        lines.append("    }")
        lines.append("")
        lines.append("    // Get input and output tensors")
        lines.append("    input_tensor = interpreter->input(0);")
        lines.append("    output_tensor = interpreter->output(0);")
        lines.append("")
        lines.append(f"    HAR_LOG(\"TFLite init OK. Arena used: %d / %d bytes\",")
        lines.append(f"            interpreter->arena_used_bytes(), TENSOR_ARENA_SIZE);")
        lines.append("    return true;")
        lines.append("}")
        lines.append("")
        lines.append(f"int tflite_predict(float features[NUM_FEATURES], float probabilities[NUM_CLASSES]) {{")
        lines.append("    if (interpreter == nullptr) {")
        lines.append("        if (!tflite_init()) return -1;")
        lines.append("    }")
        lines.append("")
        lines.append("    // Copy features to input tensor")
        lines.append("    for (int i = 0; i < NUM_FEATURES; i++) {")
        lines.append("        input_tensor->data.f[i] = features[i];")
        lines.append("    }")
        lines.append("")
        lines.append("    // Run inference")
        lines.append("    TfLiteStatus invoke_status = interpreter->Invoke();")
        lines.append("    if (invoke_status != kTfLiteOk) {")
        lines.append("        HAR_LOG(\"Invoke failed\");")
        lines.append("        return -1;")
        lines.append("    }")
        lines.append("")
        lines.append("    // Copy output probabilities")
        lines.append("    float max_prob = -1.0f;")
        lines.append("    int predicted_class = 0;")
        lines.append("    for (int i = 0; i < NUM_CLASSES; i++) {")
        lines.append("        probabilities[i] = output_tensor->data.f[i];")
        lines.append("        if (probabilities[i] > max_prob) {")
        lines.append("            max_prob = probabilities[i];")
        lines.append("            predicted_class = i;")
        lines.append("        }")
        lines.append("    }")
        lines.append("")
        lines.append("    return predicted_class;")
        lines.append("}")
        return '\n'.join(lines)

    def _generate_utility_functions(self) -> str:
        """Generate utility functions for TFLite deployment."""
        lines = []
        lines.append("")
        lines.append("// ---- TFLite Utility Functions ----")
        lines.append("")
        lines.append("void tflite_print_info() {")
        lines.append(f"    HAR_LOG(\"Model: TFLite Micro ({self.model_type})\");")
        lines.append(f"    HAR_LOG(\"Model size: %u bytes\", g_har_model_len);")
        lines.append(f"    HAR_LOG(\"Arena size: %d bytes\", TENSOR_ARENA_SIZE);")
        lines.append(f"    HAR_LOG(\"Features: %d, Classes: %d\", NUM_FEATURES, NUM_CLASSES);")
        if self._arena_size:
            lines.append(f"    if (interpreter != nullptr) {{")
            lines.append(f"        HAR_LOG(\"Arena used: %d bytes\", interpreter->arena_used_bytes());")
            lines.append(f"    }}")
        lines.append("}")
        lines.append("")
        lines.append("const char* tflite_get_class_name(int class_idx) {")
        lines.append("    if (class_idx < 0 || class_idx >= NUM_CLASSES) return \"unknown\";")
        lines.append("    return CLASS_NAMES[class_idx];")
        lines.append("}")
        return '\n'.join(lines)

    def generate_example_sketch(self, header_filename: str = None) -> str:
        """Generate TFLite Micro Arduino sketch."""
        if header_filename is None:
            header_filename = 'har_model.h'

        # Get model name (strip extension)
        model_name = header_filename.replace('.h', '')

        sketch = []
        sketch.append(f"/*")
        sketch.append(f" * HAR TFLite Micro Deployment")
        sketch.append(f" * Model type: {self.model_type}")
        sketch.append(f" * Classes: {', '.join(self.classes)}")
        sketch.append(f" * Features: {len(self.feature_names)}")
        sketch.append(f" * Deployment approach: TensorFlow Lite for Microcontrollers")
        sketch.append(f" *")
        sketch.append(f" * Generated by HAR Edge Deployment Framework")
        sketch.append(f" *")
        sketch.append(f" * Required Arduino libraries:")
        sketch.append(f" *   - Arduino_TensorFlowLite (install via Library Manager)")
        sketch.append(f" *   - Arduino_LSM6DS3 (for IMU on Seeed XIAO nRF52840 Sense)")
        sketch.append(f" */")
        sketch.append(f"")
        sketch.append(f"#include \"{header_filename}\"")
        sketch.append(f"")

        # Platform-specific IMU include
        if self.platform in ('seeed_xiao', 'arduino'):
            sketch.append(f"#include <LSM6DS3.h>")
            sketch.append(f"#include <Wire.h>")
            sketch.append(f"")
            sketch.append(f"LSM6DS3 imu(I2C_MODE, 0x6A);")
        elif self.platform == 'esp32':
            sketch.append(f"// Configure your IMU library here")
            sketch.append(f"#include <Wire.h>")

        sketch.append(f"")
        sketch.append(f"// Sensor data buffer")
        sketch.append(f"float sensor_buffer[WINDOW_SIZE][N_CHANNELS];")
        sketch.append(f"int sample_count = 0;")
        sketch.append(f"")

        # Calculate stride
        overlap = self.overlap
        stride_str = f"int(WINDOW_SIZE * {1.0 - overlap:.2f})"
        sketch.append(f"// Sliding window stride (overlap = {overlap*100:.0f}%)")
        sketch.append(f"const int STRIDE = {stride_str};")
        sketch.append(f"bool buffer_full = false;")
        sketch.append(f"")
        sketch.append(f"// Timing")
        sampling_rate = self.model_data.get('model_params', {}).get('sampling_rate', 100)
        sketch.append(f"const unsigned long SAMPLE_INTERVAL_US = {int(1000000 / sampling_rate)};  // {sampling_rate} Hz")
        sketch.append(f"unsigned long last_sample_time = 0;")
        sketch.append(f"")

        # Confidence threshold
        sketch.append(f"// Confidence threshold")
        sketch.append(f"const float CONFIDENCE_THRESHOLD = 0.6f;")
        sketch.append(f"")

        # Setup function
        sketch.append(f"void setup() {{")
        sketch.append(f"    Serial.begin(115200);")
        sketch.append(f"    while (!Serial && millis() < 3000);")
        sketch.append(f"")
        sketch.append(f"    Serial.println(\"HAR TFLite Micro - Initializing...\");")
        sketch.append(f"")

        if self.platform in ('seeed_xiao', 'arduino'):
            sketch.append(f"    // Initialize IMU")
            sketch.append(f"    if (imu.begin() != 0) {{")
            sketch.append(f"        Serial.println(\"ERROR: IMU initialization failed!\");")
            sketch.append(f"        while (1);")
            sketch.append(f"    }}")
        else:
            sketch.append(f"    // Initialize your IMU here")

        sketch.append(f"")
        sketch.append(f"    // Initialize TFLite Micro")
        sketch.append(f"    if (!tflite_init()) {{")
        sketch.append(f"        Serial.println(\"ERROR: TFLite initialization failed!\");")
        sketch.append(f"        while (1);")
        sketch.append(f"    }}")
        sketch.append(f"")
        sketch.append(f"    tflite_print_info();")
        sketch.append(f"    Serial.println(\"Ready! Collecting sensor data...\");")
        sketch.append(f"}}")
        sketch.append(f"")

        # Loop function
        sketch.append(f"void loop() {{")
        sketch.append(f"    unsigned long now = micros();")
        sketch.append(f"    if (now - last_sample_time < SAMPLE_INTERVAL_US) return;")
        sketch.append(f"    last_sample_time = now;")
        sketch.append(f"")
        sketch.append(f"    // Read IMU data")
        sketch.append(f"    float aX, aY, aZ, gX, gY, gZ;")

        if self.platform in ('seeed_xiao', 'arduino'):
            sketch.append(f"    aX = imu.readFloatAccelX();")
            sketch.append(f"    aY = imu.readFloatAccelY();")
            sketch.append(f"    aZ = imu.readFloatAccelZ();")
            sketch.append(f"    gX = imu.readFloatGyroX();")
            sketch.append(f"    gY = imu.readFloatGyroY();")
            sketch.append(f"    gZ = imu.readFloatGyroZ();")
        else:
            sketch.append(f"    // TODO: Read from your IMU")
            sketch.append(f"    // aX = ...; aY = ...; aZ = ...;")
            sketch.append(f"    // gX = ...; gY = ...; gZ = ...;")

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

        sketch.append(f"    // Check if window is full")
        sketch.append(f"    if (sample_count >= WINDOW_SIZE) {{")
        sketch.append(f"        // Extract features")
        sketch.append(f"        float features[NUM_FEATURES];")
        sketch.append(f"        extract_features(sensor_buffer, WINDOW_SIZE, features);")
        sketch.append(f"")
        sketch.append(f"        // Scale features")
        sketch.append(f"        scale_features(features);")
        sketch.append(f"")
        sketch.append(f"        // Run TFLite Micro inference")
        sketch.append(f"        float probabilities[NUM_CLASSES];")
        sketch.append(f"        int predicted = tflite_predict(features, probabilities);")
        sketch.append(f"")
        sketch.append(f"        if (predicted >= 0) {{")
        sketch.append(f"            float confidence = probabilities[predicted];")
        sketch.append(f"            const char* class_name = tflite_get_class_name(predicted);")
        sketch.append(f"")
        sketch.append(f"            if (confidence >= CONFIDENCE_THRESHOLD) {{")
        sketch.append(f"                Serial.print(\"PREDICTION:\");")
        sketch.append(f"                Serial.print(class_name);")
        sketch.append(f"                Serial.print(\":\");")
        sketch.append(f"                Serial.println(confidence, 4);")
        sketch.append(f"            }} else {{")
        sketch.append(f"                Serial.print(\"LOW_CONF:\");")
        sketch.append(f"                Serial.print(class_name);")
        sketch.append(f"                Serial.print(\":\");")
        sketch.append(f"                Serial.println(confidence, 4);")
        sketch.append(f"            }}")
        sketch.append(f"        }}")
        sketch.append(f"")
        sketch.append(f"        // Slide window by STRIDE")
        sketch.append(f"        int shift = STRIDE;")
        sketch.append(f"        for (int i = 0; i < WINDOW_SIZE - shift; i++) {{")
        sketch.append(f"            for (int j = 0; j < N_CHANNELS; j++) {{")
        sketch.append(f"                sensor_buffer[i][j] = sensor_buffer[i + shift][j];")
        sketch.append(f"            }}")
        sketch.append(f"        }}")
        sketch.append(f"        sample_count = WINDOW_SIZE - shift;")
        sketch.append(f"    }}")
        sketch.append(f"}}")

        return '\n'.join(sketch)

    def _generate_placeholder_model(self, error_msg: str) -> str:
        """Generate placeholder when TF is not installed."""
        lines = []
        lines.append(f"// ========================================")
        lines.append(f"// TFLite Model Placeholder")
        lines.append(f"// ========================================")
        lines.append(f"// TensorFlow is required to convert the model to TFLite format.")
        lines.append(f"// Install with: pip install tensorflow")
        lines.append(f"// Error: {error_msg}")
        lines.append(f"//")
        lines.append(f"// After installing TensorFlow, regenerate the code to get")
        lines.append(f"// the actual model byte array embedded here.")
        lines.append(f"// ========================================")
        lines.append(f"")
        lines.append(f"// Placeholder model array — replace with actual conversion output")
        lines.append(f"alignas(16) const unsigned char g_har_model[] = {{0x00}};")
        lines.append(f"const unsigned int g_har_model_len = 0;")
        lines.append(f"")
        lines.append(f"#warning \"TFLite model not converted — install tensorflow and regenerate\"")
        return '\n'.join(lines)
