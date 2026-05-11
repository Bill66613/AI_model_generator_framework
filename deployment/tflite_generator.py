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
                 quantization: str = 'none',
                 confidence_threshold: float = 0.6,
                 smoothing_window: int = 1,
                 enable_iir_filter: bool = False,
                 enable_kalman_filter: bool = False):
        # Pass actual quantization to base so header macros are correct
        super().__init__(model_data, platform, optimization, overlap, quantization=quantization,
                         confidence_threshold=confidence_threshold,
                         smoothing_window=smoothing_window,
                         enable_iir_filter=enable_iir_filter,
                         enable_kalman_filter=enable_kalman_filter)
        self.tflite_quantization = quantization  # Store for TFLite converter
        self._tflite_bytes = None
        self._tflite_c_array = None
        self._tflite_ops = None  # Populated by convert_model() via enumerate_ops()
        self._arena_size = None
        # Set to True when enumerate_ops() hits an unknown op
        self._use_all_ops_resolver = False

    def _is_esp32_platform(self) -> bool:
        """Check if target platform is ESP32-based (limited DRAM)."""
        return self.platform in ('esp32', 'm5stack', 'esp_idf')

    def _generate_arena_declaration(self) -> str:
        """Generate tensor arena declaration, platform-aware.

        ESP32/M5Stack: heap-allocate the arena (preferring PSRAM when
        available) to avoid exhausting the ~320 KB DRAM with a large BSS
        array.  Other platforms: simple static array.
        """
        if self._is_esp32_platform():
            return (
                "// Tensor arena — heap-allocated to avoid DRAM overflow on ESP32.\n"
                "// Prefers PSRAM (ps_malloc) when available; falls back to regular malloc.\n"
                "static uint8_t* tensor_arena = nullptr;\n"
                "\n"
                "static bool allocate_arena() {\n"
                "    if (tensor_arena != nullptr) return true;\n"
                "#if defined(BOARD_HAS_PSRAM) || defined(ESP_PSRAM_FOUND)\n"
                "    tensor_arena = (uint8_t*)ps_malloc(TENSOR_ARENA_SIZE);\n"
                "    if (tensor_arena) { HAR_LOG(\"Arena: PSRAM (%d bytes)\", TENSOR_ARENA_SIZE); return true; }\n"
                "#endif\n"
                "    tensor_arena = (uint8_t*)malloc(TENSOR_ARENA_SIZE);\n"
                "    if (tensor_arena) { HAR_LOG(\"Arena: heap (%d bytes)\", TENSOR_ARENA_SIZE); return true; }\n"
                "    HAR_LOG(\"ERROR: arena allocation failed (%d bytes)\", TENSOR_ARENA_SIZE);\n"
                "    return false;\n"
                "}\n"
            )
        return f"static uint8_t tensor_arena[TENSOR_ARENA_SIZE];\n"

    def _generate_arena_init_check(self) -> list:
        """Generate arena allocation call at the start of tflite_init() for ESP32."""
        if self._is_esp32_platform():
            return [
                "    // Allocate tensor arena from heap / PSRAM",
                "    if (!allocate_arena()) return false;",
                "",
            ]
        return []

    def _load_representative_data(self):
        """Load real training data for INT8 quantization calibration.

        For CNN models, loads the raw_train.npy file from the training directory.
        For MLP/other models, loads features from training CSV.
        Returns None if no data available (converter will fall back to synthetic).
        """
        import os
        import glob

        if self.tflite_quantization == 'none':
            return None

        model_info = self.model_data.get('model_info', {})
        training_dir = self.model_data.get('training_dir', '')

        # Try to find training dir from model_info or model_params
        if not training_dir:
            training_dir = model_info.get('training_dir', '')
        if not training_dir:
            model_params = self.model_data.get('model_params', {})
            training_dir = model_params.get('training_dir', '')

        if not training_dir or not os.path.isdir(training_dir):
            return None

        try:
            if self.model_type == 'pytorch_cnn':
                # Load raw sensor windows: shape (n_windows, window_size, n_channels)
                raw_files = glob.glob(os.path.join(
                    training_dir, '*_raw_train.npy'))
                if raw_files:
                    data = np.load(raw_files[0])
                    # Use up to 200 samples for calibration
                    if len(data) > 200:
                        indices = np.random.default_rng(42).choice(
                            len(data), 200, replace=False)
                        data = data[indices]
                    logger.info(
                        f"Loaded {len(data)} real CNN windows for INT8 calibration")
                    return data.astype(np.float32)
            else:
                # MLP/RF/SVM: load feature CSV matching model's feature count
                import pandas as pd
                train_files = glob.glob(
                    os.path.join(training_dir, '*_train.csv'))
                num_features = len(
                    self.feature_names) if self.feature_names else 0
                df = None
                for tf in sorted(train_files, reverse=True):
                    candidate = pd.read_csv(tf, nrows=1)
                    feat_cols = [c for c in candidate.columns if c != 'label']
                    if num_features == 0 or len(feat_cols) == num_features:
                        df = pd.read_csv(tf)
                        break
                if df is not None:
                    feature_cols = [c for c in df.columns if c != 'label']
                    data = df[feature_cols].values
                    if len(data) > 200:
                        indices = np.random.default_rng(42).choice(
                            len(data), 200, replace=False)
                        data = data[indices]
                    # Apply StandardScaler: the TFLite model expects scaled
                    # inputs (same as sklearn pipeline). Raw features have
                    # ranges like [0, 9627] but the model sees ~N(0,1).
                    means = np.array(self.feature_means, dtype=np.float32)
                    stds = np.array(self.feature_stds, dtype=np.float32)
                    stds[stds < 1e-7] = 1.0  # prevent division by zero
                    data = (data - means) / stds
                    logger.info(
                        f"Loaded {len(data)} real feature samples for INT8 calibration (scaled)")
                    return data.astype(np.float32)
        except Exception as e:
            logger.warning(f"Could not load representative data: {e}")

        return None

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

        # Enrich model_params with the correct window_size_samples computed
        # by the base generator from fe_config.  The converter needs this to
        # build the Keras model with the right input shape — a mismatch causes
        # a RESHAPE error during TFLite conversion/quantization calibration.
        converter_params = dict(self.model_data.get('model_params', {}))
        converter_params['window_size_samples'] = self.window_size
        converter_params['n_channels'] = 6  # default, may be overridden below

        # Propagate n_channels from fe_config or model_info
        model_info = self.model_data.get('model_info', {})
        fe_config = model_info.get('fe_config', {})
        n_ch = (fe_config.get('num_channels')
                or len(fe_config.get('sensor_columns', []))
                or converter_params.get('n_channels', 6))
        converter_params['n_channels'] = n_ch

        converter = TFLiteConverter(
            model_object=model_object,
            model_type=self.model_type,
            feature_names=self.feature_names,
            classes=self.classes,
            model_params=converter_params
        )

        # Load real representative data for INT8 quantization calibration.
        # CNN models are especially sensitive: N(0,1) calibration data causes
        # the quantization range to be too narrow for real IMU values (~9.8 m/s²),
        # resulting in near-uniform (garbage) output probabilities on-device.
        representative_data = self._load_representative_data()

        self._tflite_bytes = converter.convert(
            quantization=self.tflite_quantization,
            representative_data=representative_data
        )

        # Generate C array (platform-aware: uses PROGMEM on ESP32 to keep DRAM free)
        self._tflite_c_array = converter.to_c_array('g_har_model', platform=self.platform)

        # Enumerate actual ops used in the model for resolver generation
        self._tflite_ops = converter.enumerate_ops()
        if self._tflite_ops is None:
            # enumerate_ops() hit an unknown op — fall back to AllOpsResolver
            self._use_all_ops_resolver = True

        # Estimate arena size
        self._arena_size = self._estimate_arena_size()

        return self._tflite_bytes

    def _estimate_arena_size(self) -> int:
        """
        Estimate TFLite Micro tensor arena size.

        The arena must hold all intermediate tensors during inference.
        For INT8 quantized models, internal tensors are 1 byte (not 4),
        so the arena can be much smaller than for float32 models.
        """
        n_features = len(self.feature_names)
        n_classes = len(self.classes)
        # Bytes per value: 1 for INT8 internal tensors, 2 for INT16, 4 for float32
        if self.tflite_quantization == 'int16':
            bpv = 2
        elif self.tflite_quantization == 'int8':
            bpv = 1
        else:
            bpv = 4

        if self.model_type == 'pytorch_cnn':
            window_size = self.model_data.get('model_params', {}).get(
                'window_size_samples', 150)
            n_channels = self.model_data.get('model_params', {}).get(
                'n_channels', 6)
            # I/O tensors are float32 even for quantized models (float32 I/O mode)
            input_size = window_size * n_channels * 4
            output_size = n_classes * 4
            # Peak activation: two conv layers alive simultaneously (INT8)
            peak_activation = window_size * 128 * bpv * 2
            # Conv2D im2col scratch: ONE buffer reused across all conv layers
            # Largest is conv2: kernel=5, in_channels=32 → 5*32*window_size
            im2col_scratch = window_size * 5 * 32 * bpv
            # Dense layers (small)
            dense_size = (128 + 64 + n_classes) * bpv
            # Quantize/Dequantize overhead + TFLite bookkeeping
            overhead = 4096
            arena = int((input_size + output_size + peak_activation +
                        im2col_scratch + dense_size + overhead) * 1.3)
        elif self.model_type in ('neural_network', 'pytorch_mlp'):
            # MLP: input features + hidden layers + output
            weights = self.model_data.get('weights', {})
            hidden_size = weights.get('hidden_size', 64) if weights else 64
            arena = (n_features * 4 + hidden_size *
                     2 * bpv + n_classes * 4) * 2
        else:
            # RF/SVM: mostly the input + output tensors
            arena = (n_features + n_classes) * 4 * 4

        # Round up to nearest 1KB, minimum 8KB, maximum 200KB
        arena = max(arena, 8 * 1024)
        arena = min(arena, 200 * 1024)
        arena = ((arena + 1023) // 1024) * 1024

        return arena

    # --- Overrides for CNN TFLite ---

    def generate_implementation(self, header_filename=None) -> str:
        """Override: CNN TFLite skips feature extraction; MLP uses base with har_predict_internal wrapper."""
        if self.model_type == 'pytorch_cnn':
            return self._generate_cnn_tflite_implementation(header_filename)
        return super().generate_implementation(header_filename)

    def _get_impl_includes(self) -> str:
        """Add TFLite Micro includes at the top level of the .cpp file."""
        base_includes = super()._get_impl_includes()
        if self._use_all_ops_resolver:
            resolver_include = '#include "tensorflow/lite/micro/all_ops_resolver.h"'
        else:
            resolver_include = '#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"'
        tflite_includes = (
            '#include <TensorFlowLite.h>\n'
            '#include "tensorflow/lite/micro/micro_interpreter.h"\n'
            '#include "tensorflow/lite/schema/schema_generated.h"\n'
            f'{resolver_include}'
        )
        return f"{base_includes}\n{tflite_includes}"

    def _generate_cnn_tflite_implementation(self, header_filename=None) -> str:
        """CNN TFLite implementation: TFLite byte array + raw-window inference only.
        No feature extraction or scaling arrays — CNN operates directly on raw IMU windows.
        """
        import os
        header_include = os.path.basename(
            header_filename) if header_filename else 'har_model.h'
        classes_str = ', '.join([f'"{cls}"' for cls in self.classes])
        impl_lines = [
            f'/*',
            f' * HAR CNN TFLite Micro Implementation',
            f' * CNN operates on raw sensor windows — no feature extraction or scaling needed.',
            f' * Model Type: {self.model_type}  |  Platform: {self.platform}',
            f' */',
            f'',
            f'#include "{header_include}"',
            self._get_impl_includes(),
            f'',
            f'// Activity class names',
            f'const char* activity_names[NUM_CLASSES] = {{',
            f'    {classes_str}',
            f'}};',
            f'',
            self._generate_iir_filter_implementation(),
            self._generate_kalman_filter_implementation(),
            self._generate_model_specific_implementation(),
            f'',
            f'void har_init() {{',
            f'    // Lazy init — TFLite initialises on first prediction call',
            f'}}',
            f'',
            self._generate_prediction_function(),
            f'',
            f'const char* get_activity_name(int class_id) {{',
            f'    if (class_id >= 0 && class_id < NUM_CLASSES) {{',
            f'        return activity_names[class_id];',
            f'    }}',
            f'    return "unknown";',
            f'}}',
            f'',
            self._generate_utility_functions(),
        ]
        return '\n'.join(impl_lines).strip()

    def _get_function_declarations(self) -> str:
        """Override: CNN TFLite exposes har_predict_from_window instead of har_predict."""
        if self.model_type == 'pytorch_cnn':
            return (
                'void har_init();\n'
                'int har_predict_from_window(float sensor_data[WINDOW_SIZE][N_CHANNELS], float* confidence);\n'
                'const char* get_activity_name(int class_id);'
            )
        return super()._get_function_declarations()

    # --- Abstract method implementations ---

    def _get_model_specific_declarations(self) -> str:
        """TFLite-specific header declarations."""
        # Trigger lazy conversion so _tflite_bytes and _arena_size are populated
        if self._tflite_bytes is None:
            self.convert_model()  # Let exceptions propagate

        lines = []
        lines.append("")
        lines.append("// ---- TFLite Micro Model ----")
        arena_size = self._arena_size if self._arena_size is not None else self._estimate_arena_size()
        lines.append(f"#define TENSOR_ARENA_SIZE {arena_size}")
        lines.append(
            f"#define TFLITE_MODEL_SIZE {len(self._tflite_bytes) if self._tflite_bytes else 0}")
        lines.append("")
        lines.append("extern const unsigned char g_har_model[];")
        lines.append("extern const unsigned int g_har_model_len;")
        lines.append("")
        if self.model_type == 'pytorch_cnn':
            lines.append(
                "// TFLite Micro inference (CNN: raw sensor window input)")
            lines.append("bool tflite_init();")
            lines.append("void tflite_print_info();")
            lines.append(
                f"int tflite_predict_window(float sensor_data[WINDOW_SIZE][N_CHANNELS], float probabilities[NUM_CLASSES]);")
        else:
            lines.append("// TFLite Micro inference function")
            lines.append("bool tflite_init();")
            lines.append("void tflite_print_info();")
            lines.append(
                f"int tflite_predict(float features[NUM_FEATURES], float probabilities[NUM_CLASSES]);")
        return '\n'.join(lines)

    def _generate_model_specific_implementation(self) -> str:
        """Generate the TFLite model C byte array.

        Raises ImportError/ValueError if conversion fails — caller must handle.
        """
        if self._tflite_c_array is None:
            self.convert_model()  # Let exceptions propagate — UI will show failure

        return self._tflite_c_array

    def _generate_prediction_function(self) -> str:
        """Generate TFLite Micro prediction wrapper."""
        if self.model_type == 'pytorch_cnn':
            return self._generate_cnn_tflite_predict()
        lines = []
        lines.append("// ---- TFLite Micro Prediction ----")
        lines.append(
            "// Feature extraction is handled by extract_features() from the base implementation")
        lines.append("")
        lines.append("// TFLite Micro globals (headers included at the top of this file)")
        lines.append("static const tflite::Model* model = nullptr;")
        lines.append("static tflite::MicroInterpreter* interpreter = nullptr;")
        lines.append("static TfLiteTensor* input_tensor = nullptr;")
        lines.append("static TfLiteTensor* output_tensor = nullptr;")
        lines.append(self._generate_arena_declaration())
        lines.append("")
        lines.append("bool tflite_init() {")
        lines.extend(self._generate_arena_init_check())
        lines.append("    // Load model")
        lines.append("    model = tflite::GetModel(g_har_model);")
        lines.append("    if (model->version() != TFLITE_SCHEMA_VERSION) {")
        lines.append("        HAR_LOG(\"Model schema version mismatch!\");")
        lines.append("        return false;")
        lines.append("    }")
        lines.extend(self._generate_resolver_code())
        lines.append("")
        lines.append("    // Build interpreter")
        lines.append("    static tflite::MicroInterpreter static_interpreter(")
        lines.append(
            "        model, resolver, tensor_arena, TENSOR_ARENA_SIZE, nullptr);")
        lines.append("    interpreter = &static_interpreter;")
        lines.append("")
        lines.append("    // Allocate tensors")
        lines.append(
            "    TfLiteStatus allocate_status = interpreter->AllocateTensors();")
        lines.append("    if (allocate_status != kTfLiteOk) {")
        lines.append("        HAR_LOG(\"AllocateTensors() failed\");")
        lines.append("        return false;")
        lines.append("    }")
        lines.append("")
        lines.append("    // Get input and output tensors")
        lines.append("    input_tensor = interpreter->input(0);")
        lines.append("    output_tensor = interpreter->output(0);")
        lines.append("")
        lines.append(
            f"    HAR_LOG(\"TFLite init OK. Arena used: %d / %d bytes\",")
        lines.append(
            f"            interpreter->arena_used_bytes(), TENSOR_ARENA_SIZE);")
        lines.append("    return true;")
        lines.append("}")
        lines.append("")
        lines.append(
            f"int tflite_predict(float features[NUM_FEATURES], float probabilities[NUM_CLASSES]) {{")
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
        lines.append("")
        # Bridge: base class har_predict() calls har_predict_internal() → delegate to tflite_predict()
        lines.append(
            "// har_predict_internal: bridges base har_predict() → tflite_predict()")
        lines.append(
            f"int har_predict_internal(float features[NUM_FEATURES], float probs_out[NUM_CLASSES]) {{")
        lines.append("    return tflite_predict(features, probs_out);")
        lines.append("}")
        return '\n'.join(lines)

    def _generate_resolver_code(self) -> list:
        """Generate MicroMutableOpResolver registration code based on actual model ops.

        If the model uses an op that isn't yet in BUILTIN_OP_NAMES (enumerate_ops
        returns None), we fall back to AllOpsResolver for safe compilation.
        AllOpsResolver is larger but guarantees no missing-op runtime failures.
        """
        ops = self._tflite_ops
        # _use_all_ops_resolver is set when enumerate_ops() hit an unknown op.
        if self._use_all_ops_resolver:
            # Header already added by _get_impl_includes() — only emit the declaration here.
            lines = []
            lines.append(
                "    // WARNING: model contains an unrecognized op code.")
            lines.append(
                "    // AllOpsResolver is used for safe compilation; for a smaller binary,")
            lines.append(
                "    // add the missing op to BUILTIN_OP_NAMES in tflite_converter.py.")
            lines.append("    static tflite::AllOpsResolver resolver;")
            return lines

        if not ops:
            # Fallback if convert_model() wasn't called yet
            if self.model_type == 'pytorch_cnn':
                ops = ['Conv2D', 'Dequantize', 'ExpandDims', 'FullyConnected',
                       'MaxPool2D', 'Mean', 'Quantize', 'Reshape', 'Softmax']
            else:
                ops = ['Dequantize', 'FullyConnected', 'Quantize', 'Softmax']

        lines = []
        lines.append(
            f"    // Register exactly the {len(ops)} ops used by this model")
        lines.append(
            f"    static tflite::MicroMutableOpResolver<{len(ops)}> resolver;")
        for op in sorted(ops):
            lines.append(f"    resolver.Add{op}();")
        return lines

    def _generate_cnn_tflite_predict(self) -> str:
        """Generate TFLite Micro inference for CNN (raw sensor window input)."""
        lines = []
        lines.append(
            "// ---- TFLite Micro Prediction (CNN: raw sensor window input) ----")
        lines.append(
            "// CNN operates directly on raw IMU windows — no feature extraction needed.")
        lines.append("")
        lines.append("// TFLite Micro globals (headers included at the top of this file)")
        lines.append("static const tflite::Model* model = nullptr;")
        lines.append("static tflite::MicroInterpreter* interpreter = nullptr;")
        lines.append("static TfLiteTensor* input_tensor = nullptr;")
        lines.append("static TfLiteTensor* output_tensor = nullptr;")
        lines.append(self._generate_arena_declaration())
        lines.append("")
        lines.append("bool tflite_init() {")
        lines.extend(self._generate_arena_init_check())
        lines.append("    model = tflite::GetModel(g_har_model);")
        lines.append("    if (model->version() != TFLITE_SCHEMA_VERSION) {")
        lines.append("        HAR_LOG(\"Model schema version mismatch!\");")
        lines.append("        return false;")
        lines.append("    }")
        lines.extend(self._generate_resolver_code())
        lines.append("    static tflite::MicroInterpreter static_interpreter(")
        lines.append(
            "        model, resolver, tensor_arena, TENSOR_ARENA_SIZE, nullptr);")
        lines.append("    interpreter = &static_interpreter;")
        lines.append(
            "    TfLiteStatus allocate_status = interpreter->AllocateTensors();")
        lines.append("    if (allocate_status != kTfLiteOk) {")
        lines.append("        HAR_LOG(\"AllocateTensors() failed\");")
        lines.append("        return false;")
        lines.append("    }")
        lines.append("    input_tensor = interpreter->input(0);")
        lines.append("    output_tensor = interpreter->output(0);")
        lines.append(
            f"    HAR_LOG(\"TFLite CNN init OK. Arena used: %d / %d bytes\",")
        lines.append(
            f"            interpreter->arena_used_bytes(), TENSOR_ARENA_SIZE);")
        lines.append("    return true;")
        lines.append("}")
        lines.append("")
        lines.append("// CNN TFLite inference from raw sensor window.")
        lines.append(
            "// Input: sensor_data[WINDOW_SIZE][N_CHANNELS] — raw IMU readings.")
        lines.append(
            f"int tflite_predict_window(float sensor_data[WINDOW_SIZE][N_CHANNELS], float probabilities[NUM_CLASSES]) {{")
        lines.append("    if (interpreter == nullptr) {")
        lines.append("        if (!tflite_init()) return -1;")
        lines.append("    }")
        lines.append(
            "    // Flatten [WINDOW_SIZE][N_CHANNELS] into the TFLite input tensor (shape: 1,WINDOW_SIZE,N_CHANNELS)")
        lines.append("    int idx = 0;")
        lines.append("    for (int t = 0; t < WINDOW_SIZE; t++) {")
        lines.append("        for (int ch = 0; ch < N_CHANNELS; ch++) {")
        lines.append(
            "            input_tensor->data.f[idx++] = sensor_data[t][ch];")
        lines.append("        }")
        lines.append("    }")
        lines.append("    TfLiteStatus invoke_status = interpreter->Invoke();")
        lines.append("    if (invoke_status != kTfLiteOk) {")
        lines.append("        HAR_LOG(\"Invoke failed\");")
        lines.append("        return -1;")
        lines.append("    }")
        lines.append("    float max_prob = -1.0f;")
        lines.append("    int predicted_class = 0;")
        lines.append("    for (int i = 0; i < NUM_CLASSES; i++) {")
        lines.append("        probabilities[i] = output_tensor->data.f[i];")
        lines.append("        if (probabilities[i] > max_prob) {")
        lines.append("            max_prob = probabilities[i];")
        lines.append("            predicted_class = i;")
        lines.append("        }")
        lines.append("    }")
        lines.append("    return predicted_class;")
        lines.append("}")
        lines.append("")
        lines.append(
            "// High-level window prediction with confidence threshold.")
        lines.append(
            "// Returns predicted class id, or -1 if below CONFIDENCE_THRESHOLD.")
        lines.append(
            f"int har_predict_from_window(float sensor_data[WINDOW_SIZE][N_CHANNELS], float* confidence) {{")
        lines.append("    if (sensor_data == NULL) {")
        lines.append("        if (confidence) *confidence = 0.0f;")
        lines.append("        return -1;")
        lines.append("    }")
        lines.append("    float probabilities[NUM_CLASSES];")
        lines.append(
            "    int predicted = tflite_predict_window(sensor_data, probabilities);")
        lines.append("    float conf = 0.0f;")
        lines.append("    for (int i = 0; i < NUM_CLASSES; i++) {")
        lines.append(
            "        if (probabilities[i] > conf) conf = probabilities[i];")
        lines.append("    }")
        lines.append("    if (confidence) *confidence = conf;")
        lines.append("    if (predicted < 0 || predicted >= NUM_CLASSES) {")
        lines.append("        if (confidence) *confidence = 0.0f;")
        lines.append("        return -1;")
        lines.append("    }")
        lines.append("    if (conf < CONFIDENCE_THRESHOLD) return -1;")
        lines.append("    return predicted;")
        lines.append("}")
        return '\n'.join(lines)

    def _generate_utility_functions(self) -> str:
        """Generate utility functions for TFLite deployment."""
        lines = []
        lines.append("")
        lines.append("// ---- TFLite Utility Functions ----")
        lines.append("")
        lines.append("void tflite_print_info() {")
        lines.append(
            f"    HAR_LOG(\"Model: TFLite Micro ({self.model_type})\");")
        lines.append(
            f"    HAR_LOG(\"Model size: %u bytes\", g_har_model_len);")
        lines.append(
            f"    HAR_LOG(\"Arena size: %d bytes\", TENSOR_ARENA_SIZE);")
        lines.append(
            f"    HAR_LOG(\"Features: %d, Classes: %d\", NUM_FEATURES, NUM_CLASSES);")
        if self._arena_size:
            lines.append(f"    if (interpreter != nullptr) {{")
            lines.append(
                f"        HAR_LOG(\"Arena used: %d bytes\", interpreter->arena_used_bytes());")
            lines.append(f"    }}")
        lines.append("}")
        lines.append("")
        lines.append("const char* tflite_get_class_name(int class_idx) {")
        lines.append(
            "    if (class_idx < 0 || class_idx >= NUM_CLASSES) return \"unknown\";")
        lines.append("    return activity_names[class_idx];")
        lines.append("}")
        return '\n'.join(lines)

    def generate_example_sketch(self, header_filename: str = None) -> str:
        """Generate TFLite Micro Arduino sketch.

        Reuses base class `_get_platform_specific_code()` for IMU init/read
        to ensure consistency with the direct deployment path (same defines,
        same sensor conversion, same I2C init sequence).
        """
        if header_filename is None:
            header_filename = 'har_model.h'

        # Reuse the canonical platform-specific IMU code from base class
        platform_code = self._get_platform_specific_code()

        sketch = []
        sketch.append(f"/*")
        sketch.append(f" * HAR TFLite Micro Deployment")
        sketch.append(f" * Model type: {self.model_type}")
        sketch.append(f" * Classes: {', '.join(self.classes)}")
        sketch.append(f" * Features: {len(self.feature_names)}")
        sketch.append(
            f" * Deployment approach: TensorFlow Lite for Microcontrollers")
        sketch.append(f" *")
        sketch.append(f" * Generated by HAR Edge Deployment Framework")
        sketch.append(f" */")
        sketch.append(f"")
        sketch.append(f"#include \"{header_filename}\"")
        sketch.append(f"")

        # Platform includes and defines from base (includes Wire.begin, CONVERT_G_TO_MS2, etc.)
        sketch.append(platform_code['includes'])
        sketch.append(f"")
        sketch.append(platform_code['defines'])
        sketch.append(f"")

        sketch.append(f"// Sensor data buffer")
        sketch.append(f"float sensor_buffer[WINDOW_SIZE][N_CHANNELS];")
        sketch.append(f"int sample_count = 0;")
        sketch.append(f"")

        # Sliding window
        overlap = self.overlap
        sketch.append(f"// Sliding window (overlap = {overlap*100:.0f}%)")
        sketch.append(platform_code['overlap_defines'])
        sketch.append(f"")

        # Timing — use SAMPLING_RATE from header
        sketch.append(f"// Timing — SAMPLING_RATE is defined in the header")
        sketch.append(
            f"const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLING_RATE;")
        sketch.append(f"unsigned long last_sample_time = 0;")
        sketch.append(f"")

        sketch.append(
            f"// CONFIDENCE_THRESHOLD is defined in the header (default: {self.confidence_threshold:.2f})")
        sketch.append(f"")

        # Setup function
        sketch.append(f"void setup() {{")
        sketch.append(f"    Serial.begin(115200);")
        sketch.append(f"    while (!Serial && millis() < 3000);")
        sketch.append(f"")
        sketch.append(
            f"    Serial.println(\"HAR TFLite Micro - Initializing...\");")
        sketch.append(f"")

        # IMU init from base (includes Wire.begin, address fallback, diagnostics)
        sketch.append(platform_code['imu_init'])
        sketch.append(f"")

        sketch.append(f"    // Initialize TFLite Micro")
        sketch.append(f"    if (!tflite_init()) {{")
        sketch.append(
            f"        Serial.println(\"ERROR: TFLite initialization failed! Check TENSOR_ARENA_SIZE.\");")
        sketch.append(
            f"        while (1) {{ delay(2000); Serial.println(\"TFLite init failed\"); }}")
        sketch.append(f"    }}")
        sketch.append(f"")
        sketch.append(f"    tflite_print_info();")
        sketch.append(
            f"    Serial.println(\"Ready! Collecting sensor data...\");")
        sketch.append(f"}}")
        sketch.append(f"")

        # Loop function
        sketch.append(f"void loop() {{")
        sketch.append(f"    unsigned long now = micros();")
        sketch.append(
            f"    if (now - last_sample_time < SAMPLE_INTERVAL_US) return;")
        sketch.append(f"    last_sample_time = now;")
        sketch.append(f"")

        # Sensor read from base (includes CONVERT_G_TO_MS2 multiplication)
        sketch.append(platform_code['sensor_read'])
        sketch.append(f"")

        # On-device IIR filter (matches base generator behavior)
        if self.enable_iir_filter:
            sketch.append(f"    // Apply on-device IIR filter")
            sketch.append(f"    #ifdef IIR_FILTER_ENABLED")
            sketch.append(f"    {{")
            sketch.append(
                f"        float raw_sample[N_CHANNELS] = {{aX, aY, aZ, gX, gY, gZ}};")
            sketch.append(f"        iir_filter_sample(raw_sample);")
            sketch.append(
                f"        aX = raw_sample[0]; aY = raw_sample[1]; aZ = raw_sample[2];")
            sketch.append(
                f"        gX = raw_sample[3]; gY = raw_sample[4]; gZ = raw_sample[5];")
            sketch.append(f"    }}")
            sketch.append(f"    #endif")
            sketch.append(f"")

        # On-device Kalman filter (matches base generator behavior)
        if self.enable_kalman_filter:
            parity_note = 'exact parity with training' if self._has_kalman_parity(
            ) else 'device-only — model trained without Kalman'
            sketch.append(
                f"    // Apply on-device Kalman filter ({parity_note})")
            sketch.append(f"    #ifdef KALMAN_FILTER_ENABLED")
            sketch.append(f"    {{")
            sketch.append(
                f"        float raw_sample[N_CHANNELS] = {{aX, aY, aZ, gX, gY, gZ}};")
            sketch.append(f"        kalman_filter_sample(raw_sample);")
            sketch.append(
                f"        aX = raw_sample[0]; aY = raw_sample[1]; aZ = raw_sample[2];")
            sketch.append(
                f"        gX = raw_sample[3]; gY = raw_sample[4]; gZ = raw_sample[5];")
            sketch.append(f"    }}")
            sketch.append(f"    #endif")
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

        # Persistent prediction (survives across loop iterations)
        sketch.append(
            f"    // Persistent last prediction — survives across loop iterations")
        sketch.append(f"    static const char* last_activity = NULL;")
        sketch.append(f"")
        sketch.append(f"    // Check if window is full")
        sketch.append(f"    if (sample_count >= WINDOW_SIZE) {{")

        if self.model_type == 'pytorch_cnn':
            sketch.append(
                f"        // CNN TFLite: pass raw window directly, no feature extraction needed")
            sketch.append(f"        float confidence = 0.0f;")
            sketch.append(
                f"        int predicted = har_predict_from_window(sensor_buffer, &confidence);")
        else:
            sketch.append(
                f"        // Extract features from raw sensor window")
            sketch.append(f"        float features[NUM_FEATURES];")
            sketch.append(
                f"        extract_features(sensor_buffer, WINDOW_SIZE, features);")
            sketch.append(f"")
            sketch.append(
                f"        // Run inference — har_predict() applies StandardScaler internally")
            sketch.append(f"        float confidence = 0.0f;")
            sketch.append(
                f"        int predicted = har_predict(features, &confidence);")

        sketch.append(f"")
        sketch.append(
            f"        // Debug: log prediction details (lines starting with # are ignored by parser)")
        sketch.append(
            f"        Serial.print(\"# PRED: class=\"); Serial.print(predicted);")
        sketch.append(
            f"        Serial.print(\" conf=\"); Serial.print(confidence, 4);")
        sketch.append(f"        Serial.print(\" name=\");")
        sketch.append(
            f"        Serial.println(predicted >= 0 ? get_activity_name(predicted) : \"none\");")
        sketch.append(f"")
        sketch.append(f"        // Update persistent prediction")
        sketch.append(
            f"        if (predicted >= 0 && confidence >= CONFIDENCE_THRESHOLD) {{")
        sketch.append(
            f"            last_activity = get_activity_name(predicted);")
        sketch.append(f"        }} else {{")
        sketch.append(f"            last_activity = \"unknown\";")
        sketch.append(f"        }}")
        sketch.append(f"")

        # Sliding window shift
        sketch.append(f"        // Slide window")
        sketch.append(
            f"        int keep = WINDOW_SIZE - (int)buffer_index_shift;")
        sketch.append(f"        for (int i = 0; i < keep; i++)")
        sketch.append(f"            for (int j = 0; j < N_CHANNELS; j++)")
        sketch.append(
            f"                sensor_buffer[i][j] = sensor_buffer[i + (int)buffer_index_shift][j];")
        sketch.append(f"        sample_count = keep;")
        sketch.append(f"    }}")
        sketch.append(f"")

        # Always output sensor CSV (for Device Test tab compatibility)
        sketch.append(
            f"    // Sensor CSV output (for Device Test tab live plotting)")
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

    def _generate_placeholder_model(self, error_msg: str) -> str:
        """Generate placeholder when TF is not installed."""
        lines = []
        lines.append(f"// ========================================")
        lines.append(f"// TFLite Model Placeholder")
        lines.append(f"// ========================================")
        lines.append(
            f"// TensorFlow is required to convert the model to TFLite format.")
        lines.append(f"// Install with: pip install tensorflow")
        lines.append(f"// Error: {error_msg}")
        lines.append(f"//")
        lines.append(
            f"// After installing TensorFlow, regenerate the code to get")
        lines.append(f"// the actual model byte array embedded here.")
        lines.append(f"// ========================================")
        lines.append(f"")
        lines.append(
            f"// Placeholder model array — replace with actual conversion output")
        lines.append(
            f"alignas(16) const unsigned char g_har_model[] = {{0x00}};")
        lines.append(f"const unsigned int g_har_model_len = 0;")
        lines.append(f"")
        lines.append(
            f"#warning \"TFLite model not converted — install tensorflow and regenerate\"")
        return '\n'.join(lines)
