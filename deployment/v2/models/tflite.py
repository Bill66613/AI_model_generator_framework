"""TFLite Micro model block — generates har_model.h/.cpp for TFLite Micro deployment.

The v2 TFLite approach keeps the SAME har_classifier.h/.cpp and har_features.h/.cpp
as direct deployment.  Only har_model.h/.cpp changes: instead of hand-coded weights
it embeds the .tflite model as a C byte array and wraps the TFLite Micro interpreter
behind the standard  har_model_predict(features, probabilities)  interface.

This means the user sketch and the inference chain are completely unaware of whether
the underlying model is RF/NN weights or a TFLite flatbuffer.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional, Tuple

from .base import ModelBlock

logger = logging.getLogger(__name__)


class TFLiteModelBlock(ModelBlock):
    """
    Generates har_model.h + har_model.cpp for TFLite Micro inference.

    The block:
      1. Converts the model to .tflite via TFLiteConverter (if model_object present)
      2. Embeds the model bytes as  const unsigned char g_har_model[]
      3. Generates tflite_init() and har_model_predict() using TFLite Micro API

    If model_object is absent (loaded from .joblib without the live sklearn object)
    the block generates a stub that prints an error — the user must re-export the model.

    Extra output: self.tflite_bytes is populated after generate(); the generator
    adds it as 'har_model.tflite' in the files dict.
    """

    def __init__(
        self,
        model_data: Dict[str, Any],
        feature_names: List[str],
        classes: List[str],
        precision: int,
        quantization: str = "none",
        platform: str = "arduino",
        window_size: int = 150,
        sampling_rate: int = 100,
    ):
        super().__init__(model_data, feature_names, classes, precision)
        self.quantization = quantization
        self.platform = platform
        self.window_size = window_size
        self.sampling_rate = sampling_rate

        self.tflite_bytes: Optional[bytes] = None
        self._c_array: str = ""
        self._ops: Optional[list] = None
        self._use_all_ops: bool = False
        self._arena_size: int = 0

        # Try conversion immediately (best-effort; errors surface in generate())
        self._convert_error: Optional[str] = None
        try:
            self._do_convert()
        except Exception as exc:
            self._convert_error = str(exc)
            logger.warning("TFLite conversion deferred: %s", exc)

    # ------------------------------------------------------------------

    def generate(self) -> Tuple[str, str]:
        if self._convert_error and not self.tflite_bytes:
            return self._stub_header(), self._stub_impl(self._convert_error)
        return self._header(), self._impl()

    @property
    def model_type_id(self) -> str:
        return "tflite"

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _do_convert(self):
        from deployment.converters.tflite_converter import TFLiteConverter

        model_object = self.model_data.get("model_object")
        if model_object is None:
            # Pre-extracted bytes stored directly (e.g. re-loaded from disk)
            if "tflite_bytes" in self.model_data:
                self.tflite_bytes = self.model_data["tflite_bytes"]
                self._c_array = self._bytes_to_c_array(self.tflite_bytes)
                self._arena_size = self._estimate_arena()
                return
            raise ValueError(
                "TFLite conversion requires 'model_object' in model_data. "
                "Re-train or re-export the model to enable TFLite deployment."
            )

        # Enrich converter params
        converter_params = dict(self.model_data.get("model_params", {}))
        converter_params["window_size_samples"] = self.window_size
        converter_params["n_channels"] = 6

        model_info = self.model_data.get("model_info", {})
        fe_cfg = model_info.get("fe_config", {})
        n_ch = (
            fe_cfg.get("num_channels")
            or len(fe_cfg.get("sensor_columns", []))
            or 6
        )
        converter_params["n_channels"] = n_ch

        converter = TFLiteConverter(
            model_object=model_object,
            model_type=self.model_data.get("model_type", "unknown"),
            feature_names=self.feature_names,
            classes=self.classes,
            model_params=converter_params,
        )

        rep_data = self.model_data.get("representative_data")
        self.tflite_bytes = converter.convert(
            quantization=self.quantization,
            representative_data=rep_data,
        )
        self._c_array = converter.to_c_array("g_har_model", platform=self.platform)
        raw_ops = converter.enumerate_ops()
        if raw_ops is None:
            self._use_all_ops = True
        else:
            self._ops = raw_ops
        self._arena_size = self._estimate_arena()

    def _bytes_to_c_array(self, data: bytes) -> str:
        """Minimal fallback: embed raw bytes as a C array."""
        hex_vals = ", ".join(f"0x{b:02x}" for b in data)
        return (
            f"const unsigned char g_har_model[] = {{{hex_vals}}};\n"
            f"const unsigned int g_har_model_len = {len(data)};\n"
        )

    def _estimate_arena(self) -> int:
        bpv = {"int8": 1, "int16": 2}.get(self.quantization, 4)
        n = self.n_features
        nc = self.n_classes
        model_type = self.model_data.get("model_type", "")
        if model_type in ("neural_network", "pytorch_mlp"):
            weights = self.model_data.get("weights", {})
            h = weights.get("hidden_size", 64) if weights else 64
            arena = (n * 4 + h * 2 * bpv + nc * 4) * 2
        else:
            arena = (n + nc) * 4 * 4
        arena = max(arena, 8 * 1024)
        arena = min(arena, 200 * 1024)
        return ((arena + 1023) // 1024) * 1024

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _header(self) -> str:
        arena = self._arena_size
        model_len = len(self.tflite_bytes) if self.tflite_bytes else 0
        return f"""\
#pragma once
#include "har_config.h"
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
{"#include \"tensorflow/lite/micro/all_ops_resolver.h\"" if self._use_all_ops else "#include \"tensorflow/lite/micro/micro_mutable_op_resolver.h\""}

/*
 * har_model.h — TFLite Micro model
 * Quantization: {self.quantization}
 * Arena size:   {arena} bytes
 * Model size:   {model_len} bytes
 *
 * Input:  features[HAR_NUM_FEATURES]  — ALREADY StandardScaler-normalized
 * Output: probabilities[HAR_NUM_CLASSES] — class probabilities (sum to 1)
 */

#define TENSOR_ARENA_SIZE {arena}
#define TFLITE_MODEL_SIZE {model_len}

extern const unsigned char g_har_model[];
extern const unsigned int  g_har_model_len;

/* Internal TFLite init — called lazily by har_model_predict() */
bool tflite_init(void);

void har_model_predict(
    const float features[HAR_NUM_FEATURES],
    float probabilities[HAR_NUM_CLASSES]
);
"""

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _impl(self) -> str:
        arena_decl = self._arena_declaration()
        resolver_code = self._resolver_code()

        return f"""\
#include "har_model.h"
#include <math.h>

/* ---- TFLite model (embedded as C byte array) ---- */
{self._c_array}

/* ---- TFLite Micro globals ---- */
static const tflite::Model* _model = nullptr;
static tflite::MicroInterpreter* _interpreter = nullptr;
static TfLiteTensor* _input  = nullptr;
static TfLiteTensor* _output = nullptr;
{arena_decl}

bool tflite_init(void) {{
{self._arena_init_check()}
    _model = tflite::GetModel(g_har_model);
    if (_model->version() != TFLITE_SCHEMA_VERSION) {{
        HAR_LOG("TFLite schema version mismatch!");
        return false;
    }}
{resolver_code}
    static tflite::MicroInterpreter static_interp(
        _model, resolver, tensor_arena, TENSOR_ARENA_SIZE, nullptr);
    _interpreter = &static_interp;
    if (_interpreter->AllocateTensors() != kTfLiteOk) {{
        HAR_LOG("AllocateTensors() failed");
        return false;
    }}
    _input  = _interpreter->input(0);
    _output = _interpreter->output(0);
    HAR_LOG("TFLite init OK. Arena used: %d / %d", _interpreter->arena_used_bytes(), TENSOR_ARENA_SIZE);
    return true;
}}

void har_model_predict(
    const float features[HAR_NUM_FEATURES],
    float probabilities[HAR_NUM_CLASSES]
) {{
    if (_interpreter == nullptr) {{
        if (!tflite_init()) {{
            /* init failed — return uniform probabilities */
            for (int i = 0; i < HAR_NUM_CLASSES; i++)
                probabilities[i] = 1.0f / (float)HAR_NUM_CLASSES;
            return;
        }}
    }}
    /* Copy scaled features into TFLite input tensor */
    for (int i = 0; i < HAR_NUM_FEATURES; i++)
        _input->data.f[i] = features[i];

    if (_interpreter->Invoke() != kTfLiteOk) {{
        HAR_LOG("TFLite Invoke() failed");
        for (int i = 0; i < HAR_NUM_CLASSES; i++)
            probabilities[i] = 1.0f / (float)HAR_NUM_CLASSES;
        return;
    }}
    for (int i = 0; i < HAR_NUM_CLASSES; i++)
        probabilities[i] = _output->data.f[i];
}}
"""

    def _arena_declaration(self) -> str:
        is_esp32 = self.platform in ("esp32", "m5stack", "m5stickc", "m5stick")
        if is_esp32:
            return (
                "static uint8_t* tensor_arena = nullptr;\n\n"
                "static bool _alloc_arena(void) {\n"
                "    if (tensor_arena) return true;\n"
                "#if defined(BOARD_HAS_PSRAM) || defined(ESP_PSRAM_FOUND)\n"
                "    tensor_arena = (uint8_t*)ps_malloc(TENSOR_ARENA_SIZE);\n"
                "    if (tensor_arena) return true;\n"
                "#endif\n"
                "    tensor_arena = (uint8_t*)malloc(TENSOR_ARENA_SIZE);\n"
                "    return tensor_arena != nullptr;\n"
                "}"
            )
        return "static uint8_t tensor_arena[TENSOR_ARENA_SIZE];"

    def _arena_init_check(self) -> str:
        is_esp32 = self.platform in ("esp32", "m5stack", "m5stickc", "m5stick")
        if is_esp32:
            return (
                "    if (!_alloc_arena()) {\n"
                '        HAR_LOG("Arena allocation failed");\n'
                "        return false;\n"
                "    }"
            )
        return ""

    def _resolver_code(self) -> str:
        if self._use_all_ops:
            return (
                "    /* AllOpsResolver: model contains unrecognized ops */\n"
                "    static tflite::AllOpsResolver resolver;"
            )
        ops = self._ops or ["Dequantize", "FullyConnected", "Quantize", "Softmax"]
        ops_sorted = sorted(ops)
        lines = [
            f"    static tflite::MicroMutableOpResolver<{len(ops_sorted)}> resolver;"
        ]
        for op in ops_sorted:
            lines.append(f"    resolver.Add{op}();")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stubs for conversion failure
    # ------------------------------------------------------------------

    def _stub_header(self) -> str:
        return (
            "#pragma once\n"
            '#include "har_config.h"\n'
            "/* ERROR: TFLite model conversion failed — see har_model.cpp */\n"
            "void har_model_predict(\n"
            "    const float features[HAR_NUM_FEATURES],\n"
            "    float probabilities[HAR_NUM_CLASSES]\n"
            ");\n"
        )

    def _stub_impl(self, error: str) -> str:
        safe_err = error.replace('"', "'").replace("\n", " ")
        return (
            '#include "har_model.h"\n'
            f'/* TFLite conversion error: {safe_err} */\n'
            f'/* Re-train the model and regenerate to fix this. */\n'
            "void har_model_predict(\n"
            "    const float features[HAR_NUM_FEATURES],\n"
            "    float probabilities[HAR_NUM_CLASSES]\n"
            ") {\n"
            "    (void)features;\n"
            "    for (int i = 0; i < HAR_NUM_CLASSES; i++)\n"
            "        probabilities[i] = 1.0f / (float)HAR_NUM_CLASSES;\n"
            "}\n"
        )
