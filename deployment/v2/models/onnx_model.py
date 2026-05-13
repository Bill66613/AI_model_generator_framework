"""ONNX Runtime model block — generates har_model.h/.cpp for ONNX Runtime inference.

ONNX Runtime targets Linux/Windows edge devices (Raspberry Pi, Jetson Nano, PC)
that have more resources than bare microcontrollers.

The v2 pattern is identical: har_features.h/.cpp and har_classifier.h/.cpp stay the
same; only har_model.h/.cpp changes to call the ONNX Runtime C++ API.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional, Tuple

from .base import ModelBlock

logger = logging.getLogger(__name__)


class ONNXModelBlock(ModelBlock):
    """
    Generates har_model.h + har_model.cpp for ONNX Runtime inference.

    Supported platforms:
      - Linux/Windows edge (Raspberry Pi, Jetson, PC) — full ONNX Runtime C++ API
      - Microcontroller with ONNX Runtime Micro (experimental)

    If model_object is absent, generates a stub with an error comment.

    Extra output: self.onnx_bytes is populated after generate(); the generator
    adds it as 'har_model.onnx' in the files dict.
    """

    def __init__(
        self,
        model_data: Dict[str, Any],
        feature_names: List[str],
        classes: List[str],
        precision: int,
        quantization: str = "none",
        platform: str = "generic_cpp",
    ):
        super().__init__(model_data, feature_names, classes, precision)
        self.quantization = quantization
        self.platform = platform

        self.onnx_bytes: Optional[bytes] = None
        self._convert_error: Optional[str] = None

        try:
            self._do_convert()
        except Exception as exc:
            self._convert_error = str(exc)
            logger.warning("ONNX conversion deferred: %s", exc)

    # ------------------------------------------------------------------

    def generate(self) -> Tuple[str, str]:
        if self._convert_error and not self.onnx_bytes:
            # Surface the error so the UI shows a failure, not silent broken stubs.
            raise RuntimeError(
                f"ONNX conversion failed: {self._convert_error}\n"
                "Make sure skl2onnx and onnxruntime are installed: "
                "uv sync --extra onnx"
            )
        return self._header(), self._impl()

    @property
    def model_type_id(self) -> str:
        return "onnx"

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _do_convert(self):
        # Accept pre-converted bytes stored in model_data
        if "onnx_bytes" in self.model_data:
            self.onnx_bytes = self.model_data["onnx_bytes"]
            return

        model_object = self.model_data.get("model_object")
        if model_object is None:
            raise ValueError(
                "ONNX conversion requires 'model_object' in model_data. "
                "Re-train or re-export the model to enable ONNX deployment."
            )

        from deployment.converters.onnx_converter import ONNXConverter

        converter = ONNXConverter(
            model_object=model_object,
            model_type=self.model_data.get("model_type", "unknown"),
            feature_names=self.feature_names,
            classes=self.classes,
            model_params=self.model_data.get("model_params", {}),
        )
        self.onnx_bytes = converter.convert()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _header(self) -> str:
        model_len = len(self.onnx_bytes) if self.onnx_bytes else 0
        return f"""\
#pragma once
#include "har_config.h"

/*
 * har_model.h — ONNX Runtime model
 * Model size: {model_len} bytes
 *
 * Input:  features[HAR_NUM_FEATURES]  — ALREADY StandardScaler-normalized
 * Output: probabilities[HAR_NUM_CLASSES] — class probabilities (sum to 1)
 *
 * Requires: ONNX Runtime C++ library linked at compile time.
 * Include path: -I<onnxruntime>/include
 * Library:       -lonnxruntime
 */

/* Optional: call before first prediction to warm up the session */
bool onnx_init(const char* model_path);

void har_model_predict(
    const float features[HAR_NUM_FEATURES],
    float probabilities[HAR_NUM_CLASSES]
);
"""

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _probe_onnx_output_name(self) -> str:
        """Return the name of the probability output tensor by inspecting the ONNX graph.

        skl2onnx (zipmap=False) names it 'probabilities'.
        PyTorch torch.onnx.export names it 'output'.
        Probing the actual graph is robust against version differences.
        """
        if not self.onnx_bytes:
            return "probabilities"
        try:
            import onnx
            model = onnx.load_from_string(self.onnx_bytes)
            # Prefer a 2-D float32 output — that's the probability tensor.
            # elem_type 1 = FLOAT in ONNX TensorProto.DataType.
            for out in model.graph.output:
                t = out.type.tensor_type
                if t.elem_type == 1 and len(t.shape.dim) == 2:
                    return out.name
            # Fallback: last output node
            if model.graph.output:
                return model.graph.output[-1].name
        except Exception:
            pass
        return "probabilities"

    def _impl(self) -> str:
        model_len = len(self.onnx_bytes) if self.onnx_bytes else 0
        prob_output = self._probe_onnx_output_name()
        return f"""\
#include "har_model.h"
#include <onnxruntime_cxx_api.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <array>

/* ---- ONNX Runtime session (lazy-initialized) ---- */
static Ort::Env*            _env         = nullptr;
static Ort::Session*        _session     = nullptr;
static Ort::AllocatorWithDefaultOptions _allocator;

bool onnx_init(const char* model_path) {{
    if (_session != nullptr) return true;
    try {{
        _env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "HAR");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        _session = new Ort::Session(*_env, model_path, opts);
        return true;
    }} catch (...) {{
        return false;
    }}
}}

static void _softmax(float* x, int n) {{
    float m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) {{ x[i] = expf(x[i] - m); s += x[i]; }}
    if (s > 0.0f) for (int i = 0; i < n; i++) x[i] /= s;
}}

void har_model_predict(
    const float features[HAR_NUM_FEATURES],
    float probabilities[HAR_NUM_CLASSES]
) {{
    if (_session == nullptr) {{
        HAR_LOG("ONNX session not initialized. Call onnx_init() first.");
        for (int i = 0; i < HAR_NUM_CLASSES; i++)
            probabilities[i] = 1.0f / (float)HAR_NUM_CLASSES;
        return;
    }}

    /* Build input tensor */
    std::array<int64_t, 2> input_shape{{{{1, HAR_NUM_FEATURES}}}};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        const_cast<float*>(features), HAR_NUM_FEATURES,
        input_shape.data(), input_shape.size());

    /* Run inference */
    const char* input_names[]  = {{"float_input"}};
    const char* output_names[] = {{"{prob_output}"}};
    try {{
        auto output_tensors = _session->Run(
            Ort::RunOptions{{nullptr}},
            input_names, &input_tensor, 1,
            output_names, 1);

        float* raw = output_tensors[0].GetTensorMutableData<float>();
        for (int i = 0; i < HAR_NUM_CLASSES; i++)
            probabilities[i] = raw[i];
        /* Ensure probabilities sum to 1 (in case model outputs raw logits) */
        float sum = 0.0f;
        for (int i = 0; i < HAR_NUM_CLASSES; i++) sum += probabilities[i];
        if (sum < 0.99f || sum > 1.01f) _softmax(probabilities, HAR_NUM_CLASSES);
    }} catch (...) {{
        HAR_LOG("ONNX inference error");
        for (int i = 0; i < HAR_NUM_CLASSES; i++)
            probabilities[i] = 1.0f / (float)HAR_NUM_CLASSES;
    }}
}}
"""

    # ------------------------------------------------------------------
    # Stubs
    # ------------------------------------------------------------------

    def _stub_header(self) -> str:
        return (
            "#pragma once\n"
            '#include "har_config.h"\n'
            "/* ERROR: ONNX model conversion failed — see har_model.cpp */\n"
            "bool onnx_init(const char* model_path);\n"
            "void har_model_predict(\n"
            "    const float features[HAR_NUM_FEATURES],\n"
            "    float probabilities[HAR_NUM_CLASSES]\n"
            ");\n"
        )

    def _stub_impl(self, error: str) -> str:
        safe_err = error.replace('"', "'").replace("\n", " ")
        return (
            '#include "har_model.h"\n'
            f'/* ONNX conversion error: {safe_err} */\n'
            "bool onnx_init(const char* model_path) { (void)model_path; return false; }\n"
            "void har_model_predict(\n"
            "    const float features[HAR_NUM_FEATURES],\n"
            "    float probabilities[HAR_NUM_CLASSES]\n"
            ") {\n"
            "    (void)features;\n"
            "    for (int i = 0; i < HAR_NUM_CLASSES; i++)\n"
            "        probabilities[i] = 1.0f / (float)HAR_NUM_CLASSES;\n"
            "}\n"
        )
