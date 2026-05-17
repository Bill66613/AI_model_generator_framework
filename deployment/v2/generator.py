"""
HARCodeGenerator — main coordinator for v2 clean architecture.

Usage:
    gen = HARCodeGenerator(model_data, platform='esp32', ...)
    files = gen.generate_files()   # returns Dict[filename -> content]
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple

from .feature_block import FeatureBlock
from .classifier_block import ClassifierBlock
from .models import create_model_block
from .platforms import create_platform_sketch


class HARCodeGenerator:
    """
    Clean code generator that produces a well-structured multi-file deployment.

    The generated files have strict separation of concerns:
      har_config.h        — constants only, no code
      har_features.h/.cpp — feature extraction, no knowledge of model type
      har_model.h/.cpp    — model weights + predict(), no knowledge of platform
      har_classifier.h/.cpp — inference chain wrapper, stable across all deployments
      {sketch}.ino        — platform setup + sensor loop, calls har_classify()
    """

    def __init__(
        self,
        model_data: Dict[str, Any],
        platform: str = "arduino",
        optimization: str = "balanced",
        overlap: float = 0.5,
        confidence_threshold: float = 0.6,
        smoothing_window: int = 1,
        quantization: str = "none",
        deployment_approach: str = "direct",
    ):
        self.model_data = model_data
        self.platform = platform
        self.optimization = optimization
        self.overlap = max(0.0, min(0.99, overlap))
        self.confidence_threshold = max(
            0.0, min(1.0, float(confidence_threshold)))
        self.smoothing_window = max(1, min(9, int(smoothing_window)))
        self.quantization = quantization
        self.deployment_approach = deployment_approach

        # --- Core metadata ---
        self.model_type = model_data.get("model_type", "unknown")
        self.feature_names: List[str] = [
            str(f) for f in model_data.get("feature_names", [])]
        self.classes: List[str] = [str(c)
                                   for c in model_data.get("classes", [])]
        self.feature_means: List[float] = model_data.get(
            "feature_means", [0.0] * len(self.feature_names))
        self.feature_stds: List[float] = model_data.get(
            "feature_stds", [1.0] * len(self.feature_names))

        # --- Window / sampling config ---
        model_info = model_data.get("model_info", {})
        fe_cfg = model_info.get("fe_config", {})
        # Also look in model_params for legacy paths
        model_params = model_data.get("model_params", {})
        sampling_rate = fe_cfg.get(
            "sampling_rate") or model_params.get("sampling_rate", 100)
        window_ms = fe_cfg.get("window_size_ms") or model_params.get(
            "window_size_ms", 1500)
        self.sampling_rate: int = int(sampling_rate)
        self.window_size: int = int(
            round(window_ms / 1000 * self.sampling_rate))
        self.preprocessing: Dict = fe_cfg.get("preprocessing") or {}

        # --- CNN override: use flat window as "feature" vector ---
        if self.model_type in ("pytorch_cnn", "pytorch_cnn2d"):
            n_ch = model_data.get("n_channels", 6)
            self.feature_names = [
                f"cnn_in_{t}_{c}"
                for t in range(self.window_size)
                for c in range(n_ch)
            ]
            # Identity scaler — CNN normalises internally (no StandardScaler)
            self.feature_means = [0.0] * len(self.feature_names)
            self.feature_stds  = [1.0] * len(self.feature_names)

        # --- Derived precision ---
        self.precision = {"accuracy": 6, "balanced": 4,
                          "speed": 3, "power": 3}.get(optimization, 4)

        # --- Build blocks ---
        self._feature_block = FeatureBlock(
            feature_names=self.feature_names,
            window_size=self.window_size,
            sampling_rate=self.sampling_rate,
            preprocessing=self.preprocessing,
            platform=platform,
        )
        self._model_block = create_model_block(
            model_data=model_data,
            feature_names=self.feature_names,
            classes=self.classes,
            precision=self.precision,
            deployment_approach=deployment_approach,
            quantization=quantization,
            platform=platform,
            window_size=self.window_size,
            sampling_rate=self.sampling_rate,
        )
        self._classifier_block = ClassifierBlock(
            feature_names=self.feature_names,
            feature_means=self.feature_means,
            feature_stds=self.feature_stds,
            classes=self.classes,
            confidence_threshold=self.confidence_threshold,
            smoothing_window=self.smoothing_window,
            precision=self.precision,
            platform=platform,
            skip_scaler=self.model_type in ("pytorch_cnn", "pytorch_cnn2d"),
        )
        # Derive sketch name: same as sketch folder (Arduino IDE requirement)
        self.sketch_name = _safe_c_ident(
            model_data.get('model_name', 'har_sketch')).lower()
        if not self.sketch_name or self.sketch_name == 'unknown':
            self.sketch_name = 'har_sketch'
        self._sketch = create_platform_sketch(
            platform=platform,
            sketch_name=self.sketch_name,
            classes=self.classes,
            window_size=self.window_size,
            sampling_rate=self.sampling_rate,
            overlap=self.overlap,
            smoothing_window=self.smoothing_window,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_files(self) -> Dict[str, str]:
        """
        Generate all deployment files.

        Returns dict mapping filename → content string (text files) or bytes (binary).
        The filenames are suitable for dropping into a single Arduino sketch folder.
        """
        # MicroPython is structurally different (all Python, no C++)
        if self.platform == "micropython":
            return self._generate_micropython_files()

        files: Dict[str, str] = {}

        files["har_config.h"] = self._generate_config_h()

        feat_h, feat_cpp = self._feature_block.generate()
        files["har_features.h"] = feat_h
        files["har_features.cpp"] = feat_cpp

        model_h, model_cpp = self._model_block.generate()
        files["har_model.h"] = model_h
        files["har_model.cpp"] = model_cpp

        clf_h, clf_cpp = self._classifier_block.generate()
        files["har_classifier.h"] = clf_h
        files["har_classifier.cpp"] = clf_cpp

        # Platform sketch — may return multiple files (e.g. Zephyr: main.c + CMakeLists.txt)
        sketch_files = self._sketch.generate_files()
        files.update(sketch_files)

        # Extra binary files (TFLite / ONNX)
        if hasattr(self._model_block, "tflite_bytes") and self._model_block.tflite_bytes:
            files["har_model.tflite"] = self._model_block.tflite_bytes
            # Arduino IDE build workaround for TFLite Micro / GCC stl_emulation.h conflict
            if self.platform in ("arduino", "seeed_xiao", "esp32", "m5stack",
                                 "m5stick", "m5stickc", "teensy"):
                # build_opt.h: one compiler flag per line, NO comments allowed
                files["build_opt.h"] = "-fpermissive\n"

        if hasattr(self._model_block, "onnx_bytes") and self._model_block.onnx_bytes:
            files["har_model.onnx"] = self._model_block.onnx_bytes

        return files

    # ------------------------------------------------------------------
    # MicroPython code generation (Python output, not C++)
    # ------------------------------------------------------------------

    def _generate_micropython_files(self) -> Dict[str, str]:
        """Generate a self-contained MicroPython module + example main.py."""
        from .platforms.micropython import MicroPythonCodeGen
        mp_gen = MicroPythonCodeGen(
            model_data=self.model_data,
            feature_names=self.feature_names,
            classes=self.classes,
            feature_means=self.feature_means,
            feature_stds=self.feature_stds,
            window_size=self.window_size,
            sampling_rate=self.sampling_rate,
            precision=self.precision,
            model_type=self.model_type,
            sketch_name=self.sketch_name,
        )
        return mp_gen.generate_files()

    # ------------------------------------------------------------------
    # Config header generation (simple constants, no logic)
    # ------------------------------------------------------------------

    def _generate_config_h(self) -> str:
        safe_classes = [_safe_c_ident(c).upper() for c in self.classes]
        enum_entries = "\n    ".join(
            f"HAR_CLASS_{name} = {i}," for i, name in enumerate(safe_classes)
        )
        feature_list = "\n".join(
            f" *   [{i:3d}] {name}" for i, name in enumerate(self.feature_names)
        )
        return f"""\
#pragma once
/* ============================================================
 * HAR Configuration
 * Generated by HAR Edge Framework v2
 * Model:       {self.model_type}
 * Platform:    {self.platform}
 * Optimization:{self.optimization}
 * ============================================================ */

/* ---- Data dimensions ---- */
#define HAR_NUM_FEATURES          {len(self.feature_names)}
#define HAR_NUM_CLASSES           {len(self.classes)}
#define HAR_N_CHANNELS            6        /* aX, aY, aZ, gX, gY, gZ */
#define HAR_WINDOW_SIZE           {self.window_size}

/* ---- Timing ---- */
#define HAR_SAMPLE_RATE           {self.sampling_rate}
#define HAR_SAMPLING_RATE         HAR_SAMPLE_RATE   /* alias */
#define HAR_STEP_SIZE             {max(1, int(self.window_size * (1.0 - self.overlap)))}

/* ---- Inference settings ---- */
#define HAR_CONFIDENCE_THRESHOLD  {self.confidence_threshold:.2f}f
#define HAR_SMOOTHING_WINDOW      {self.smoothing_window}
#define HAR_OVERLAP               {self.overlap:.2f}f

/* ---- Feature order (C++ canonical extraction order) ----
{feature_list}
 * ---- */

/* ---- Activity class enum ---- */
typedef enum {{
    HAR_CLASS_UNKNOWN = -1,
    {enum_entries}
}} har_class_t;

/* ---- Class name lookup (defined in har_classifier.cpp) ---- */
extern const char* HAR_CLASS_NAMES[HAR_NUM_CLASSES];

/* ---- Debug logging (define HAR_DEBUG before including to enable) ---- */
#ifndef HAR_DEBUG
#  define HAR_LOG(fmt, ...)        do {{}} while (0)
#  define HAR_LOG_FLOAT(lbl, val)  do {{}} while (0)
#else
#  include <stdio.h>
#  define HAR_LOG(fmt, ...)        printf("[HAR] " fmt "\\n", ##__VA_ARGS__)
#  define HAR_LOG_FLOAT(lbl, val)  printf("[HAR] %s: %f\\n", (lbl), (double)(val))
#endif
"""


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _safe_c_ident(name: str) -> str:
    """Convert a label string to a valid C identifier."""
    safe = name.upper().replace(" ", "_").replace("-", "_")
    safe = "".join(c for c in safe if c.isalnum() or c == "_")
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe or "UNKNOWN"
