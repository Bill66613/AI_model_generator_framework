"""
ClassifierBlock — generates har_classifier.h and har_classifier.cpp.

This is the COMPLETE inference chain:
  1. har_extract_features(window, features)
  2. apply_scaler(features)       — StandardScaler: (x - mean) / std
  3. har_model_predict(features, probabilities)
  4. find best class + confidence threshold
  5. (optional) temporal smoothing

This file NEVER changes between model types.
Only the scaler arrays (means, stds) are model-specific numerical values.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple


class ClassifierBlock:
    """Generates har_classifier.h and har_classifier.cpp."""

    def __init__(
        self,
        feature_names: List[str],
        feature_means: List[float],
        feature_stds: List[float],
        classes: List[str],
        confidence_threshold: float,
        smoothing_window: int,
        precision: int,
        platform: str,
        skip_scaler: bool = False,
    ):
        self.feature_names = feature_names
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        self.classes = classes
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.precision = precision
        self.platform = platform
        self.skip_scaler = skip_scaler

    def generate(self) -> Tuple[str, str]:
        return self._header(), self._impl()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _header(self) -> str:
        return """\
#pragma once
#include "har_config.h"

/*
 * har_classifier.h — HAR inference chain entry point
 *
 * Call har_classify() once per window to get a prediction.
 * This function runs the full pipeline:
 *   feature extraction → scaler normalization → model predict
 *   → confidence threshold → temporal smoothing
 */

typedef struct {
    int   predicted_class;               /* HAR_CLASS_UNKNOWN (-1) if below threshold */
    float confidence;                    /* probability of predicted class, 0..1 */
    float probabilities[HAR_NUM_CLASSES]; /* all class probabilities */
} har_result_t;

/**
 * Run the full inference pipeline on one sensor window.
 *
 * @param window  Raw sensor data [HAR_WINDOW_SIZE][HAR_N_CHANNELS]
 *                channels: [aX, aY, aZ, gX, gY, gZ]
 *                units: m/s² (accel), deg/s (gyro)
 * @param result  Output struct written with prediction results.
 */
void har_classify(
    const float window[HAR_WINDOW_SIZE][HAR_N_CHANNELS],
    har_result_t *result
);

/** Convenience: return human-readable class name string. */
const char *har_get_class_name(int class_id);

/** Optional: reset temporal smoothing history. */
void har_reset_smoothing(void);
"""

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _impl(self) -> str:
        n_feat = len(self.feature_names)
        n_cls = len(self.classes)
        prec = self.precision

        # Scaler arrays (skip for CNN — saves flash)
        if self.skip_scaler:
            scaler_arrays = "/* No scaler arrays needed for CNN (raw sensor input) */"
        else:
            means_vals = ", ".join(f"{v:.{prec}f}f" for v in self.feature_means)
            stds_vals = ", ".join(f"{v:.{prec}f}f" for v in self.feature_stds)
            scaler_arrays = f"""\
/* ---- StandardScaler parameters (extracted from trained model) ---- */
static const float SCALER_MEANS[HAR_NUM_FEATURES] = {{
    {means_vals}
}};

static const float SCALER_STDS[HAR_NUM_FEATURES] = {{
    {stds_vals}
}};"""

        # Class name table
        class_names = ", ".join(f'"{c}"' for c in self.classes)

        # Temporal smoothing
        smoothing_code = self._smoothing_code()

        # Platform-specific includes
        includes = self._platform_includes()

        return f"""\
#include "har_classifier.h"
#include "har_features.h"
#include "har_model.h"
{includes}

{scaler_arrays}

/* ---- Class name table ---- */
const char *HAR_CLASS_NAMES[HAR_NUM_CLASSES] = {{{class_names}}};

const char *har_get_class_name(int class_id) {{
    if (class_id >= 0 && class_id < HAR_NUM_CLASSES)
        return HAR_CLASS_NAMES[class_id];
    return "unknown";
}}

/* ---- Scaler: (x - mean) / std  applied in-place ---- */
static void _apply_scaler(float *features) {{
    for (int i = 0; i < HAR_NUM_FEATURES; i++) {{
        /* Replace NaN/Inf with 0 */
        if (features[i] != features[i] || features[i] > 1e30f || features[i] < -1e30f)
            features[i] = 0.0f;
{self._scaler_body()}
    }}
}}

{smoothing_code}

/* ====================================================================
 * har_classify — The single entry point for inference.
 *
 * Pipeline:
 *   1. Extract features from raw window
 *   2. Apply StandardScaler normalization
 *   3. Call model-specific predict (writes class probabilities)
 *   4. Find best class, apply confidence threshold
 *   5. (Optional) temporal smoothing
 * ==================================================================== */
void har_classify(
    const float window[HAR_WINDOW_SIZE][HAR_N_CHANNELS],
    har_result_t *result
) {{
    /* --- Step 1: Feature extraction --- */
    float features[HAR_NUM_FEATURES];
    har_extract_features(window, features);

    /* --- Step 2: StandardScaler normalization --- */
    _apply_scaler(features);

    /* --- Step 3: Model inference --- */
    har_model_predict(features, result->probabilities);

    /* --- Step 4: Find best class --- */
    int best = 0;
    for (int c = 1; c < HAR_NUM_CLASSES; c++)
        if (result->probabilities[c] > result->probabilities[best]) best = c;

    result->confidence = result->probabilities[best];
    result->predicted_class = (result->confidence >= HAR_CONFIDENCE_THRESHOLD)
                               ? best : HAR_CLASS_UNKNOWN;

    /* --- Step 5: Temporal smoothing --- */
#if HAR_SMOOTHING_WINDOW > 1
    result->predicted_class = _smooth(result->predicted_class);
#endif
}}
"""

    def _scaler_body(self) -> str:
        """Return the inner scaler loop body.

        For CNN models (skip_scaler=True), returns nothing extra — NaN guard
        already applied, raw values pass through unchanged.
        For standard models, applies StandardScaler + [-10,10] clamp.
        """
        if self.skip_scaler:
            return "        /* CNN mode: raw values pass through (no scaling/clamp) */"
        return """\
        float std = SCALER_STDS[i];
        if (std < 1e-7f) std = 1.0f;  /* guard against zero std */
        features[i] = (features[i] - SCALER_MEANS[i]) / std;
        /* Clamp to [-10, 10] to prevent extreme scaled values */
        if (features[i] >  10.0f) features[i] =  10.0f;
        if (features[i] < -10.0f) features[i] = -10.0f;"""

    def _smoothing_code(self) -> str:
        if self.smoothing_window <= 1:
            return "/* Temporal smoothing disabled (HAR_SMOOTHING_WINDOW == 1) */"

        n = self.smoothing_window
        return f"""\
/* ---- Temporal majority-vote smoothing ---- */
static int _smooth_history[HAR_SMOOTHING_WINDOW];
static int _smooth_head = 0;
static int _smooth_count = 0;

void har_reset_smoothing(void) {{
    _smooth_head = 0;
    _smooth_count = 0;
    for (int i = 0; i < HAR_SMOOTHING_WINDOW; i++) _smooth_history[i] = HAR_CLASS_UNKNOWN;
}}

static int _smooth(int predicted) {{
    _smooth_history[_smooth_head] = predicted;
    _smooth_head = (_smooth_head + 1) % HAR_SMOOTHING_WINDOW;
    if (_smooth_count < HAR_SMOOTHING_WINDOW) _smooth_count++;

    int votes[HAR_NUM_CLASSES] = {{0}};
    int unknown_votes = 0;
    for (int i = 0; i < _smooth_count; i++) {{
        int pc = _smooth_history[i];
        if (pc >= 0 && pc < HAR_NUM_CLASSES) votes[pc]++;
        else unknown_votes++;
    }}
    int best = HAR_CLASS_UNKNOWN, best_v = unknown_votes;
    for (int c = 0; c < HAR_NUM_CLASSES; c++)
        if (votes[c] > best_v) {{ best_v = votes[c]; best = c; }}
    return best;
}}
"""

    def _platform_includes(self) -> str:
        if self.platform in ("generic_c", "generic_cpp"):
            return "#include <math.h>\n#include <string.h>"
        elif self.platform in ("esp_idf",):
            return "#include <math.h>"
        else:
            return "#include <math.h>"
