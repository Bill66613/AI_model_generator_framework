"""SVM model block — generates har_model.h/.cpp for SVM (RBF kernel, OvO voting)."""

from __future__ import annotations
from typing import Dict, Any, List, Tuple

import numpy as np

from .base import ModelBlock


class SVMModelBlock(ModelBlock):
    """
    Generates C++ SVM (RBF kernel) inference code using One-vs-One voting.

    Uses sklearn's native OvO representation directly.  For each of the
    C(n_classes, 2) pairs, the binary decision function is computed and
    the winning class gets one vote.  The class with the most votes is
    the prediction.  Confidence = votes_won / (n_classes − 1).

    This is much more robust than OvR + softmax because each vote depends
    only on the *sign* of the decision function, not its magnitude.
    Even when RBF kernel values are small (due to minor C++/Python feature
    parity differences), the sign is typically preserved.

    RBF kernel: K(x, sv) = exp( -gamma * ||x - sv||² )
    """

    def __init__(self, model_data, feature_names, classes, precision,
                 quantization: str = "none"):
        super().__init__(model_data, feature_names, classes, precision)
        self._sv = np.array(model_data.get("support_vectors", []))
        # OvO data (preferred — native sklearn format)
        self._ovo_dual_coef = np.array(
            model_data.get("ovo_dual_coef", []))  # [n_classes-1, n_sv]
        self._ovo_intercept = np.array(
            model_data.get("ovo_intercept", []))   # [n_pairs]
        self._n_support = model_data.get("n_support", [])
        # Fallback OvR data (for backward compat)
        self._dual_coef = np.array(model_data.get("dual_coefficients", []))
        self._intercept = np.array(model_data.get("intercept", []))
        self._gamma = float(model_data.get("gamma", 0.1))
        self.quantization = quantization

    # ------------------------------------------------------------------

    def generate(self) -> Tuple[str, str]:
        n_sv = len(self._sv) if len(self._sv) > 0 else 0
        use_ovo = (len(self._ovo_dual_coef) > 0
                   and len(self._ovo_intercept) > 0
                   and len(self._n_support) > 0)
        if use_ovo:
            return self._header_ovo(n_sv), self._impl_ovo(n_sv)
        # Fallback to OvR for backward compat (unlikely)
        return self._header_ovr(n_sv), self._impl_ovr(n_sv)

    # ------------------------------------------------------------------
    # OvO voting (preferred — robust to feature parity imperfections)
    # ------------------------------------------------------------------

    def _header_ovo(self, n_sv: int) -> str:
        n_pairs = len(self.classes) * (len(self.classes) - 1) // 2
        quant_line = ""
        if self.quantization in ('int8', 'int16'):
            quant_line = f"\n * Quantization  : {self.quantization.upper()} (support vectors quantized)"
        return f"""\
#pragma once
#include "har_config.h"

/*
 * har_model.h — SVM (RBF kernel, One-vs-One voting)
 * Support vectors : {n_sv}
 * Pairs           : {n_pairs}
 * Gamma           : {self._gamma:.6g}{quant_line}
 *
 * Uses sklearn's native OvO pairwise voting.  For each pair (ci, cj)
 * the binary decision function is evaluated and the winning class gets
 * one vote.  Confidence = votes_won / (n_classes − 1).
 *
 * Input:  features[HAR_NUM_FEATURES]  — ALREADY StandardScaler-normalized
 * Output: probabilities[HAR_NUM_CLASSES] — vote fractions per class
 */
void har_model_predict(
    const float features[HAR_NUM_FEATURES],
    float probabilities[HAR_NUM_CLASSES]
);
"""

    def _impl_ovo(self, n_sv: int) -> str:
        if n_sv == 0:
            return self._empty_impl()
        if self.quantization in ('int8', 'int16'):
            return self._impl_ovo_quantized(n_sv)
        return self._impl_ovo_float(n_sv)

    def _impl_ovo_float(self, n_sv: int) -> str:
        n_classes = len(self.classes)
        n_pairs = n_classes * (n_classes - 1) // 2
        prec = self.precision

        # Support vectors (same as before)
        sv_rows = []
        for sv in self._sv:
            vals = ", ".join(f"{v:.{prec}f}f" for v in sv)
            sv_rows.append(f"    {{{vals}}}")
        sv_arr = ",\n".join(sv_rows)

        # OvO packed dual_coef: shape [n_classes-1, n_sv]
        dc_rows = []
        for row in self._ovo_dual_coef:
            vals = ", ".join(f"{v:.{prec}f}f" for v in row)
            dc_rows.append(f"    {{{vals}}}")
        dc_arr = ",\n".join(dc_rows)

        # OvO intercepts: one per pair
        ic_vals = ", ".join(
            f"{v:.{prec}f}f" for v in self._ovo_intercept)

        # Per-class SV counts
        ns_vals = ", ".join(str(n) for n in self._n_support)

        return f"""\
#include "har_model.h"
#include <math.h>

#define HAR_SVM_N_SV    {n_sv}
#define HAR_SVM_N_PAIRS {n_pairs}

static const float har_svm_sv[HAR_SVM_N_SV][HAR_NUM_FEATURES] = {{
{sv_arr}
}};

/* sklearn packed dual_coef_: shape [{n_classes - 1}][{n_sv}] */
static const float har_svm_dual_coef[{n_classes - 1}][HAR_SVM_N_SV] = {{
{dc_arr}
}};

/* Per-pair intercepts (n_pairs = {n_pairs}) */
static const float har_svm_ovo_intercept[HAR_SVM_N_PAIRS] = {{{ic_vals}}};

/* Number of SVs belonging to each class */
static const int har_svm_n_support[HAR_NUM_CLASSES] = {{{ns_vals}}};

static const float HAR_SVM_GAMMA = {self._gamma:.{prec}f}f;

void har_model_predict(
    const float features[HAR_NUM_FEATURES],
    float probabilities[HAR_NUM_CLASSES]
) {{
    /* ---- Step 1: pre-compute RBF kernel for all SVs ---- */
    float kernel[HAR_SVM_N_SV];
    for (int s = 0; s < HAR_SVM_N_SV; s++) {{
        float dist2 = 0.0f;
        for (int i = 0; i < HAR_NUM_FEATURES; i++) {{
            float d = features[i] - har_svm_sv[s][i];
            dist2 += d * d;
        }}
        kernel[s] = expf(-HAR_SVM_GAMMA * dist2);
    }}

    /* ---- Step 2: cumulative SV start indices per class ---- */
    int class_start[HAR_NUM_CLASSES + 1];
    class_start[0] = 0;
    for (int c = 0; c < HAR_NUM_CLASSES; c++)
        class_start[c + 1] = class_start[c] + har_svm_n_support[c];

    /* ---- Step 3: OvO pairwise voting ---- */
    int votes[HAR_NUM_CLASSES];
    for (int c = 0; c < HAR_NUM_CLASSES; c++) votes[c] = 0;

    int pair = 0;
    for (int ci = 0; ci < HAR_NUM_CLASSES; ci++) {{
        for (int cj = ci + 1; cj < HAR_NUM_CLASSES; cj++) {{
            float f = har_svm_ovo_intercept[pair];

            /* SVs of class ci: dual_coef row = cj - 1 */
            for (int s = class_start[ci]; s < class_start[ci + 1]; s++)
                f += har_svm_dual_coef[cj - 1][s] * kernel[s];

            /* SVs of class cj: dual_coef row = ci */
            for (int s = class_start[cj]; s < class_start[cj + 1]; s++)
                f += har_svm_dual_coef[ci][s] * kernel[s];

            if (f > 0.0f)
                votes[ci]++;
            else
                votes[cj]++;

            pair++;
        }}
    }}

    /* ---- Step 4: confidence = votes / (n_classes - 1) ---- */
    float denom = (float)(HAR_NUM_CLASSES - 1);
    for (int c = 0; c < HAR_NUM_CLASSES; c++)
        probabilities[c] = (float)votes[c] / denom;
}}
"""

    # ------------------------------------------------------------------
    # OvO quantized (int8/int16 support vectors)
    # ------------------------------------------------------------------

    def _impl_ovo_quantized(self, n_sv: int) -> str:
        from deployment.quantization import quantize_symmetric_int8, quantize_symmetric_int16

        n_classes = len(self.classes)
        n_pairs = n_classes * (n_classes - 1) // 2
        n_features = len(self.feature_names)
        prec = self.precision

        # Quantize support vectors (per-tensor symmetric)
        sv_np = np.array(self._sv, dtype=np.float32)
        quantize_fn = (quantize_symmetric_int8 if self.quantization == 'int8'
                       else quantize_symmetric_int16)
        qt_sv = quantize_fn(sv_np)
        c_type = 'int8_t' if self.quantization == 'int8' else 'int16_t'

        # Memory stats
        orig_bytes = sv_np.size * 4
        quant_bytes = sv_np.size * (1 if self.quantization == 'int8' else 2)

        # Flatten SV data row-major: sv[s * n_features + i]
        flat_sv = qt_sv.data.flatten()
        sv_vals = ", ".join(str(int(v)) for v in flat_sv)
        # Wrap at ~80 chars
        sv_lines = []
        vals_list = [str(int(v)) for v in flat_sv]
        line = "    "
        for i, v in enumerate(vals_list):
            if i > 0:
                line += ", "
            if len(line) + len(v) > 100:
                sv_lines.append(line)
                line = "    " + v
            else:
                line += v
        sv_lines.append(line)
        sv_arr = "\n".join(sv_lines)

        # Dual coef stays float (small array, quantization could flip decision signs)
        dc_rows = []
        for row in self._ovo_dual_coef:
            vals = ", ".join(f"{v:.{prec}f}f" for v in row)
            dc_rows.append(f"    {{{vals}}}")
        dc_arr = ",\n".join(dc_rows)

        ic_vals = ", ".join(
            f"{v:.{prec}f}f" for v in self._ovo_intercept)
        ns_vals = ", ".join(str(n) for n in self._n_support)

        return f"""\
#include "har_model.h"
#include <math.h>
#include <stdint.h>

#define HAR_SVM_N_SV    {n_sv}
#define HAR_SVM_N_PAIRS {n_pairs}

/*
 * Support vectors: {self.quantization.upper()} quantized
 * Original: {orig_bytes:,} bytes (float32) → Quantized: {quant_bytes:,} bytes
 * Compression: {orig_bytes / quant_bytes:.1f}x
 * Scale: {qt_sv.scale:.8f}
 */
static const {c_type} har_svm_sv_q[HAR_SVM_N_SV * HAR_NUM_FEATURES] = {{
{sv_arr}
}};

static const float HAR_SVM_SV_SCALE = {qt_sv.scale:.8f}f;

/* sklearn packed dual_coef_: shape [{n_classes - 1}][{n_sv}] (float — not quantized) */
static const float har_svm_dual_coef[{n_classes - 1}][HAR_SVM_N_SV] = {{
{dc_arr}
}};

/* Per-pair intercepts (n_pairs = {n_pairs}) */
static const float har_svm_ovo_intercept[HAR_SVM_N_PAIRS] = {{{ic_vals}}};

/* Number of SVs belonging to each class */
static const int har_svm_n_support[HAR_NUM_CLASSES] = {{{ns_vals}}};

static const float HAR_SVM_GAMMA = {self._gamma:.{prec}f}f;

void har_model_predict(
    const float features[HAR_NUM_FEATURES],
    float probabilities[HAR_NUM_CLASSES]
) {{
    /* ---- Step 1: pre-compute RBF kernel for all SVs ---- */
    /* Dequantize SVs on the fly: sv_float = sv_q * scale */
    float kernel[HAR_SVM_N_SV];
    for (int s = 0; s < HAR_SVM_N_SV; s++) {{
        float dist2 = 0.0f;
        const {c_type}* sv_row = &har_svm_sv_q[s * HAR_NUM_FEATURES];
        for (int i = 0; i < HAR_NUM_FEATURES; i++) {{
            float sv_val = (float)sv_row[i] * HAR_SVM_SV_SCALE;
            float d = features[i] - sv_val;
            dist2 += d * d;
        }}
        kernel[s] = expf(-HAR_SVM_GAMMA * dist2);
    }}

    /* ---- Step 2: cumulative SV start indices per class ---- */
    int class_start[HAR_NUM_CLASSES + 1];
    class_start[0] = 0;
    for (int c = 0; c < HAR_NUM_CLASSES; c++)
        class_start[c + 1] = class_start[c] + har_svm_n_support[c];

    /* ---- Step 3: OvO pairwise voting ---- */
    int votes[HAR_NUM_CLASSES];
    for (int c = 0; c < HAR_NUM_CLASSES; c++) votes[c] = 0;

    int pair = 0;
    for (int ci = 0; ci < HAR_NUM_CLASSES; ci++) {{
        for (int cj = ci + 1; cj < HAR_NUM_CLASSES; cj++) {{
            float f = har_svm_ovo_intercept[pair];

            /* SVs of class ci: dual_coef row = cj - 1 */
            for (int s = class_start[ci]; s < class_start[ci + 1]; s++)
                f += har_svm_dual_coef[cj - 1][s] * kernel[s];

            /* SVs of class cj: dual_coef row = ci */
            for (int s = class_start[cj]; s < class_start[cj + 1]; s++)
                f += har_svm_dual_coef[ci][s] * kernel[s];

            if (f > 0.0f)
                votes[ci]++;
            else
                votes[cj]++;

            pair++;
        }}
    }}

    /* ---- Step 4: confidence = votes / (n_classes - 1) ---- */
    float denom = (float)(HAR_NUM_CLASSES - 1);
    for (int c = 0; c < HAR_NUM_CLASSES; c++)
        probabilities[c] = (float)votes[c] / denom;
}}
"""

    # ------------------------------------------------------------------
    # OvR fallback (kept for backward compat with models lacking OvO data)
    # ------------------------------------------------------------------

    def _header_ovr(self, n_sv: int) -> str:
        return f"""\
#pragma once
#include "har_config.h"

/*
 * har_model.h — SVM (RBF kernel, One-vs-Rest fallback)
 * Support vectors: {n_sv}
 * Gamma: {self._gamma:.6g}
 *
 * Input:  features[HAR_NUM_FEATURES]  — ALREADY StandardScaler-normalized
 * Output: probabilities[HAR_NUM_CLASSES] — softmax(OvR decision scores)
 */
void har_model_predict(
    const float features[HAR_NUM_FEATURES],
    float probabilities[HAR_NUM_CLASSES]
);
"""

    def _impl_ovr(self, n_sv: int) -> str:
        if n_sv == 0:
            return self._empty_impl()

        sv_rows = []
        for sv in self._sv:
            vals = ", ".join(f"{v:.{self.precision}f}f" for v in sv)
            sv_rows.append(f"    {{{vals}}}")
        sv_arr = ",\n".join(sv_rows)

        dc_rows = []
        for row in self._dual_coef:
            vals = ", ".join(f"{v:.{self.precision}f}f" for v in row)
            dc_rows.append(f"    {{{vals}}}")
        dc_arr = ",\n".join(dc_rows)

        ic_vals = ", ".join(
            f"{v:.{self.precision}f}f" for v in self._intercept)

        return f"""\
#include "har_model.h"
#include <math.h>

#define HAR_SVM_N_SV {n_sv}

static const float har_svm_sv[HAR_SVM_N_SV][HAR_NUM_FEATURES] = {{
{sv_arr}
}};

static const float har_svm_dual_coef[HAR_NUM_CLASSES][HAR_SVM_N_SV] = {{
{dc_arr}
}};

static const float har_svm_intercept[HAR_NUM_CLASSES] = {{{ic_vals}}};

static const float HAR_SVM_GAMMA = {self._gamma:.{self.precision}f}f;

{self._softmax_code()}

void har_model_predict(
    const float features[HAR_NUM_FEATURES],
    float probabilities[HAR_NUM_CLASSES]
) {{
    float scores[HAR_NUM_CLASSES];
    for (int c = 0; c < HAR_NUM_CLASSES; c++) {{
        float s = har_svm_intercept[c];
        for (int sv = 0; sv < HAR_SVM_N_SV; sv++) {{
            /* RBF kernel: exp(-gamma * ||features - sv||^2) */
            float dist2 = 0.0f;
            for (int i = 0; i < HAR_NUM_FEATURES; i++) {{
                float d = features[i] - har_svm_sv[sv][i];
                dist2 += d * d;
            }}
            s += har_svm_dual_coef[c][sv] * expf(-HAR_SVM_GAMMA * dist2);
        }}
        scores[c] = s;
    }}
    _softmax(scores, HAR_NUM_CLASSES);
    for (int c = 0; c < HAR_NUM_CLASSES; c++)
        probabilities[c] = scores[c];
}}
"""

    def _empty_impl(self) -> str:
        return f"""\
#include "har_model.h"
#include <math.h>
/* No SVM data — uniform prediction */
void har_model_predict(const float features[HAR_NUM_FEATURES],
                       float probabilities[HAR_NUM_CLASSES]) {{
    (void)features;
    for(int i=0;i<HAR_NUM_CLASSES;i++) probabilities[i]=1.0f/(float)HAR_NUM_CLASSES;
}}
"""
