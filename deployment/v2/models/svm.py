"""SVM model block — generates har_model.h/.cpp for SVM (RBF kernel, OvR)."""

from __future__ import annotations
from typing import Dict, Any, List, Tuple

import numpy as np

from .base import ModelBlock


class SVMModelBlock(ModelBlock):
    """
    Generates C++ SVM (RBF kernel) inference code.

    Uses One-vs-Rest (OvR) coefficients which have already been converted
    from sklearn's OvO format by the factory's extract_svm_parameters().

    Decision function per class k:
      score_k = sum_sv( dual_coef[k][sv] * RBF(features, support_vectors[sv]) ) + intercept[k]
    Predicted class = argmax(softmax(scores))

    RBF kernel: K(x, sv) = exp( -gamma * ||x - sv||² )
    """

    def __init__(self, model_data, feature_names, classes, precision):
        super().__init__(model_data, feature_names, classes, precision)
        self._sv = np.array(model_data.get("support_vectors", []))
        self._dual_coef = np.array(model_data.get("dual_coefficients", []))
        self._intercept = np.array(model_data.get("intercept", []))
        self._gamma = float(model_data.get("gamma", 0.1))

    # ------------------------------------------------------------------

    def generate(self) -> Tuple[str, str]:
        n_sv = len(self._sv) if len(self._sv) > 0 else 0
        return self._header(n_sv), self._impl(n_sv)

    # ------------------------------------------------------------------

    def _header(self, n_sv: int) -> str:
        return f"""\
#pragma once
#include "har_config.h"

/*
 * har_model.h — SVM (RBF kernel, One-vs-Rest)
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

    def _impl(self, n_sv: int) -> str:
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
    for (int c = 0; c < HAR_NUM_CLASSES; c++) probabilities[c] = scores[c];
    _softmax(probabilities, HAR_NUM_CLASSES);
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
