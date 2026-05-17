"""NeuralNetwork model block — generates har_model.h/.cpp for MLP."""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from .base import ModelBlock


class NeuralNetworkModelBlock(ModelBlock):
    """
    Generates C++ forward-pass code for a multi-layer perceptron.

    Supports:
      - sklearn MLPClassifier  (model_data['weights'] dict)
      - pytorch_mlp export     (model_data['pytorch_coefs'] / ['pytorch_intercepts'])

    Architecture: Input → [Hidden layers with ReLU] → Output (softmax)

    All weight/bias arrays have already been reordered to match the C++
    feature extraction order by the factory before this generator runs.
    """

    def __init__(self, model_data, feature_names, classes, precision):
        super().__init__(model_data, feature_names, classes, precision)
        self._coefs, self._intercepts = self._extract_weights(model_data)

    # ------------------------------------------------------------------

    def generate(self) -> Tuple[str, str]:
        arch = " → ".join(
            [str(self.n_features)]
            + [str(len(b)) for b in self._intercepts[:-1]]
            + [str(self.n_classes)]
        )
        return self._header(arch), self._impl(arch)

    # ------------------------------------------------------------------

    def _header(self, arch: str) -> str:
        return f"""\
#pragma once
#include "har_config.h"

/*
 * har_model.h — Neural Network (MLP)
 * Architecture: {arch}
 * Activations:  ReLU (hidden), softmax (output)
 *
 * Input:  features[HAR_NUM_FEATURES]  — ALREADY StandardScaler-normalized
 * Output: probabilities[HAR_NUM_CLASSES] — softmax probabilities (sum to 1)
 */
void har_model_predict(
    const float features[HAR_NUM_FEATURES],
    float probabilities[HAR_NUM_CLASSES]
);
"""

    def _impl(self, arch: str) -> str:
        if not self._coefs:
            return self._empty_impl()

        # Emit weight/bias arrays for each layer
        layer_arrays = []
        layer_sizes = []

        for li, (W, b) in enumerate(zip(self._coefs, self._intercepts)):
            n_in, n_out = len(W), len(b)
            layer_sizes.append((n_in, n_out))

            # Weight matrix: W[n_in][n_out]  (input → neuron)
            rows = []
            for row in W:
                vals = ", ".join(f"{v:.{self.precision}f}f" for v in row)
                rows.append(f"    {{{vals}}}")
            w_arr = ",\n".join(rows)

            b_vals = ", ".join(f"{v:.{self.precision}f}f" for v in b)

            layer_arrays.append(f"""\
static const float har_nn_W{li}[{n_in}][{n_out}] = {{
{w_arr}
}};
static const float har_nn_b{li}[{n_out}] = {{{b_vals}}};""")

        n_layers = len(self._coefs)
        max_hidden = max((s[1] for s in layer_sizes[:-1]), default=0)

        # Build forward pass
        forward_steps = []
        for li in range(n_layers):
            n_in, n_out = layer_sizes[li]
            is_last = (li == n_layers - 1)
            in_buf = "features" if li == 0 else f"buf{(li-1) % 2}"
            out_buf = f"buf{li % 2}"
            act = "/* softmax applied below */" if is_last else "if (v < 0.0f) v = 0.0f;  /* ReLU */"
            forward_steps.append(f"""\
    /* Layer {li}: {n_in} → {n_out} */
    for (int o = 0; o < {n_out}; o++) {{
        float v = har_nn_b{li}[o];
        for (int i = 0; i < {n_in}; i++) v += {in_buf}[i] * har_nn_W{li}[i][o];
        {act}
        {"probabilities[o]" if is_last else f"{out_buf}[o]"} = v;
    }}""")

        forward_code = "\n".join(forward_steps)

        # Need two ping-pong buffers sized to max hidden layer
        buf_decl = ""
        if n_layers > 1 and max_hidden > 0:
            buf_decl = f"    float buf0[{max_hidden}], buf1[{max_hidden}];"

        return f"""\
#include "har_model.h"
#include <math.h>

{chr(10).join(layer_arrays)}

{self._softmax_code()}

void har_model_predict(
    const float features[HAR_NUM_FEATURES],
    float probabilities[HAR_NUM_CLASSES]
) {{
{buf_decl}
{forward_code}
    _softmax(probabilities, HAR_NUM_CLASSES);
}}
"""

    # ------------------------------------------------------------------

    def _extract_weights(self, model_data):
        """
        Extract (coefs, intercepts) from model_data.

        Returns:
          coefs:      list of 2-D lists, shape [n_layers][n_in][n_out]
          intercepts: list of 1-D lists, shape [n_layers][n_out]
        """
        # Path 1: pytorch_mlp export
        if "pytorch_coefs" in model_data and "pytorch_intercepts" in model_data:
            coefs = [np.array(c).tolist() for c in model_data["pytorch_coefs"]]
            intercepts = [np.array(b).tolist()
                          for b in model_data["pytorch_intercepts"]]
            return coefs, intercepts

        # Path 2: sklearn-style dict
        weights = model_data.get("weights", {})
        if not weights:
            return [], []

        coefs = []
        intercepts = []

        # Support single hidden layer (legacy) and multi-layer
        if "input_weights" in weights and "hidden_biases" in weights:
            W0 = np.array(weights["input_weights"])
            b0 = np.array(weights["hidden_biases"])
            coefs.append(W0.tolist())
            intercepts.append(b0.tolist())

            if "output_weights" in weights and "output_biases" in weights:
                W1 = np.array(weights["output_weights"])
                b1 = np.array(weights["output_biases"])
                coefs.append(W1.tolist())
                intercepts.append(b1.tolist())

        return coefs, intercepts

    def _empty_impl(self) -> str:
        return f"""\
#include "har_model.h"
#include <math.h>
/* No weight data — uniform prediction */
static void _softmax(float *x, int n) {{
    float s=0; for(int i=0;i<n;i++) s+=1.0f;
    for(int i=0;i<n;i++) x[i]=1.0f/s;
}}
void har_model_predict(const float features[HAR_NUM_FEATURES],
                       float probabilities[HAR_NUM_CLASSES]) {{
    (void)features;
    for(int i=0;i<HAR_NUM_CLASSES;i++) probabilities[i]=1.0f/(float)HAR_NUM_CLASSES;
}}
"""
