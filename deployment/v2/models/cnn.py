"""
CNNModelBlock — generates har_model.h and har_model.cpp for 1D-CNN models.

The CNN operates on raw sensor windows (no manual feature extraction).
When used with FeatureBlock in CNN mode, features[] is a flat copy of the
sensor window (WINDOW_SIZE * N_CHANNELS values), and har_model_predict()
reshapes it internally before running the CNN forward pass.

Layer types supported (from PyTorchTrainer.export_cnn_weights):
  conv1d    — Conv1D + fused ReLU
  maxpool1d — MaxPool1D (stride = kernel_size)
  dense     — Fully-connected (last layer: no ReLU; others: ReLU)

Global average pooling is applied between the final conv/pool layer
and the first dense layer.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple

import numpy as np

from .base import ModelBlock


def _wrap_vals(vals: str, width: int = 100) -> str:
    """Wrap C literal value list at ~width chars for readability.

    Intermediate lines keep their trailing comma so the C array stays valid:
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f
    """
    parts = [p.strip() for p in vals.split(",") if p.strip()]
    lines = []
    current: list = []
    current_len = 0
    for p in parts:
        token = p + ", "
        if current_len + len(token) > width and current:
            # Intermediate line — strip trailing space but KEEP the comma
            lines.append("    " + "".join(current).rstrip())
            current = []
            current_len = 0
        current.append(token)
        current_len += len(token)
    if current:
        # Last line — strip trailing comma and space
        lines.append("    " + "".join(current).rstrip().rstrip(","))
    return "\n".join(lines)


def _wrap_int_vals(vals: str, width: int = 120) -> str:
    """Like _wrap_vals but for integer (int8) arrays — more values per line."""
    return _wrap_vals(vals, width=width)


class CNNModelBlock(ModelBlock):
    """Model block for pytorch_cnn / pytorch_cnn2d models."""

    def __init__(
        self,
        model_data: Dict[str, Any],
        feature_names: List[str],
        classes: List[str],
        precision: int = 4,
        window_size: int = 150,
        sampling_rate: int = 100,
        platform: str = "arduino",
        confidence_threshold: float = 0.6,
        quantization: str = "none",
        **_,
    ):
        super().__init__(model_data, feature_names, classes, precision)
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.platform = platform
        self.confidence_threshold = confidence_threshold
        self.use_int8 = quantization in ("int8",)
        # Derive n_channels from feature_names (cnn_in_{t}_{c} pattern) or fallback
        self.n_channels = model_data.get("n_channels", 6)
        self.layers: List[Dict[str, Any]] = []
        self._extract_layers()

    @property
    def model_type_id(self) -> str:
        return "cnn"

    # ------------------------------------------------------------------
    # Layer weight extraction
    # ------------------------------------------------------------------

    def _extract_layers(self):
        """Extract CNN layer descriptions from model_data."""
        # 1. Try live model object
        model_obj = self.model_data.get("model_object")
        if model_obj is not None:
            try:
                if hasattr(model_obj, "_pytorch_trainer"):
                    cnn_export = model_obj._pytorch_trainer.export_cnn_weights()
                    self.layers = cnn_export.get("layers", [])
                    if self.layers:
                        self._normalise_layers()
                        return
            except Exception:
                pass

        # 2. Weights stored at save time
        cnn_weights = self.model_data.get("cnn_weights", {})
        if cnn_weights:
            self.layers = cnn_weights.get("layers", [])
            if self.layers:
                self._normalise_layers()
                return

        # 3. Legacy key
        pytorch_cnn_weights = self.model_data.get("pytorch_cnn_weights", {})
        if pytorch_cnn_weights:
            self.layers = pytorch_cnn_weights.get("layers", [])
            if self.layers:
                self._normalise_layers()

    def _normalise_layers(self):
        supported = {"conv1d", "maxpool1d", "dense"}
        unsupported = sorted(
            {layer.get("type", "") for layer in self.layers
             if layer.get("type") not in supported}
        )
        if unsupported:
            raise NotImplementedError(
                "Direct C++ code generation for these CNN layers is not supported yet: "
                f"{', '.join(unsupported)}. "
                "Please use pytorch_cnn for direct deployment, or use a different deployment approach."
            )

        for layer in self.layers:
            t = layer.get("type", "")
            if t == "conv1d":
                layer.setdefault("padding", 0)
                w = layer.get("weights", [[[]]])
                layer.setdefault("out_channels", len(w))
                layer.setdefault("in_channels", len(w[0]) if w else 0)
                layer.setdefault("kernel_size", len(w[0][0]) if w and w[0] else 3)
            elif t == "maxpool1d":
                layer.setdefault("kernel_size", 2)
            elif t == "dense":
                layer.setdefault("in_features", 0)
                layer.setdefault("out_features", 0)

    # ------------------------------------------------------------------
    # Buffer size
    # ------------------------------------------------------------------

    def _compute_max_buf(self) -> int:
        seq = self.window_size
        ch = self.n_channels
        max_buf = seq * ch
        for layer in self.layers:
            t = layer.get("type")
            if t == "conv1d":
                ch = layer["out_channels"]
                max_buf = max(max_buf, seq * ch)
            elif t == "maxpool1d":
                seq = seq // layer["kernel_size"]
                max_buf = max(max_buf, seq * ch)
            elif t == "dense":
                max_buf = max(max_buf, layer.get("in_features", 0),
                              layer.get("out_features", 0))
        return max_buf

    # ------------------------------------------------------------------
    # C array formatters
    # ------------------------------------------------------------------

    def _fmt_1d(self, arr: np.ndarray, name: str) -> str:
        prec = self.precision
        vals = ", ".join(f"{v:.{prec}f}" for v in arr.flat)
        return (f"static const float {name}[{arr.size}] = {{\n"
                f"{_wrap_vals(vals)}\n}};")

    def _fmt_2d(self, arr: np.ndarray, name: str) -> str:
        prec = self.precision
        vals = ", ".join(f"{v:.{prec}f}" for v in arr.flat)
        return (f"/* shape ({arr.shape[0]}, {arr.shape[1]}) */\n"
                f"static const float {name}[{arr.size}] = {{\n"
                f"{_wrap_vals(vals)}\n}};")

    def _fmt_3d(self, arr: np.ndarray, name: str) -> str:
        prec = self.precision
        vals = ", ".join(f"{v:.{prec}f}" for v in arr.flat)
        return (f"/* shape ({arr.shape[0]}, {arr.shape[1]}, {arr.shape[2]}) */\n"
                f"static const float {name}[{arr.size}] = {{\n"
                f"{_wrap_vals(vals)}\n}};")

    def _fmt_int8(self, arr: np.ndarray, name: str) -> str:
        """Per-tensor INT8 quantization: store as int8_t + float scale.

        scale = max(|W|) / 127  — safe dequant: w_float = int8_val * scale
        Biases are NOT quantized (kept as float — they are small).
        """
        flat = np.asarray(arr, dtype=np.float32).flatten()
        max_abs = float(np.max(np.abs(flat)))
        scale = max_abs / 127.0 if max_abs > 1e-12 else 1.0
        q = np.clip(np.round(flat / scale), -127, 127).astype(np.int8)
        vals = ", ".join(str(int(v)) for v in q)
        shape_comment = ""
        if arr.ndim > 1:
            shape_comment = f"/* shape {arr.shape} — int8 quantized */\n"
        return (
            f"{shape_comment}"
            f"static const float {name}_scale = {scale:.{self.precision + 2}f}f;\n"
            f"static const int8_t {name}[{arr.size}] = {{\n"
            f"{_wrap_int_vals(vals)}\n}};"
        )

    # ------------------------------------------------------------------
    # Weight arrays
    # ------------------------------------------------------------------

    def _weight_arrays(self) -> str:
        if not self.layers:
            return "/* No CNN weights found — stub model */\n"
        parts: List[str] = []
        for i, layer in enumerate(self.layers):
            tag = f"l{i}"
            t = layer.get("type")
            if t == "conv1d":
                w = np.array(layer["weights"], dtype=np.float32)   # (out_ch, in_ch, k)
                b = np.array(layer["bias"], dtype=np.float32)       # (out_ch,)
                if self.use_int8:
                    parts.append(self._fmt_int8(w, f"{tag}_w"))
                else:
                    parts.append(self._fmt_3d(w, f"{tag}_w"))
                parts.append(self._fmt_1d(b, f"{tag}_b"))           # bias stays float
            elif t == "dense":
                w = np.array(layer["weights"], dtype=np.float32)    # (in, out)
                b = np.array(layer["bias"], dtype=np.float32)       # (out,)
                if self.use_int8:
                    parts.append(self._fmt_int8(w, f"{tag}_w"))
                else:
                    parts.append(self._fmt_2d(w, f"{tag}_w"))
                parts.append(self._fmt_1d(b, f"{tag}_b"))           # bias stays float
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Layer function bodies
    # ------------------------------------------------------------------

    def _layer_functions(self) -> str:
        if self.use_int8:
            return """\
/* int8 weight variants — weights stored as int8_t, dequantized on the fly */
static void cnn_conv1d(const float *inp, float *out,
                       const int8_t *weights, float w_scale, const float *bias,
                       int seq_len, int in_ch, int out_ch,
                       int kernel_size, int padding) {
    for (int o = 0; o < out_ch; o++) {
        for (int t = 0; t < seq_len; t++) {
            float sum = bias[o];
            for (int ic = 0; ic < in_ch; ic++) {
                for (int k = 0; k < kernel_size; k++) {
                    int pos = t - padding + k;
                    if (pos >= 0 && pos < seq_len) {
                        sum += inp[pos * in_ch + ic]
                             * ((float)weights[(o * in_ch + ic) * kernel_size + k] * w_scale);
                    }
                }
            }
            out[t * out_ch + o] = sum > 0.0f ? sum : 0.0f; /* fused ReLU */
        }
    }
}

static void cnn_maxpool(const float *inp, float *out,
                        int seq_len, int channels, int pool_k) {
    int out_len = seq_len / pool_k;
    for (int t = 0; t < out_len; t++) {
        for (int c = 0; c < channels; c++) {
            float mx = -1e30f;
            for (int p = 0; p < pool_k; p++) {
                float v = inp[(t * pool_k + p) * channels + c];
                if (v > mx) mx = v;
            }
            out[t * channels + c] = mx;  /* output layout: [out_len][channels] */
        }
    }
}

static void cnn_global_avg_pool(const float *inp, float *out,
                                int seq_len, int channels) {
    for (int c = 0; c < channels; c++) {
        float s = 0.0f;
        for (int t = 0; t < seq_len; t++) s += inp[t * channels + c];
        out[c] = s / (float)seq_len;
    }
}

static void cnn_dense(const float *inp, float *out,
                      const int8_t *weights, float w_scale, const float *bias,
                      int in_f, int out_f, int apply_relu) {
    for (int o = 0; o < out_f; o++) {
        float s = bias[o];
        for (int i = 0; i < in_f; i++)
            s += inp[i] * ((float)weights[i * out_f + o] * w_scale);
        out[o] = (apply_relu && s < 0.0f) ? 0.0f : s;
    }
}
"""
        return """\
static void cnn_conv1d(const float *inp, float *out,
                       const float *weights, const float *bias,
                       int seq_len, int in_ch, int out_ch,
                       int kernel_size, int padding) {
    for (int o = 0; o < out_ch; o++) {
        for (int t = 0; t < seq_len; t++) {
            float sum = bias[o];
            for (int ic = 0; ic < in_ch; ic++) {
                for (int k = 0; k < kernel_size; k++) {
                    int pos = t - padding + k;
                    if (pos >= 0 && pos < seq_len) {
                        sum += inp[pos * in_ch + ic]
                             * weights[(o * in_ch + ic) * kernel_size + k];
                    }
                }
            }
            out[t * out_ch + o] = sum > 0.0f ? sum : 0.0f; /* fused ReLU */
        }
    }
}

static void cnn_maxpool(const float *inp, float *out,
                        int seq_len, int channels, int pool_k) {
    int out_len = seq_len / pool_k;
    for (int t = 0; t < out_len; t++) {
        for (int c = 0; c < channels; c++) {
            float mx = -1e30f;
            for (int p = 0; p < pool_k; p++) {
                float v = inp[(t * pool_k + p) * channels + c];
                if (v > mx) mx = v;
            }
            out[t * channels + c] = mx;  /* output layout: [out_len][channels] */
        }
    }
}

static void cnn_global_avg_pool(const float *inp, float *out,
                                int seq_len, int channels) {
    for (int c = 0; c < channels; c++) {
        float s = 0.0f;
        for (int t = 0; t < seq_len; t++) s += inp[t * channels + c];
        out[c] = s / (float)seq_len;
    }
}

static void cnn_dense(const float *inp, float *out,
                      const float *weights, const float *bias,
                      int in_f, int out_f, int apply_relu) {
    for (int o = 0; o < out_f; o++) {
        float s = bias[o];
        for (int i = 0; i < in_f; i++) s += inp[i] * weights[i * out_f + o];
        out[o] = (apply_relu && s < 0.0f) ? 0.0f : s;
    }
}
"""

    # ------------------------------------------------------------------
    # Forward-pass body (inside har_model_predict)
    # ------------------------------------------------------------------

    def _forward_pass(self) -> str:
        lines: List[str] = []
        max_buf = self._compute_max_buf()
        is_esp32 = self.platform in ("esp32", "m5stack", "m5stickc", "m5stick")
        if is_esp32:
            lines.append(f"    static float *buf_a = nullptr, *buf_b = nullptr;")
            lines.append(f"    if (!buf_a) {{")
            lines.append(f"        buf_a = (float*)malloc({max_buf} * sizeof(float));")
            lines.append(f"        buf_b = (float*)malloc({max_buf} * sizeof(float));")
            lines.append(f"        if (!buf_a || !buf_b) {{ return; }}")
            lines.append(f"    }}")
        else:
            lines.append(f"    static float buf_a[{max_buf}];")
            lines.append(f"    static float buf_b[{max_buf}];")
        lines.append("")

        # Copy features into buf_a (memcpy is faster than element-wise loop).
        # features[] is in [T][C] layout which is what conv1d expects.
        lines.append(f"    memcpy(buf_a, features, HAR_NUM_FEATURES * sizeof(float));")
        lines.append("")

        src, dst = "buf_a", "buf_b"
        seq_expr = "HAR_WINDOW_SIZE"
        ch_expr  = "HAR_N_CHANNELS"
        dense_idx = 0
        total_dense = sum(1 for l in self.layers if l.get("type") == "dense")

        for i, layer in enumerate(self.layers):
            t = layer.get("type")
            tag = f"l{i}"
            if t == "conv1d":
                oc = layer["out_channels"]
                ic = layer["in_channels"]
                ks = layer["kernel_size"]
                pd = layer["padding"]
                lines.append(f"    /* Conv1D layer {i}: {ic}\u2192{oc}, k={ks}, pad={pd} */")
                if self.use_int8:
                    lines.append(f"    cnn_conv1d({src}, {dst}, {tag}_w, {tag}_w_scale, {tag}_b,")
                else:
                    lines.append(f"    cnn_conv1d({src}, {dst}, {tag}_w, {tag}_b,")
                lines.append(f"               {seq_expr}, {ic}, {oc}, {ks}, {pd});")
                ch_expr = str(oc)
                src, dst = dst, src
                lines.append("")
            elif t == "maxpool1d":
                pk = layer["kernel_size"]
                lines.append(f"    /* MaxPool1D layer {i}: k={pk} */")
                lines.append(f"    cnn_maxpool({src}, {dst}, {seq_expr}, {ch_expr}, {pk});")
                seq_expr = f"({seq_expr} / {pk})"
                src, dst = dst, src
                lines.append("")
            elif t == "dense":
                dense_idx += 1
                is_last = (dense_idx == total_dense)
                in_f  = layer["in_features"]
                out_f = layer["out_features"]
                relu  = "0" if is_last else "1"
                w_scale_arg = f", {tag}_w_scale" if self.use_int8 else ""

                if dense_idx == 1:
                    lines.append(f"    /* GlobalAvgPool \u2192 Dense layer {i}: {in_f}\u2192{out_f} */")
                    lines.append(f"    float gap_{i}[{in_f}];")
                    lines.append(f"    cnn_global_avg_pool({src}, gap_{i}, {seq_expr}, {ch_expr});")
                    lines.append(f"    float d{dense_idx}[{out_f}];")
                    lines.append(f"    cnn_dense(gap_{i}, d{dense_idx}, {tag}_w{w_scale_arg}, {tag}_b, {in_f}, {out_f}, {relu});")
                    prev_var = f"d{dense_idx}"
                else:
                    lines.append(f"    /* Dense layer {i}: {in_f}\u2192{out_f} */")
                    lines.append(f"    float d{dense_idx}[{out_f}];")
                    lines.append(f"    cnn_dense({prev_var}, d{dense_idx}, {tag}_w{w_scale_arg}, {tag}_b, {in_f}, {out_f}, {relu});")
                    prev_var = f"d{dense_idx}"
                lines.append("")

        # Softmax over final output
        final_out_f = self.layers[-1]["out_features"] if self.layers and self.layers[-1].get("type") == "dense" else len(self.classes)
        n_cls = len(self.classes)
        final_var = f"d{dense_idx}" if dense_idx else "buf_a"
        lines.append(f"    /* Softmax */")
        lines.append(f"    float max_l = {final_var}[0];")
        lines.append(f"    for (int i = 1; i < HAR_NUM_CLASSES; i++)")
        lines.append(f"        if ({final_var}[i] > max_l) max_l = {final_var}[i];")
        lines.append(f"    float sum_e = 0.0f;")
        lines.append(f"    for (int i = 0; i < HAR_NUM_CLASSES; i++) {{")
        lines.append(f"        probabilities[i] = expf({final_var}[i] - max_l);")
        lines.append(f"        sum_e += probabilities[i];")
        lines.append(f"    }}")
        lines.append(f"    for (int i = 0; i < HAR_NUM_CLASSES; i++)")
        lines.append(f"        probabilities[i] /= sum_e;")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Defines for layer geometry
    # ------------------------------------------------------------------

    def _layer_defines(self) -> str:
        parts: List[str] = []
        for i, layer in enumerate(self.layers):
            t = layer.get("type")
            tag = f"CNN_L{i}"
            if t == "conv1d":
                parts.append(f"#define {tag}_OUT_CH   {layer['out_channels']}")
                parts.append(f"#define {tag}_IN_CH    {layer['in_channels']}")
                parts.append(f"#define {tag}_KSIZE    {layer['kernel_size']}")
            elif t == "maxpool1d":
                parts.append(f"#define {tag}_POOL     {layer['kernel_size']}")
            elif t == "dense":
                parts.append(f"#define {tag}_IN       {layer['in_features']}")
                parts.append(f"#define {tag}_OUT      {layer['out_features']}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # ModelBlock.generate() interface
    # ------------------------------------------------------------------

    def generate(self) -> Tuple[str, str]:
        n_layers = len(self.layers)
        layer_summary = (
            " → ".join(
                f"{l.get('type','?')}({l.get('out_channels', l.get('out_features','?'))})"
                for l in self.layers
            ) if self.layers else "no layers"
        )

        header = f"""\
#pragma once
#include "har_config.h"
#include <math.h>
{"#include <stdint.h>" if self.use_int8 else ""}
/*
 * har_model.h — 1D-CNN model
 * Generated by HAR Edge Framework v2
 *
 * Architecture: {layer_summary}
 * Input: HAR_WINDOW_SIZE × HAR_N_CHANNELS raw sensor values (flattened)
 *        features[i*HAR_N_CHANNELS + c]  (c = aX,aY,aZ,gX,gY,gZ)
 * Output: probabilities[HAR_NUM_CLASSES] (softmax)
 *
 * HAR_NUM_FEATURES == HAR_WINDOW_SIZE * HAR_N_CHANNELS (set in har_config.h)
 */

/* Layer geometry */
{self._layer_defines()}

/**
 * Run 1D-CNN forward pass.
 * features[] is a flattened copy of the raw sensor window
 * (HAR_NUM_FEATURES = HAR_WINDOW_SIZE * HAR_N_CHANNELS elements).
 * probabilities[] is filled with softmax class probabilities.
 */
void har_model_predict(const float features[HAR_NUM_FEATURES],
                       float probabilities[HAR_NUM_CLASSES]);
"""

        impl = f"""\
#include "har_model.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
{"#include <stdint.h>" if self.use_int8 else ""}
/* ---- Weight arrays ---- */
{self._weight_arrays()}

/* ---- Layer kernel functions ---- */
{self._layer_functions()}

/* ---- Forward pass ---- */
void har_model_predict(const float features[HAR_NUM_FEATURES],
                       float probabilities[HAR_NUM_CLASSES]) {{
{self._forward_pass()}
}}
"""
        return header, impl
