"""models package — provides create_model_block() factory."""

from __future__ import annotations
from typing import Dict, Any, List

from .base import ModelBlock
from .random_forest import RandomForestModelBlock
from .neural_network import NeuralNetworkModelBlock
from .svm import SVMModelBlock


def create_model_block(
    model_data: Dict[str, Any],
    feature_names: List[str],
    classes: List[str],
    precision: int = 4,
    deployment_approach: str = "direct",
    quantization: str = "none",
    platform: str = "arduino",
    window_size: int = 150,
    sampling_rate: int = 100,
) -> ModelBlock:
    """Return the appropriate ModelBlock for the given model type and deployment approach."""
    model_type = model_data.get("model_type", "unknown")

    # TFLite Micro deployment — wraps any model type in the TFLite interpreter
    if deployment_approach == "tflite_micro":
        from .tflite import TFLiteModelBlock
        return TFLiteModelBlock(
            model_data, feature_names, classes, precision,
            quantization=quantization, platform=platform,
            window_size=window_size, sampling_rate=sampling_rate,
        )

    # ONNX Runtime deployment
    if deployment_approach == "onnx_runtime":
        from .onnx_model import ONNXModelBlock
        return ONNXModelBlock(
            model_data, feature_names, classes, precision,
            quantization=quantization, platform=platform,
        )

    # Direct deployment — model-type-specific hand-coded C++
    if model_type in ("random_forest",):
        return RandomForestModelBlock(model_data, feature_names, classes, precision)

    if model_type in ("neural_network", "pytorch_mlp"):
        return NeuralNetworkModelBlock(model_data, feature_names, classes, precision,
                                       quantization=quantization)

    if model_type in ("svm",):
        return SVMModelBlock(model_data, feature_names, classes, precision,
                             quantization=quantization)

    if model_type in ("pytorch_cnn", "pytorch_cnn2d"):
        from .cnn import CNNModelBlock
        return CNNModelBlock(
            model_data, feature_names, classes, precision,
            window_size=window_size,
            sampling_rate=sampling_rate,
            platform=platform,
            quantization=quantization,
        )

    # Fallback: minimal stub that compiles but always returns class 0
    return _StubModelBlock(model_data, feature_names, classes, precision, model_type)


class _StubModelBlock(ModelBlock):
    def __init__(self, model_data, feature_names, classes, precision, model_type):
        super().__init__(model_data, feature_names, classes, precision)
        self._mt = model_type

    def generate(self):
        n_cls = len(self.classes)
        stub_probs = ", ".join(["1.0f"] + ["0.0f"] * (n_cls - 1)) if n_cls > 1 else "1.0f"
        header = f"""\
#pragma once
#include "har_config.h"
/* STUB: model type '{self._mt}' not supported by v2 generator */
void har_model_predict(const float features[HAR_NUM_FEATURES],
                       float probabilities[HAR_NUM_CLASSES]);
"""
        impl = f"""\
#include "har_model.h"
/* STUB implementation — always predicts class 0 */
void har_model_predict(const float features[HAR_NUM_FEATURES],
                       float probabilities[HAR_NUM_CLASSES]) {{
    (void)features;
    float p[{n_cls}] = {{{stub_probs}}};
    for (int i = 0; i < HAR_NUM_CLASSES; i++) probabilities[i] = p[i];
}}
"""
        return header, impl

    @property
    def model_type_id(self):
        return "stub"
