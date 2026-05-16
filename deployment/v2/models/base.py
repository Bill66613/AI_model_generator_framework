"""Abstract ModelBlock base class."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple


class ModelBlock(ABC):
    """
    Abstract base class for model-specific code blocks.

    Subclasses generate har_model.h and har_model.cpp.

    Contract:
      - har_model.h must declare:
            void har_model_predict(const float features[HAR_NUM_FEATURES],
                                   float probabilities[HAR_NUM_CLASSES]);
      - Input 'features' are ALREADY StandardScaler-normalized.
      - Output 'probabilities' must be non-negative and sum to ~1.0
        (softmax applied inside the model, or vote fractions for RF).
    """

    def __init__(
        self,
        model_data: Dict[str, Any],
        feature_names: List[str],
        classes: List[str],
        precision: int,
    ):
        self.model_data = model_data
        self.feature_names = feature_names
        self.classes = classes
        self.precision = precision
        self.n_features = len(feature_names)
        self.n_classes = len(classes)

    @abstractmethod
    def generate(self) -> Tuple[str, str]:
        """Return (har_model.h content, har_model.cpp content)."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _float_arr(self, values, per_row: int = 8) -> str:
        """Format a 1-D list of floats as a C array initializer."""
        rows = []
        for i in range(0, len(values), per_row):
            row = values[i: i + per_row]
            rows.append(
                "    " + ", ".join(f"{v:.{self.precision}f}f" for v in row))
        return ",\n".join(rows)

    def _softmax_code(self) -> str:
        """Inline softmax utility used by NN and SVM."""
        return """\
static void _softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    if (sum > 0.0f) for (int i = 0; i < n; i++) x[i] /= sum;
}
"""
