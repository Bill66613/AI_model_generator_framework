"""Abstract PlatformSketch base class."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class PlatformSketch(ABC):
    """
    Abstract base class for platform-specific sketch generation.

    Subclasses generate the thin user sketch ({sketch_name}.ino).

    The sketch must only:
      1. Include har_classifier.h
      2. Initialise the board + IMU
      3. Maintain a rolling sensor window buffer
      4. Call har_classify() on every inference step
      5. Output results via serial (CSV format)

    No feature extraction, no model code — that lives in har_features.cpp
    and har_model.cpp respectively.
    """

    def __init__(
        self,
        platform: str,
        sketch_name: str,
        classes: List[str],
        window_size: int,
        sampling_rate: int,
        overlap: float,
        smoothing_window: int,
    ):
        self.platform = platform
        self.sketch_name = sketch_name
        self.classes = classes
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.overlap = overlap
        self.smoothing_window = smoothing_window
        # Number of new samples to collect before running inference
        self.step_size = max(1, int(window_size * (1.0 - overlap)))

    @abstractmethod
    def generate(self) -> str:
        """Return the .ino file content."""
