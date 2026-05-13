"""platforms package — provides create_platform_sketch() factory."""

from __future__ import annotations
from typing import Dict, Any, List

from .base import PlatformSketch
from .arduino import ArduinoSketch


def create_platform_sketch(
    platform: str,
    sketch_name: str,
    classes: List[str],
    window_size: int,
    sampling_rate: int,
    overlap: float,
    smoothing_window: int,
) -> PlatformSketch:
    """
    Return the appropriate PlatformSketch for the given platform string.

    Supported platforms (case-insensitive):
      arduino, esp32, m5stack, m5stick, seeed_xiao, nano_33, mkr_imu
      generic_cpp, generic_c  (fallback — no board-specific IMU)
    """
    p = platform.lower().replace("-", "_")

    # All Arduino-family boards share the same sketch template;
    # board-specific differences are inside the ArduinoSketch class.
    ARDUINO_FAMILY = {
        "arduino", "esp32", "m5stack", "m5stick", "m5stickc",
        "seeed_xiao", "nano_33", "mkr_imu", "arduino_imu",
        "arm_cortex_m",  # treated as generic Arduino-C++
    }

    if p in ARDUINO_FAMILY:
        return ArduinoSketch(
            platform=p,
            sketch_name=sketch_name,
            classes=classes,
            window_size=window_size,
            sampling_rate=sampling_rate,
            overlap=overlap,
            smoothing_window=smoothing_window,
        )

    # Fallback: generic Arduino-compatible C++ sketch
    return ArduinoSketch(
        platform="generic",
        sketch_name=sketch_name,
        classes=classes,
        window_size=window_size,
        sampling_rate=sampling_rate,
        overlap=overlap,
        smoothing_window=smoothing_window,
    )
