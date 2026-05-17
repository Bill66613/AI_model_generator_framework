"""
HAR Code Generator v2 — Clean modular architecture.

Generated file structure (per deployment):
  har_config.h        — All #defines (HAR_NUM_FEATURES, HAR_WINDOW_SIZE, …)
  har_features.h/.cpp — Feature extraction (one implementation per feature mode)
  har_model.h/.cpp    — Model-specific weights + har_model_predict()
  har_classifier.h/.cpp — Complete inference chain: features→scale→predict→threshold
  {sketch}.ino        — Platform sketch (thin: sensor read + har_classify() call)

Adding a new model type: implement ModelBlock subclass in models/
Adding a new platform:   implement PlatformSketch subclass in platforms/
Everything else stays the same.
"""

from .generator import HARCodeGenerator

__all__ = ["HARCodeGenerator"]
