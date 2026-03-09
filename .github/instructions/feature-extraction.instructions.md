---
description: "Feature extraction parity rules for Python statistical computations. Use when modifying feature computation, adding new features, or changing statistical formulas."
applyTo: "utils/feature_extraction.py"
---
# Feature Extraction Parity Rules

## This file is the Python side of the training-deployment parity chain.
Any formula change here MUST be mirrored in:
- `deployment/base_generator.py` (C++ templates)
- `deployment/micropython_generator.py` (MicroPython — separate implementation)

## 15 stats per signal
mean, std, min, max, range, median, q25, q75, iqr, skewness, kurtosis, rms, energy, zero_crossings, mean_crossing_rate

## 6 feature modes
| Mode | Count |
|------|-------|
| `orientation_invariant_time_only` | 33 |
| `orientation_invariant` | 47 |
| `time_domain` | 90 |
| `all` | 138 |
| `frequency_domain` | 48 |
| `raw` | 6 |

## Critical formulas
- Kurtosis/skewness: population std (N, not N-1) for z-scores
- Features are sorted alphabetically in Python DataFrames
- C++ extracts in computation order — `reorder_model_parameters()` handles mapping
