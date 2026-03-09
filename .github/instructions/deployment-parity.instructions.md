---
description: "Training-deployment parity rules for code generator files. Use when editing C++ template generation, feature extraction formulas, weight reordering, or scaler parameters."
applyTo: "deployment/**/*.py"
---
# Code Generator Parity Rules

## CRITICAL INVARIANT
Python feature extraction (`utils/feature_extraction.py`) and C++ templates (`deployment/base_generator.py`) must produce **identical** results for the same input.

## Before editing any generator:
1. Read `utils/feature_extraction.py` to understand the Python side
2. Read `deployment/base_generator.py` for the C++ templates
3. Check `academic-paper-vietnamese/TECHNICAL_FINDINGS.md` for known issues

## Formula rules:
- **Kurtosis/skewness**: Use population std (divide by N, not N-1) for z-score normalization
- **Zero-crossing rate**: Count sign changes / (N-1)
- **Mean-crossing rate**: Count mean-crossings / (N-1)
- **Feature order**: Python alphabetical → C++ computation order. Update `reorder_model_parameters()` if changed.

## After editing:
- If `base_generator.py` changed → check `micropython_generator.py` (has separate feature extraction)
- Run `python -m pytest tests/ -v`
- Document findings in `academic-paper-vietnamese/TECHNICAL_FINDINGS.md`
