---
name: parity-validation
description: 'Validate training-deployment parity between Python feature extraction and C++ code generators. Use when checking formula consistency, verifying kurtosis/skewness/ZCR/MCR formulas, comparing Python vs C++ output, or debugging prediction mismatches between trained model and generated embedded code.'
argument-hint: 'Describe what to validate (e.g., "check skewness formula parity")'
---

# Training-Deployment Parity Validation

## When to Use
- After modifying `utils/feature_extraction.py` or any `deployment/*_generator.py`
- Before merging code generator changes
- When device predictions differ from Python predictions
- When adding new statistical features

## The Parity Chain
```
Python (training)              C++ (deployment)              MicroPython
utils/feature_extraction.py ↔ deployment/base_generator.py ↔ deployment/micropython_generator.py
```

## Procedure

### 1. Formula Audit
Compare the 15 statistical features across all three implementations:

| Stat | Key Rule |
|------|----------|
| mean | sum/N |
| std | population std (ddof=0) |
| skewness | z-scores use population std |
| kurtosis | z-scores use population std, subtract 3 for excess |
| zero_crossings | count sign changes / (N-1) |
| mean_crossing_rate | count mean-crossings / (N-1) |
| rms | sqrt(sum(x²)/N) |
| energy | sum(x²)/N |

### 2. Feature Order Check
- Python: alphabetically-sorted DataFrame columns
- C++: computation order (acc_mag → gyro_mag → jerk for orientation_invariant)
- Mapping: `CodeGeneratorFactory.reorder_model_parameters()`

Verify with [reorder audit script](./scripts/audit_reorder.py).

### 3. Scaler Parity
Check that scaler mean/scale arrays are reordered to match C++ feature order.

### 4. Run Validation
```bash
python validate_deployment.py
python -m pytest tests/test_main.py -v
```

### 5. Check Known Issues
Read `academic-paper-vietnamese/TECHNICAL_FINDINGS.md` — 11 documented findings with code references.

## References
- [Parity checklist](./references/parity-checklist.md)
- [Feature order mapping](./references/feature-order-mapping.md)
