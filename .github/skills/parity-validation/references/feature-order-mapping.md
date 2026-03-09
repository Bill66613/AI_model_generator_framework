# Feature Order Mapping

## orientation_invariant_time_only (33 features)

### Python (alphabetical) → C++ (computation) order

**C++ computation order** (11 stats × 3 signals):
```
acc_mag:  mean, std, min, max, range, median, q25, q75, iqr, skewness, kurtosis, rms, energy, zero_crossings, mean_crossing_rate
gyro_mag: mean, std, min, max, range, median, q25, q75, iqr, skewness, kurtosis, rms, energy, zero_crossings, mean_crossing_rate  
jerk_mag: mean, std, min, max, range, median, q25, q75, iqr, skewness, kurtosis, rms, energy, zero_crossings, mean_crossing_rate
```

**Python alphabetical**: sorted by feature name string (e.g., `acc_mag_energy`, `acc_mag_iqr`, `acc_mag_kurtosis`, ...)

**Mapping**: `CodeGeneratorFactory.reorder_model_parameters()` builds the index mapping from alphabetical→computation and applies it to:
1. Model weights (NN: first layer weights columns, RF: feature_importances, SVM: support vectors)
2. Scaler mean array
3. Scaler scale array

## Critical Rule
If you add/remove a feature or change extraction order in C++, you MUST update `reorder_model_parameters()` in `deployment/code_generator_factory.py`.
