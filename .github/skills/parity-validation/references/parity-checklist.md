# Parity Checklist

Use this checklist when validating training-deployment parity.

## Statistical Formulas (per signal)

| # | Feature | Python (`feature_extraction.py`) | C++ (`base_generator.py`) | MicroPython | Match? |
|---|---------|----------------------------------|---------------------------|-------------|--------|
| 1 | mean | `np.mean(x)` | `sum/count` | `sum/count` | ✅ |
| 2 | std | `np.std(x, ddof=0)` | population std | population std | ✅ |
| 3 | min | `np.min(x)` | loop min | loop min | ✅ |
| 4 | max | `np.max(x)` | loop max | loop max | ✅ |
| 5 | range | `max - min` | `max - min` | `max - min` | ✅ |
| 6 | median | `np.median(x)` | sorted middle | sorted middle | ✅ |
| 7 | q25 | `np.percentile(x, 25)` | sorted index | sorted index | ✅ |
| 8 | q75 | `np.percentile(x, 75)` | sorted index | sorted index | ✅ |
| 9 | iqr | `q75 - q25` | `q75 - q25` | `q75 - q25` | ✅ |
| 10 | skewness | population std z-scores | population std z-scores | population std z-scores | ✅ (Fixed: Finding §2) |
| 11 | kurtosis | population std z-scores - 3 | population std z-scores - 3 | population std z-scores - 3 | ✅ (Fixed: Finding §2) |
| 12 | rms | `sqrt(mean(x²))` | `sqrt(sum_sq/count)` | `sqrt(sum_sq/count)` | ✅ |
| 13 | energy | `mean(x²)` | `sum_sq/count` | `sum_sq/count` | ✅ |
| 14 | zero_crossings | sign changes / (N-1) | sign changes / (N-1) | sign changes / (N-1) | ✅ |
| 15 | mean_crossing_rate | mean-crossings / (N-1) | mean-crossings / (N-1) | mean-crossings / (N-1) | ✅ |

## Feature Ordering

| Mode | Python Order | C++ Order | Reorder Needed? |
|------|-------------|-----------|-----------------|
| `orientation_invariant_time_only` | alphabetical (33 features) | acc_mag→gyro_mag→jerk (11 stats each) | Yes |
| `orientation_invariant` | alphabetical (47 features) | computation order + DFT | Yes |
| `time_domain` | alphabetical (90 features) | per-axis computation | Yes |
| `all` | alphabetical (138 features) | time then frequency | Yes |

## Window Handling

| Aspect | Rule | Why |
|--------|------|-----|
| Padding | Edge-value replication | Zero-padding creates acc_magnitude=0 (physically impossible) |
| Minimum size | Target window size from config | Short windows get padded |

## Scaler

| Aspect | Rule |
|--------|------|
| Type | StandardScaler (mean/scale) |
| Fit on | Training data only |
| Reorder | Must match feature reorder |
| Precision | Depends on optimization mode (2-6 decimals) |
