# CRITICAL BUG FIX: Feature Order Mismatch

## Problem Summary

**Training accuracy: 99.4%** → **Deployment: Completely broken predictions**

## Root Cause

The deployment code generator had the **wrong feature order** for orientation-invariant features. The neural network was trained with features in one order but the deployment code sent them in a completely different order, causing the model to receive scrambled inputs.

## Feature Order Mismatch

### Training Data Order (CORRECT)

```
0.  acc_mag_mean
1.  acc_mag_std
2.  acc_mag_rms
3.  acc_mag_energy
4.  gyro_mag_mean       ← Position 4 should be gyro_mean
5.  gyro_mag_std
6.  gyro_mag_rms
7.  gyro_mag_energy
8.  jerk_mag_mean
9.  jerk_mag_std
10. jerk_mag_rms
11. jerk_mag_energy
12. acc_mag_min
13. acc_mag_max
14. gyro_mag_min
15. gyro_mag_max
16. jerk_mag_max
```

### Deployment Code Order (WRONG - BEFORE FIX)

```
0.  acc_mag_mean
1.  acc_mag_std
2.  acc_mag_rms
3.  acc_mag_energy
4.  acc_mag_min          ← Position 4 was sending acc_min instead!
5.  acc_mag_max          ← Completely wrong from here onwards
6.  gyro_mag_mean
7.  gyro_mag_std
8.  gyro_mag_rms
9.  gyro_mag_energy
10. gyro_mag_min
11. gyro_mag_max
12. jerk_mag_mean
13. jerk_mag_std
14. jerk_mag_rms
15. jerk_mag_energy
16. jerk_mag_max
```

## Impact

The neural network expected `gyro_mag_mean` (value ~1.28) at position 4, but received `acc_mag_min` (value ~-0.49). Every input was scrambled, causing:

- Model couldn't recognize any activity correctly
- "Still" detected as "walking"
- Predictions appeared random despite perfect training accuracy

## Files Fixed

### 1. Deployment Code (User's Generated Files)

**File:** `persistent_data_new/generated/neural_network_models/har_neural_network_seeed_xiao_f17_c5_balanced/har_neural_network_seeed_xiao_f17_c5_balanced.cpp`

**Change:** Reordered feature extraction to match training data:

- Grouped acc features (mean, std, rms, energy) WITHOUT min/max
- Grouped gyro features (mean, std, rms, energy) WITHOUT min/max  
- Grouped jerk features (mean, std, rms, energy) WITHOUT max
- Added ALL min/max features at the end (positions 12-16)

### 2. Code Generator (For Future Models)

**File:** `deployment/base_generator.py`

**Change:** Fixed the `orientation_invariant` feature extraction template (lines 293-322) to generate features in the correct order.

## Next Steps

### Upload Corrected Files to XIAO Device

Copy these 3 files from `persistent_data_new/generated/neural_network_models/har_neural_network_seeed_xiao_f17_c5_balanced/`:

1. `har_neural_network_seeed_xiao_f17_c5_balanced.cpp` ✅ FIXED
2. `har_neural_network_seeed_xiao_f17_c5_balanced.h` (unchanged)
3. `har_neural_network_seeed_xiao_f17_c5_balanced.ino` (unchanged)

### Expected Results After Fix

With the correct feature order:

- ✅ Still activity → Class 1 (still)
- ✅ Walking → Class 2 (walking)  
- ✅ Running → Class 0 (running)
- ✅ Walking upstairs → Class 4 (walking_upstairs)
- ✅ Walking downstairs → Class 3 (walking_downstairs)

The model should now achieve near 98% accuracy matching the test accuracy from training.

## Verification

Run `check_feature_mismatch.py` to verify feature order matches between training and deployment.

## Lessons Learned

1. **Always verify feature order** between training and deployment
2. Code generators must **exactly match** the feature engineering pipeline
3. High training accuracy + poor deployment = **data mismatch**, not model quality
4. Unit tests should compare training feature extraction with deployment feature extraction

## Bug Origin

The code generator template grouped features by type (acc, gyro, jerk) and included min/max within each group. However, the training code extracted features in a different pattern:

- First: all mean/std/rms/energy stats
- Last: all min/max values

This subtle difference caused complete failure despite both extracting the same 17 features.
