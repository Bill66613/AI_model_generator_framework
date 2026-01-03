# CRITICAL BUG #2: Incorrect Scaler Parameters

## Problem

Even after fixing the feature order, the model still predicts everything as "running".

**Symptoms:**

- Device on table (still): acc=9.52, gyro=2.50 → Predicted: **running** ❌
- Device is actually still but predicted as running every time

## Root Cause

The model was trained on **normalized features** (StandardScaler applied in GUI app), but the StandardScaler saved with the model has **incorrect parameters**:

**Wrong scaler (from model file):**

```cpp
mean = [-0.012, -0.003, ...] // Near zero
std  = [0.986, 0.998, ...]   // Near one
```

**Why this happened:**

1. GUI app extracts raw features from windows (acc_mag_mean ≈ 10-12 m/s²)
2. GUI app applies StandardScaler normalization (acc_mag_mean → -0.9 to +2.8)
3. Normalized features saved to training CSV
4. Model training loads already-normalized CSV
5. Model applies **another** StandardScaler on already-normalized data
6. This second scaler learns useless parameters (mean≈0, std≈1)
7. Deployment code uses the useless scaler → wrong predictions!

## Value Comparison

### Raw Device Features (Still activity)

```
acc_mag_mean  = 9.52 m/s² (gravity)
gyro_mag_mean = 2.50 deg/s (almost stationary)
```

### After WRONG Scaler

```
acc_mag_mean  = (9.52 - (-0.012)) / 0.986 = 9.67  (still huge!)
gyro_mag_mean = (2.50 - (-0.010)) / 0.989 = 2.54  (still huge!)
```

→ Model sees values way outside training range → predicts randomly

### After CORRECT Scaler  

```
acc_mag_mean  = (9.52 - 11.736) / 2.506 = -0.88  ✅
gyro_mag_mean = (2.50 - 106.336) / 72.335 = -1.44 ✅
```

→ Model sees expected values for "still" activity → correct prediction!

## Correct Scaler Parameters

Computed from 187 raw training windows:

```cpp
const float feature_means[NUM_FEATURES] = {
    11.736, 3.799, 12.472, 24803.748, 106.336, 70.696, 128.197, 3483443.108,
    0.981, 0.853, 1.307, 399.312, 4.880, 22.244, 18.935, 274.843,
    3.737
};

const float feature_stds[NUM_FEATURES] = {
    2.506, 2.608, 3.114, 13417.818, 72.335, 40.986, 82.360, 3413654.774,
    0.794, 0.583, 0.977, 488.277, 2.515, 9.897, 17.201, 160.834,
    2.591
};
```

## Files Fixed

### Deployment Code

**File:** `persistent_data_new/generated/neural_network_models/har_neural_network_seeed_xiao_f17_c5_balanced/har_neural_network_seeed_xiao_f17_c5_balanced.cpp`

**Lines 21-35:** Updated `feature_means` and `feature_stds` arrays with correct parameters computed from raw window data.

## Expected Results

### Still Device (acc=9.52, gyro=2.50)

- **Before fix:** Predicted running ❌
- **After fix:** Should predict still ✅

### Walking Device (acc≈12, gyro≈150)

- **Before fix:** Predicted running ❌  
- **After fix:** Should predict walking ✅

### Running Device (acc≈15, gyro≈200)

- **Before fix:** Predicted running (accidentally correct)
- **After fix:** Should predict running ✅

## Verification

Run `compute_correct_scaler.py` to verify the scaler parameters:

```bash
python compute_correct_scaler.py
```

Expected output shows the still device (acc=9.52, gyro=2.50) scales to (-0.88, -1.44), matching the training data's "still" activity range.

## Next Steps

1. Upload the corrected `.cpp` file to XIAO device
2. Test with device on table → Should predict "still" (Class 1)  
3. Test walking → Should predict "walking" (Class 2)
4. Test running → Should predict "running" (Class 0)

## Long-term Fix

The GUI app should:

1. Save the PRE-normalization scaler parameters (from raw features)
2. Store these in the model metadata
3. Code generator should use these parameters instead of the post-normalization scaler

Currently there's a double-normalization bug in the training pipeline.
