# Model-to-Deployment Accuracy Mismatch: Root Cause Analysis

**Date**: January 13, 2026  
**Issue**: High training accuracy (90%+) but poor device testing accuracy  
**Severity**: CRITICAL - Breaks the entire deployment pipeline

---

## 🔴 CRITICAL ISSUE IDENTIFIED

### **Root Cause: Feature Extraction Mismatch Between Training and Deployment**

The framework has a **fundamental discrepancy** between:
1. **Python feature extraction** (used during training)
2. **C++ feature extraction** (deployed on device)

---

## 📊 The Problem

### Training Features (Python)
Located in `utils/model_training.py`:

**Time-Domain Features (16 per axis × 6 axes = 96 features)**:
```python
def extract_time_domain_features(df, sensor_cols):
    features = {}
    for col in sensor_cols:
        features[f'{col}_mean'] = np.mean(data)
        features[f'{col}_std'] = np.std(data)
        features[f'{col}_min'] = np.min(data)
        features[f'{col}_max'] = np.max(data)
        features[f'{col}_range'] = np.max(data) - np.min(data)
        features[f'{col}_median'] = np.median(data)
        features[f'{col}_q25'] = np.percentile(data, 25)
        features[f'{col}_q75'] = np.percentile(data, 75)
        features[f'{col}_iqr'] = np.percentile(data, 75) - np.percentile(data, 25)
        features[f'{col}_skewness'] = pd.Series(data).skew()
        features[f'{col}_kurtosis'] = pd.Series(data).kurtosis()
        features[f'{col}_rms'] = np.sqrt(np.mean(data**2))
        features[f'{col}_energy'] = np.sum(data**2)
        features[f'{col}_zero_crossings'] = len(np.where(np.diff(np.sign(data)))[0])
        features[f'{col}_mean_crossing_rate'] = len(np.where(np.diff(np.sign(data - np.mean(data))))[0])
```

**Frequency-Domain Features (8 per axis × 6 axes = 48 features)**:
```python
def extract_frequency_domain_features(df, sensor_cols, sampling_rate=100):
    # FFT-based features
    features[f'{col}_spectral_centroid']
    features[f'{col}_spectral_rolloff']
    features[f'{col}_spectral_bandwidth']
    features[f'{col}_dominant_frequency']
    features[f'{col}_dominant_frequency_magnitude']
    features[f'{col}_energy_low_freq']
    features[f'{col}_energy_mid_freq']
    features[f'{col}_energy_high_freq']
```

**Total Training Features**: **15 time + 8 freq = 23 features per axis × 6 axes = 138 features**

---

### Deployment Features (C++)
Located in `deployment/base_generator.py`, "balanced" optimization:

**Time-Domain Features ONLY (15 per axis × 6 axes = 90 features)**:
```cpp
void extract_features(float sensor_data[][6], int samples, float features[]) {
    // Only 15 features per axis:
    // mean, std, min, max, range, median, q25, q75, iqr
    // skewness, kurtosis, rms, energy
    // zero_crossings, mean_crossing_rate
    
    // Comment says: "No frequency-domain features (model retrained without FFT features)"
}
```

**Total Deployment Features**: **90 features (NO FREQUENCY FEATURES!)**

---

## 🚨 THE MISMATCH

| Component | Features | Frequency Domain |
|-----------|----------|------------------|
| **Training (Python)** | 138 | ✅ YES (48 FFT features) |
| **Deployment (C++)** | 90 | ❌ NO (commented out) |
| **Difference** | **-48 features** | **MISSING** |

### Why This Breaks Everything:

1. **Model trained with 138 features**
2. **Device provides only 90 features**
3. **Model expects specific features at specific indices**
4. **Feature mismatch = garbage predictions**

Example:
- Training: Feature #90 = `aX_spectral_centroid` (frequency feature)
- Device: Feature #90 = `gZ_rms` (different feature!)
- Result: Model completely confused

---

## 🔍 Evidence in Code

### In `base_generator.py` line 638:
```cpp
// Total features: 15 per axis * 6 axes = 90 time-domain features
// No frequency-domain features (model retrained without FFT features)
```

**This comment is a LIE!** The model was NOT retrained without FFT features. The training code in `model_training.py` still includes frequency features by default:

### In `model_training.py` line 587:
```python
def create_feature_vector(df, sensor_cols=None, sampling_rate=100, 
                          include_frequency=True):  # ← DEFAULT IS TRUE!
    time_features = extract_time_domain_features(df, sensor_cols)
    
    if include_frequency:  # This runs by default
        freq_features = extract_frequency_domain_features(df, sensor_cols, sampling_rate)
        combined_features = pd.concat([time_features, freq_features], axis=1)
    else:
        combined_features = time_features
```

---

## 💥 Impact on Each Model Type

### 1. Random Forest
**Impact**: Severe
- Trees trained on 138 features
- Deployment provides 90 features
- Missing features treated as 0.0 or use wrong data
- Result: Trees make decisions based on wrong features

### 2. SVM
**Impact**: CRITICAL
- RBF kernel computes distance in 138-dimensional space
- Deployment provides 90D vector
- Distance calculation completely wrong
- Support vectors comparison meaningless

### 3. Neural Network
**Impact**: CATASTROPHIC
- Input layer expects 138 neurons
- Deployment provides 90 values
- Remaining 48 inputs get wrong data or zeros
- All weights misaligned
- Output is random garbage

---

## 🛠️ ROOT CAUSES

### 1. **No Frequency Features in C++ Code**
The C++ generator doesn't implement FFT because:
- FFT is computationally expensive on microcontrollers
- Arduino doesn't have efficient FFT library by default
- Comment says "model retrained without FFT" but this never happened

### 2. **Training Code Includes Frequency by Default**
Users unknowingly train models with frequency features because:
- `include_frequency=True` is the default parameter
- No warning in UI about deployment mismatch
- No validation that checks feature count consistency

### 3. **No Validation Between Training and Deployment**
The code generation doesn't check:
- Number of features matches training
- Feature names match between Python and C++
- Feature order is consistent

### 4. **Overlap/Windowing Issues**
Additional smaller issue: Generated code uses overlapping windows (50% default) but training uses non-overlapping windows from Tab 2.

---

## ✅ SOLUTIONS

### **Option A: Train Without Frequency Features** (QUICK FIX)
**Pros**: Works with current C++ code  
**Cons**: Lower accuracy, wastes 8 useful features per axis

**Implementation**:
1. Modify Tab 4 (Training) to add checkbox: "Include Frequency Features"
2. Default to `include_frequency=False`
3. Add warning: "Frequency features disabled for microcontroller deployment"
4. Update `create_feature_vector` calls to pass `include_frequency=False`

**Changes needed**:
- `layouts/training.py`: Add checkbox
- `callbacks/training_callbacks.py`: Pass parameter
- `utils/model_training.py`: Already supports this parameter

---

### **Option B: Add Frequency Features to C++ Code** (BETTER SOLUTION)
**Pros**: Full accuracy, uses all features  
**Cons**: More computation, needs FFT library

**Implementation**:
1. Add lightweight FFT library (arduinoFFT or arm_math CMSIS-DSP)
2. Implement frequency feature extraction in C++
3. Match Python FFT behavior exactly

**Challenges**:
- FFT increases inference time by 2-3x
- Not all platforms have FFT support
- Must match NumPy FFT behavior precisely

---

### **Option C: Feature Set Selection During Code Generation** (BEST SOLUTION)
**Pros**: Flexible, user chooses accuracy vs. speed  
**Cons**: More complex implementation

**Implementation**:
1. Store feature configuration in model metadata
2. During training, record which features were used
3. During code generation, generate ONLY those features
4. Validate that model expects same features as code generates

**Architecture**:
```python
# In model metadata (model_info.json):
{
    "features_config": {
        "include_frequency": False,  # or True
        "feature_count": 90,  # or 138
        "feature_names": [...],
        "sampling_rate": 100
    }
}

# Code generator checks:
if model_info['feature_count'] != len(generated_features):
    raise ValueError("Feature count mismatch!")
```

---

## 🎯 RECOMMENDED IMMEDIATE ACTION

### **Phase 1: Quick Fix (1-2 hours)**

1. **Disable frequency features in training by default**:
   - Change `include_frequency=True` to `include_frequency=False` in training callbacks
   - Add UI notice: "Training without frequency features (optimized for microcontroller)"

2. **Retrain all existing models**:
   - Models must be retrained with 90 features (no frequency)
   - Update model metadata to indicate feature count

3. **Add validation**:
   - Check feature count matches during code generation
   - Throw error if mismatch detected

### **Phase 2: Proper Fix (2-4 hours)**

4. **Implement lightweight FFT in C++**:
   - Use arduinoFFT library (MIT license)
   - Add conditional compilation based on feature config
   - Implement 8 frequency features to match Python

5. **Add feature configuration UI**:
   - Tab 4: Checkbox "Include Frequency Features (slower inference)"
   - Show estimated inference time: 20ms (no FFT) vs 60ms (with FFT)
   - Generate code matching training configuration

6. **Comprehensive validation**:
   - Feature count check
   - Feature name check
   - Sampling rate consistency
   - Window size consistency

---

## 📝 Testing Plan

### Test 1: Feature Extraction Consistency
**Goal**: Verify Python and C++ extract identical features

**Method**:
```python
# Python side
python_features = create_feature_vector(window_df, include_frequency=False)

# Device side (print features via Serial)
# Compare feature vectors element-by-element
# Tolerance: ±0.01 for floating-point differences
```

**Expected**: All 90 features match within tolerance

---

### Test 2: End-to-End Prediction
**Goal**: Same input data produces same prediction

**Method**:
1. Record 10 test windows from device
2. Run through Python model → get predictions
3. Upload to device, run through C++ model → get predictions
4. Compare results

**Expected**: 100% prediction agreement

---

### Test 3: Known Activity Classification
**Goal**: Device correctly classifies known activities

**Method**:
1. Perform "walking" activity
2. Device should predict "walking" with high confidence
3. Repeat for all trained activities

**Expected**: >90% accuracy on device

---

## 💡 ADDITIONAL IMPROVEMENTS

### 1. **Feature Visualization Tool**
Create debug tool that:
- Displays Python features vs C++ features side-by-side
- Highlights discrepancies
- Shows feature importance from model

### 2. **Model Deployment Validator**
Before code generation:
```python
def validate_deployment_feasibility(model_info, platform):
    checks = {
        'feature_count': check_feature_count_consistency(),
        'memory_usage': check_memory_fits_platform(),
        'inference_time': estimate_inference_time(),
        'battery_life': estimate_battery_impact()
    }
    return validation_report
```

### 3. **Benchmark Suite**
Include known test cases:
- UCI HAR dataset predictions
- Synthetic data with known labels
- Edge cases (all zeros, saturated sensors)

---

## 🏆 SELLING POINTS (After Fixes)

### Current (Broken):
- ❌ Models work in training but fail on device
- ❌ No way to debug feature mismatch
- ❌ Silent failures with incorrect predictions

### After Fix:
- ✅ **Guaranteed Feature Consistency**: Python ↔ C++ features verified
- ✅ **Deployment Validation**: Pre-flight checks before code generation
- ✅ **Flexible Feature Sets**: Choose accuracy vs. speed
- ✅ **Debug Tools**: Compare Python vs device features in real-time
- ✅ **Confidence Metrics**: Show when predictions are uncertain
- ✅ **Automated Testing**: Built-in test suite validates deployments
- ✅ **Platform Optimization**: Automatically optimize features for target hardware

### Competitive Advantages:
| Feature | This Framework (Fixed) | Edge Impulse | SensiML | TFLite Micro |
|---------|------------------------|--------------|---------|--------------|
| Feature Validation | ✅ Automatic | ⚠️ Manual | ⚠️ Manual | ❌ None |
| Python-C++ Parity | ✅ Verified | ⚠️ Assumed | ⚠️ Assumed | ❌ No Python |
| Debug Tools | ✅ Built-in | ⚠️ Limited | ⚠️ Limited | ❌ None |
| Flexible Features | ✅ Configurable | ❌ Fixed | ⚠️ Limited | ❌ Fixed |
| Open Source | ✅ Full | ❌ Proprietary | ❌ Proprietary | ✅ Yes |

---

## 📋 Implementation Checklist

### Critical Path (Must Fix):
- [ ] Disable frequency features by default (`include_frequency=False`)
- [ ] Add feature count validation in code generator
- [ ] Test Python vs C++ feature extraction on same data
- [ ] Retrain at least one model to verify fix
- [ ] Document feature configuration in user guide

### Important (Should Fix):
- [ ] Add frequency features toggle in Tab 4 UI
- [ ] Implement lightweight FFT in C++ (arduinoFFT)
- [ ] Add feature comparison debug tool
- [ ] Create deployment validation function
- [ ] Add estimated inference time calculator

### Nice to Have (Future):
- [ ] Feature importance visualization
- [ ] Automatic platform optimization
- [ ] Benchmark suite with known datasets
- [ ] Confidence intervals on predictions
- [ ] Model versioning and compatibility checks

---

## 🔬 Technical Details

### Why NumPy FFT != Arduino FFT

**NumPy FFT**:
- Uses FFTW library (highly optimized)
- Returns complex numbers
- Frequency bins: `np.fft.fftfreq(n, 1/fs)`
- Positive frequencies only: `fft_freq > 0`

**Arduino FFT** (arduinoFFT library):
- Simplified radix-2 FFT
- May return slightly different magnitudes
- Frequency calculation must match exactly
- Needs power-of-2 window sizes for best performance

### Matching FFT Behavior

**Python**:
```python
fft_vals = np.fft.fft(data)
fft_magnitude = np.abs(fft_vals)
fft_freq = np.fft.fftfreq(len(data), 1/sampling_rate)
pos_mask = fft_freq > 0
fft_magnitude_pos = fft_magnitude[pos_mask]
```

**C++ (must match)**:
```cpp
#include <arduinoFFT.h>

double vReal[WINDOW_SIZE];
double vImag[WINDOW_SIZE];

// Copy sensor data
for (int i = 0; i < WINDOW_SIZE; i++) {
    vReal[i] = sensor_data[i][axis];
    vImag[i] = 0.0;
}

// Compute FFT
FFT.Windowing(vReal, WINDOW_SIZE, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
FFT.Compute(vReal, vImag, WINDOW_SIZE, FFT_FORWARD);
FFT.ComplexToMagnitude(vReal, vImag, WINDOW_SIZE);

// Now vReal contains magnitudes, use first half (positive frequencies)
```

---

## 📞 Summary

**The framework has a critical bug**: Training uses 138 features but deployment generates only 90. This causes completely incorrect predictions on actual devices despite high training accuracy.

**Quick fix**: Disable frequency features during training (reduce to 90 features).

**Proper fix**: Implement FFT in C++ code to match all 138 training features.

**Best fix**: Add feature configuration system with validation to prevent mismatches.

**After fixes, new selling points**:
- Feature parity validation (Python ↔ C++)
- Deployment pre-flight checks
- Debug tools for feature comparison
- Configurable feature sets
- Guaranteed accuracy preservation

This will make the framework **production-ready** and **more reliable than commercial alternatives**.
