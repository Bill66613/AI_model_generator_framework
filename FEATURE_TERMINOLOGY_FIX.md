# Feature Engineering Terminology Fix

## Problem Statement

The previous UI displayed misleading feature counts that confused raw sensor axes with extracted features:
- Showed "Features: 6" when user selected "Statistical Features" 
- This referred to the 6 raw sensor axes (aX, aY, aZ, gX, gY, gZ)
- Did NOT show the actual number of features extracted during feature engineering

## Solution Implemented

### 1. **Updated UI Labels to Show Actual Feature Counts**

**Feature Selection Dropdown** (`layouts/preprocessing.py`):
- ✅ Before: "🎯 All Features"
- ✅ After: "🎯 All Features (138: Time + Frequency)"
- ✅ Before: "📈 Statistical Features"  
- ✅ After: "📈 Raw Axes Only (6 sensors)"
- ✅ Before: "🌊 Time-Domain Only"
- ✅ After: "🌊 Time-Domain Only (90 features)"

### 2. **Clarified Terminology Throughout UI**

**Preprocessing Results** (`callbacks/preprocessing_callbacks.py`):
- Changed "🎯 Features:" to "🎯 Features Extracted:"
- Now shows: "90 features (Time-Domain Only)" instead of "6 (statistical)"

**Train-Test Split Visualization**:
- Changed annotation from "📊 Features:" to "📊 Features Extracted:"
- Shows actual extracted feature count, not raw column count

**Training Results** (`callbacks/training_callbacks.py`):
- Changed "Features:" to "Features Extracted:"
- Clarifies this is the engineered feature count used for training

### 3. **Added Helper Function**

Created `_get_feature_method_label()` to map feature selection codes to descriptive labels:

```python
def _get_feature_method_label(feature_method):
    """Convert feature method code to descriptive label."""
    labels = {
        'all': 'Time + Frequency Domain',
        'statistical': 'Raw Sensor Axes Only',
        'time_domain': 'Time-Domain Only',
        'custom': 'Custom Selection'
    }
    return labels.get(feature_method, feature_method)
```

### 4. **Updated Documentation**

Enhanced `docs/PREPROCESSING.md` with accurate feature counts:

**🎯 All Features (138 features)**
- 90 time-domain features (15 per axis × 6 axes)
- 48 frequency-domain features (8 per axis × 6 axes)
- Best for complex activities, Python-only deployment

**📈 Raw Axes Only (6 features)**
- Just raw sensor readings: aX, aY, aZ, gX, gY, gZ
- Minimal computational cost but poor performance
- For testing/debugging only

**🌊 Time-Domain Only (90 features)**
- 15 statistical features per axis × 6 axes
- Features: mean, std, min, max, range, median, q25, q75, iqr, skewness, kurtosis, rms, energy, zero_crossings, mean_crossing_rate
- **Recommended for production** - Arduino-compatible, excellent performance

## Feature Count Breakdown

Based on actual training data analysis:

| Selection Method | Feature Count | Description |
|-----------------|---------------|-------------|
| **All Features** | 138 | 90 time-domain + 48 frequency-domain (FFT) |
| **Time-Domain Only** | 90 | 15 statistical features × 6 axes |
| **Raw Axes Only** | 6 | aX, aY, aZ, gX, gY, gZ (no feature engineering) |

### Time-Domain Features (15 per axis)
1. mean - Average value
2. std - Standard deviation
3. min - Minimum value
4. max - Maximum value
5. range - max - min
6. median - 50th percentile
7. q25 - 25th percentile
8. q75 - 75th percentile
9. iqr - Interquartile range (q75 - q25)
10. skewness - Distribution asymmetry
11. kurtosis - Distribution tailedness
12. rms - Root mean square
13. energy - Sum of squared values
14. zero_crossings - Sign change count
15. mean_crossing_rate - Mean crossing count

### Frequency-Domain Features (8 per axis)
1. fft_mean - Average FFT magnitude
2. fft_std - FFT magnitude std deviation
3. fft_max - Peak FFT magnitude
4. fft_energy - Total spectral energy
5. fft_entropy - Spectral entropy
6. dominant_frequency - Frequency with max power
7. spectral_centroid - Center of spectral mass
8. spectral_rolloff - 85% energy frequency

## Files Modified

1. **callbacks/preprocessing_callbacks.py**
   - Added `_get_feature_method_label()` helper function
   - Updated preprocessing results display
   - Updated train-test split visualization annotation

2. **callbacks/training_callbacks.py**
   - Changed "Features:" to "Features Extracted:"

3. **layouts/preprocessing.py**
   - Updated Feature Selection dropdown labels with actual counts

4. **docs/PREPROCESSING.md**
   - Enhanced documentation with detailed feature breakdowns
   - Added feature engineering methodology

## Testing Recommendations

1. **UI Verification**
   - Navigate to Preprocessing tab → "Prepare for Training"
   - Select different feature methods from dropdown
   - Verify labels show correct counts: (138), (6), or (90)
   - Click "Preprocess for Training"
   - Verify results show "Features Extracted: X features (Description)"

2. **Training Verification**
   - Train a model using each feature selection method
   - Verify training results show "Features Extracted: X"
   - Check model metadata for correct feature count

3. **Visual Verification**
   - Check train-test split graph annotation
   - Should show "📊 Features Extracted: X" (not just "Features:")

## Benefits

✅ **Clarity**: Users now understand the difference between raw axes and extracted features  
✅ **Transparency**: Actual feature counts displayed before training  
✅ **Accuracy**: Terminology matches machine learning conventions  
✅ **Education**: Users learn about feature engineering process  
✅ **Decision-making**: Clear counts help users choose appropriate feature selection

## Before vs After Examples

### Scenario 1: Time-Domain Feature Selection

**Before:**
- Dropdown: "🌊 Time-Domain Only"
- Result: "🎯 Features: 6 (time_domain)"
- Confusing! Shows 6 but extracts 90

**After:**
- Dropdown: "🌊 Time-Domain Only (90 features)"
- Result: "🎯 Features Extracted: 90 features (Time-Domain Only)"
- Clear! Shows exactly what's happening

### Scenario 2: Raw Axes Selection

**Before:**
- Dropdown: "📈 Statistical Features"
- Result: "🎯 Features: 6 (statistical)"
- Misleading! "Statistical features" but only raw axes

**After:**
- Dropdown: "📈 Raw Axes Only (6 sensors)"
- Result: "🎯 Features Extracted: 6 features (Raw Sensor Axes Only)"
- Accurate! No confusion about feature engineering

## Impact on Existing Models

⚠️ **No breaking changes** - This is purely a UI/documentation update. Existing trained models remain compatible.

The actual feature extraction logic in `preprocess_for_training()` callback was not changed - only the display labels were updated for clarity.

## Next Steps

Consider these future enhancements:
1. Add tooltip/info icon explaining feature engineering process
2. Show preview of which exact features will be extracted
3. Add feature importance visualization after training
4. Export feature names with model metadata for deployment

---

**Status**: ✅ COMPLETED  
**Date**: 2025-12-12  
**Todo Item**: Fixed Feature Engineering terminology (#2)
