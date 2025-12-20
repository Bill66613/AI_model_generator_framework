# Sliding Window Generation Implementation Summary

## Features Added

### 1. **UI Components** (layouts/preprocessing.py)
Added new section after window management with:
- **Overlap Percentage Slider** (0-90%, default 50%)
  - Shows stride calculation and sample increase factor
  - Real-time preview of "10 windows → X windows"
  
- **Quality Threshold Slider** (0.3-1.0, default 0.7)
  - Higher values = stricter filtering
  - Removes low-quality windows automatically

- **Generate Button** - Creates overlapping windows from selected regions
- **Save Button** - Saves approved windows to disk (disabled until generation)
- **Preview Area** - Shows statistics and quality distribution

### 2. **Quality Metrics Function** (preprocessing_callbacks.py)
```python
compute_window_quality(window_data)
```
Checks:
- ✅ **Variance**: Detects stationary/flat signals (score -0.3)
- ✅ **Outliers**: Flags extreme sensor values (score -0.2)
- ✅ **Missing Data**: Detects NaN values (score -0.5)
- ✅ **Data Length**: Ensures sufficient samples (score -0.3)

Returns: quality_score (0-1) and list of issues

### 3. **Sliding Window Generator**
```python
generate_sliding_windows_from_current(current_windows, df, ...)
```
- Takes your **manually-selected windows** as input
- Applies sliding window with specified overlap
- Filters by quality threshold
- Returns: good_windows, flagged_windows, statistics

### 4. **Workflow Integration**

#### **Step 1: Manual Selection** (existing)
```
1. Load dataset
2. Clean & smooth data
3. Apply time windows
4. Drag windows to select good regions
```

#### **Step 2: Automatic Generation** (NEW)
```
5. Set overlap (e.g., 50%)
6. Set quality threshold (e.g., 0.7)
7. Click "Generate Sliding Windows"
8. Review preview:
   - High-quality count
   - Flagged count
   - Quality distribution chart
   - Sample increase factor
```

#### **Step 3: Save** (NEW)
```
9. Click "Save Generated Windows"
10. Windows saved as: sliding_0, sliding_1, ... sliding_N
11. Available in Training Data Preparation
```

---

## Example Usage

### Scenario: You have 10 manually-selected windows

**Without Overlap (Current):**
- Result: 10 windows for training
- Problem: Insufficient samples

**With 50% Overlap (NEW):**
- Stride = 75 samples (half of 150)
- Result: ~20 windows from same data
- **2x improvement!**

**With 75% Overlap (NEW):**
- Stride = 37 samples (quarter of 150)
- Result: ~40 windows from same data
- **4x improvement!**

---

## Quality Filtering Example

### Generated 45 Windows Total:

**Quality Distribution:**
```
Score 0.9-1.0: ████████████ 30 windows → SAVED ✅
Score 0.7-0.9: ████ 10 windows → SAVED ✅
Score 0.5-0.7: ██ 3 windows → FLAGGED ⚠️
Score 0.3-0.5: █ 2 windows → FLAGGED ⚠️
```

**With threshold = 0.7:**
- High quality: 40 windows (saved)
- Flagged: 5 windows (excluded)
- You can lower threshold to include them

---

## Benefits

### 1. **More Training Data**
- 2-4x more samples from same recording
- Better model generalization
- Reduced overfitting

### 2. **Quality Control**
- Automatic filtering of bad segments
- Catches breaks, disturbances, sensor issues
- Only high-quality data used for training

### 3. **Research Standard**
- 50% overlap is standard in HAR literature
- Used in UCI-HAR, WISDM, MobiAct datasets
- Academically sound approach

### 4. **Maintains Manual Control**
- You still select activity regions manually
- Sliding windows only within YOUR selections
- Best of both worlds: automation + quality control

---

## Configuration Recommendations

### For Initial Testing:
```
Overlap: 50%
Quality Threshold: 0.7
```
- Standard research settings
- Good balance of quantity vs quality

### For Maximum Data:
```
Overlap: 75%
Quality Threshold: 0.5
```
- 4x sample increase
- More lenient quality (includes borderline windows)
- Use if you have very limited data

### For High Precision:
```
Overlap: 50%
Quality Threshold: 0.9
```
- Strict quality filtering
- Only pristine windows
- Use for final deployment models

---

## Quality Issue Examples

### Low Variance (Score -0.3)
```
Issue: "Low variance (possibly stationary)"
Cause: Phone left on table, no movement
Action: Automatically excluded
```

### Extreme Outliers (Score -0.2)
```
Issue: "Extreme outliers in aX"
Cause: Dropped phone, sensor spike
Action: Automatically excluded
```

### Missing Data (Score -0.5)
```
Issue: "Missing data"
Cause: Sensor disconnection, Bluetooth dropout
Action: Automatically excluded
```

### Insufficient Length (Score -0.3)
```
Issue: "Insufficient data points"
Cause: Window at edge of recording
Action: Automatically excluded
```

---

## Integration with Existing Features

### Compatible with:
- ✅ All preprocessing steps (clean, smooth, filter)
- ✅ Manual window selection
- ✅ Feature extraction (time-domain, frequency-domain)
- ✅ All model types (RF, SVM, NN)
- ✅ Train/Val/Test splitting

### Workflow:
```
1. Upload Data
2. Clean & Smooth
3. Manual Window Selection (10 windows)
4. Generate Sliding Windows (→ 40 windows)  ← NEW
5. Save Windows                             ← NEW
6. Feature Extraction
7. Train/Val/Test Split
8. Train Model
```

---

## Testing the Feature

### 1. Load a dataset
```
Tabs → Preprocessing → Select dataset
```

### 2. Define manual windows
```
Apply time window (1500ms)
Drag 3-5 windows to select good regions
```

### 3. Generate sliding windows
```
Set overlap: 50%
Set quality: 0.7
Click "Generate Sliding Windows"
```

### 4. Review results
```
Check statistics:
- How many windows generated?
- Quality distribution chart
- Sample increase factor
```

### 5. Save and use
```
Click "Save Generated Windows"
Go to Training Data Preparation
See your new windows listed!
```

---

## Expected Results

### Before (10 manual windows):
```
Training: ~6 samples
Validation: ~2 samples
Test: ~2 samples
Result: Severe overfitting
```

### After (40 sliding windows, 50% overlap):
```
Training: ~24 samples
Validation: ~8 samples
Test: ~8 samples
Result: Much better generalization
```

### After (80 sliding windows, 75% overlap):
```
Training: ~48 samples
Validation: ~16 samples
Test: ~16 samples
Result: Decent training dataset!
```

---

## Next Steps

1. **Test with your current data**
   - See how many windows you can generate
   - Check quality distribution
   - Adjust thresholds as needed

2. **Collect more continuous recordings**
   - 30-60 seconds per activity
   - Minimize breaks/disturbances
   - With sliding windows: 30s → 40+ windows

3. **Train models with new data**
   - Should see immediate improvement
   - Compare: old (10 samples) vs new (40+ samples)
   - Monitor test accuracy increase

4. **Iterate on quality settings**
   - Too many flagged? Lower threshold
   - Getting bad data? Raise threshold
   - Find sweet spot for your sensors

---

## Technical Details

### File Naming:
```
persistent_data/sliding_0_running_2025.csv
persistent_data/sliding_1_running_2025.csv
...
persistent_data/sliding_39_running_2025.csv
```

### Metadata Updated:
```json
{
  "running_2025.csv": {
    "dragged_samples": [
      "dragged_window_0_running_2025",  // manual
      "sliding_0_running_2025",         // auto
      "sliding_1_running_2025",         // auto
      ...
    ]
  }
}
```

### Memory Efficient:
- Windows generated on-the-fly
- Only saved when approved
- Original data unchanged

---

## Troubleshooting

### "No windows generated"
→ Check: Do you have manual windows selected?
→ Solution: Define at least 1-2 manual windows first

### "All windows flagged"
→ Check: Quality threshold too high?
→ Solution: Lower threshold to 0.5 or 0.3

### "Quality scores all low"
→ Check: Is data noisy/bad?
→ Solution: Clean & smooth first, or collect better data

### "Not enough improvement"
→ Check: Overlap percentage
→ Solution: Increase to 75% for 4x multiplication

---

## Research Citations

This implementation follows standards from:
- **UCI-HAR Dataset**: 50% overlap, 2.56s windows
- **WISDM**: 50% overlap, 10s windows
- **MobiAct**: 66% overlap, 1-2s windows

Standard practice in Human Activity Recognition research!
