# Quick Reference: New 5-Tab Framework

## Tab Overview

| Tab # | Name | Purpose | Status |
|-------|------|---------|--------|
| 1 | 📊 Data Management | Upload CSVs, assign labels | ✅ Working |
| 2 | 🔬 Signal Preprocessing | Clean data, create windows | ✅ Working |
| 3 | ⚙️ Feature Engineering | Extract features uniformly | ⚠️ UI ready, callbacks pending |
| 4 | 🎯 Model Training | Train ML models | ⚠️ Needs update |
| 5 | 📡 Device Testing | Real-time UART testing | 📝 Placeholder |

## Feature Engineering Tab (NEW)

### Why This Tab Exists
**Problem**: Processing each activity label separately risked inconsistent settings:
- Different normalization methods
- Different feature selections
- Different train/val/test ratios

**Solution**: Process ALL labels together with unified settings:
- ✅ Same features extracted for all
- ✅ Same normalization for all
- ✅ Same split ratios for all
- ✅ Single execution prevents errors

### How to Use

1. **Select Labels** (Step 1)
   - Choose one or more activity labels
   - Click "Select All" for all labels
   - View window counts per label

2. **Configure Features** (Step 2)
   - Choose feature extraction method:
     - **All Features**: 138 (time + frequency)
     - **Time-Domain Only**: 90 features
     - **Frequency-Domain Only**: 48 features
     - **Raw Sensors**: 6 axes only
   - Choose normalization:
     - **Standard**: Z-score scaling (recommended)
     - **MinMax**: 0-1 scaling
     - **Robust**: Resistant to outliers
     - **None**: No normalization

3. **Configure Split** (Step 3)
   - Set training ratio (50-90%, default 70%)
   - Set validation ratio (5-30%, default 15%)
   - Test ratio calculated automatically
   - Set random state for reproducibility

4. **Execute** (Step 4)
   - Click "⚙️ Engineer Features for All Selected Datasets"
   - All labels processed with same settings
   - Results displayed in table
   - Data ready for training

## Key Component IDs

### Feature Engineering Tab
```python
'activity-labels-selector'          # Multi-select dropdown
'select-all-labels-btn'             # Select all button
'clear-labels-btn'                  # Clear selection button
'windows-per-label-display'         # Shows counts

'global-feature-selection'          # Feature method dropdown
'global-normalization-method'       # Normalization dropdown
'feature-count-display'             # Shows total features

'global-train-split'                # Train ratio slider
'global-val-split'                  # Val ratio slider
'global-test-split-display'         # Calculated test %
'global-random-state'               # Random seed input

'execute-feature-engineering-btn'   # Execute button
'feature-engineering-results'       # Results display
'engineered-dataset-stats'          # Statistics table

'engineered-dataset-store'          # Data store (dcc.Store)
'selected-labels-store'             # Selection store
```

## Data Stores

| Store ID | Purpose | Type | Location |
|----------|---------|------|----------|
| `stored-datasets` | Uploaded datasets | local | data_upload |
| `current-windows` | Manual/sliding windows | session | preprocessing |
| `engineered-dataset-store` | Feature-engineered data | session | feature_engineering |
| `selected-labels-store` | Selected labels | session | feature_engineering |

## Callback Implementation Guide

### Critical Callback: execute_feature_engineering()

**Inputs**:
- `n_clicks`: Execute button
- `selected_labels`: List of activity labels
- `feature_method`: all/time_domain/frequency_domain/raw
- `normalization_method`: standard/minmax/robust/none
- `train_ratio`: Float (0.5-0.9)
- `val_ratio`: Float (0.0-0.3)
- `random_state`: Int

**Outputs**:
- `feature-engineering-results`: HTML message
- `engineered-dataset-stats`: Statistics table
- `engineered-dataset-store`: Engineered dataset dict

**Implementation Steps**:
```python
1. Load all windows from all selected labels
   - For each label: read persistent_data/<label>/*.csv
   - Track label for each window

2. Extract features uniformly
   - Use SAME feature extraction settings for ALL
   - Import from utils/feature_extraction.py
   - Result: DataFrame with features + 'activity' column

3. Normalize uniformly
   - Fit SINGLE scaler on ALL data
   - Transform ALL data with same scaler
   - Save scaler for later use

4. Combine and split
   - Combine all labels into one DataFrame
   - Shuffle with random_state
   - Split: train/val/test using ratios
   - Stratify by activity label

5. Save results
   - Store in engineered-dataset-store as dict:
     {
       'train': {'X': array, 'y': array},
       'val': {'X': array, 'y': array},
       'test': {'X': array, 'y': array},
       'scaler': scaler_object,
       'feature_names': list,
       'label_mapping': dict
     }

6. Display statistics
   - Table showing:
     - Activity | Total Windows | Train | Val | Test
     - One row per label
     - Summary row at bottom
```

## File Structure

```
GUI_app/
├── app.py                          # Main app (5 tabs)
├── layouts/
│   ├── data_upload.py             # Tab 1: Data Management
│   ├── preprocessing.py           # Tab 2: Signal Preprocessing
│   ├── feature_engineering.py     # Tab 3: Feature Engineering ✨ NEW
│   ├── training.py                # Tab 4: Model Training
│   └── device_test.py             # Tab 5: Device Testing ✨ NEW
├── callbacks/
│   ├── data_callbacks.py
│   ├── preprocessing_callbacks.py
│   ├── feature_engineering_callbacks.py  ✨ NEW (to implement)
│   └── training_callbacks.py
└── utils/
    ├── feature_extraction.py      # Feature extraction functions
    └── preprocessing.py           # Signal processing functions
```

## Workflow

### Complete Pipeline
1. **Data Management** → Upload CSV, assign activity labels
2. **Signal Preprocessing** → Clean data, create time windows
3. **Feature Engineering** → Extract features with consistent settings
4. **Model Training** → Train models on engineered dataset
5. **Device Testing** → Test on real device (future)

### Feature Engineering Workflow
```
Select Labels (all at once)
    ↓
Configure Features (global settings)
    ↓
Configure Split (global ratios)
    ↓
Execute (single button)
    ↓
Results (consistent dataset ready)
```

## Common Issues & Solutions

### Issue: "No labels available"
**Solution**: Go to Tab 2, create windows, split them first

### Issue: "Different features per label"
**Solution**: This is now impossible - global settings ensure consistency

### Issue: "Training tab not loading data"
**Solution**: Update training_callbacks.py to load from engineered-dataset-store

## Testing Commands

```bash
# Run app
cd d:\Workspaces\Master\ComputerScience\Thesis\GUI_app
python app.py

# Check for errors
# Open: http://127.0.0.1:8050/
# Navigate through all 5 tabs
# Check browser console for errors
```

## Next Implementation Priority

1. **HIGH**: Implement `execute_feature_engineering()` callback
2. **HIGH**: Update training tab to use engineered-dataset-store
3. **MEDIUM**: Add validation and error handling
4. **LOW**: Implement device testing tab
5. **LOW**: Add export functionality

---

**Last Updated**: Framework restructuring complete  
**Status**: UI ready, callbacks pending implementation
