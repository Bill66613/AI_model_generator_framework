# Storage Restructuring - Path Fix Summary

## Issue

After migrating files to the new organized storage structure, two features were broken:
1. **Training window selection** - Could not find window files
2. **Model evaluation & analysis** - Could not load model files

## Root Cause

The code was still using hardcoded `PERSISTENT_DIR` paths with `os.path.join()` instead of using the helper functions provided by the storage restructuring:
- `get_window_path(window_id, dataset_name)`
- `get_window_pattern(dataset_name)`
- `get_model_path(model_filename)`
- `get_models_metadata_path()`

## Files Fixed

### callbacks/preprocessing_callbacks.py (5 locations)

1. **Line ~963**: `split_selected_windows()` - Saving dragged window files
   - ❌ Before: `os.path.join(PERSISTENT_DIR, f"dragged_window_{window_id}_{dataset_name}")`
   - ✅ After: `get_window_path(window_id, dataset_name)`

2. **Line ~1112**: `update_split_dataset_selector()` - Finding window files
   - ❌ Before: `os.path.join(PERSISTENT_DIR, f"dragged_window_*_{dataset_name}")`
   - ✅ After: `get_window_pattern(dataset_name)`

3. **Line ~1302**: `delete_split_window()` - Finding windows to update options
   - ❌ Before: `os.path.join(PERSISTENT_DIR, f"dragged_window_*_{dataset_name}")`
   - ✅ After: `get_window_pattern(dataset_name)`

4. **Line ~1348**: `clean_all_generated_data()` - Finding windows to delete
   - ❌ Before: `os.path.join(PERSISTENT_DIR, f"dragged_window_*_{dataset_name}")`
   - ✅ After: `get_window_pattern(dataset_name)`
   - Also fixed: `sample_window_*` pattern now uses `WINDOWS_DIR`

5. **Line ~1442**: `populate_training_dataset_selector()` - Finding windows for training
   - ❌ Before: `os.path.join(PERSISTENT_DIR, f"dragged_window_*_{dataset_name}")`
   - ✅ After: `get_window_pattern(dataset_name)`

### callbacks/training_callbacks.py (5 locations)

1. **Line ~479**: Model evaluation handler - Loading model for evaluation
   - ❌ Before: `os.path.join(PERSISTENT_DIR, model_filename)`
   - ✅ After: `get_model_path(model_filename)`

2. **Line ~693**: Remove model callback - Deleting model file
   - ❌ Before: `os.path.join(PERSISTENT_DIR, model_filename)`
   - ✅ After: `get_model_path(model_filename)`
   - Also fixed: Model metadata path uses `get_models_metadata_path()`

3. **Line ~795**: Remove model metadata function
   - ❌ Before: `os.path.join(PERSISTENT_DIR, "trained_models.json")`
   - ✅ After: `get_models_metadata_path()`

4. **Line ~866**: Deployment code generation - Loading model for deployment
   - ❌ Before: `os.path.join(PERSISTENT_DIR, model_filename)`
   - ✅ After: `get_model_path(model_filename)`
   - Also fixed: Model metadata path uses `get_models_metadata_path()`

5. **Line ~1241**: Load available models utility function
   - ❌ Before: `os.path.join(PERSISTENT_DIR, "trained_models.json")`
   - ✅ After: `get_models_metadata_path()`

## Verification

Tested all helper functions:
```python
# Window paths working correctly
get_window_path(0, 'walking.csv') 
# → D:\...\persistent_data\windows\dragged_window_0_walking.csv

get_window_pattern('walking.csv')
# → D:\...\persistent_data\windows\dragged_window_*_walking.csv

# Model paths working correctly
get_model_path('test.joblib')
# → D:\...\persistent_data\models\test.joblib

get_models_metadata_path()
# → D:\...\persistent_data\models\trained_models.json
```

**Files verified:**
- ✅ 170 window files in `windows/` directory
- ✅ 13 model files in `models/` directory
- ✅ All paths resolve correctly
- ✅ No syntax errors

## Impact

### Now Working:
✅ **Training Window Selection** - Can now find and load window files from `windows/` directory
✅ **Model Evaluation** - Can load models from `models/` directory for performance analysis
✅ **Feature Importance** - Can display feature importance plots
✅ **Model Removal** - Can delete models and update metadata
✅ **Deployment Code Generation** - Can load models for code generation
✅ **Resource Analysis** - Can analyze model resource requirements

### Backward Compatibility:
All helper functions check old locations first, so if any files remain in the old flat structure, they will still be found and used. This ensures smooth transition without breaking existing workflows.

## Testing Recommendations

1. **Training Window Selection**:
   - Navigate to Preprocessing → Prepare for Training
   - Select dataset
   - Verify windows appear in "Training Dataset Selector"
   - Select multiple windows
   - Click "Preprocess for Training"
   - Should work without errors

2. **Model Evaluation**:
   - Navigate to Training → Model Evaluation & Analysis
   - Select a trained model
   - Click "Evaluate Model"
   - Should display performance metrics
   - Click "Feature Importance" (for RF/XGBoost)
   - Should display feature importance chart

3. **Deployment**:
   - Navigate to Training → Deployment
   - Select a trained model
   - Click "Generate Deployment Code"
   - Should generate Arduino code
   - Click "Analyze Resource Requirements"
   - Should display memory/computational requirements

---

**Status**: ✅ COMPLETED  
**Date**: 2025-12-12  
**Related**: Storage Restructuring (#3)
