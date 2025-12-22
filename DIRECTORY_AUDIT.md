# Working Directory Integration Audit

## Summary
Several callback files are still using hardcoded directory constants instead of the dynamic working directory from `working-directory-store`.

## Status by File

### ✅ COMPLETED - data_callbacks.py
- `upload_files()` - Uses `State('working-directory-store', 'data')`
- `use_default_directory()` - Sets working directory
- `migrate_old_files()` - Uses working directory
- `display_current_working_dir()` - Displays working directory

### ✅ COMPLETED - code_generation_callbacks.py
- `populate_model_selector()` - Uses working directory for models
- `display_model_info()` - Uses working directory for models
- `generate_embedded_code()` - Uses working directory for models
- `analyze_resources()` - Uses working directory for models

### ✅ COMPLETED - feature_engineering_callbacks.py
- `populate_activity_labels()` - Uses working directory
- `update_windows_per_label()` - Uses working directory
- `execute_feature_engineering()` - Uses working directory

### 🔄 PARTIALLY COMPLETE - training_callbacks.py
**Completed:**
- `save_model_metadata()` - Updated to accept base_dir parameter
- `load_trained_model_options()` - Updated to accept base_dir parameter
- `load_training_data_summary()` - Updated to accept base_dir parameter
- `enable_training_components()` - Uses working directory

**Remaining:**
- `handle_training_actions()` - Needs State('working-directory-store', 'data') added
- Other training callbacks that access training_dir need similar updates
- Need to pass base_dir to save_model_metadata() when called

### ❌ NEEDS UPDATE - preprocessing_callbacks.py
**Issues Found:**
- Uses `METADATA_FILE` directly (20+ occurrences)
- Uses `PERSISTENT_DIR` directly (multiple callbacks)
- Uses `DATASETS_DIR` directly

**Affected Callbacks:**
1. Line 203, 219: `load_dataset_for_preprocessing()` - reads METADATA_FILE
2. Line 428: `display_raw_data()` - uses PERSISTENT_DIR
3. Line 496, 501: `save_smoothed_data()` - uses PERSISTENT_DIR, metadata_file
4. Line 529: `update_sampling_rate()` - uses METADATA_FILE
5. Line 804: `save_labeled_window()` - uses METADATA_FILE
6. Line 970: `display_dragged_windows()` - uses METADATA_FILE
7. Line 1245: `delete_window()` - uses METADATA_FILE
8. Line 1515, 1550: `generate_sliding_windows()` - uses METADATA_FILE
9. Line 1628: `toggle_window_generation()` - uses METADATA_FILE
10. Line 1768: `apply_all_cleanup()` - uses METADATA_FILE
11. Line 1878: `cleanup_all_datasets()` - uses METADATA_FILE

**Recommendation:**
All preprocessing callbacks need to add `State('working-directory-store', 'data')` and construct paths dynamically:
```python
base_dir = working_dir if working_dir else PERSISTENT_DIR
metadata_file = os.path.join(base_dir, 'metadata.json')
datasets_dir = os.path.join(base_dir, 'datasets')
windows_dir = os.path.join(base_dir, 'windows')
```

### ❌ NEEDS UPDATE - feature_engineering_callbacks.py
**Issues Found:**
- Uses `METADATA_FILE` directly (8 occurrences)
- Uses `PERSISTENT_DIR` directly
- Uses `WINDOWS_DIR` directly

**Affected Callbacks:**
1. Line 34, 38: `populate_dataset_selector_fe()` - reads METADATA_FILE
2. Line 73, 77: `display_dataset_windows()` - reads METADATA_FILE
3. Line 229: `generate_training_data()` - reads METADATA_FILE

**Recommendation:**
All feature engineering callbacks need to add `State('working-directory-store', 'data')` and construct paths dynamically.

### ❌ NEEDS UPDATE - training_callbacks.py
**Issues Found:**
- Uses `get_models_metadata_path()` which returns hardcoded path
- Uses `PERSISTENT_DIR` directly for training directory
- Uses `MODELS_DIR` indirectly

**Affected Callbacks:**
1. Line 60-70: `update_model_metadata()` - uses get_models_metadata_path()
2. Line 78-80: `populate_model_selector_training()` - uses get_models_metadata_path()
3. Line 98-107: `populate_training_dataset_selector()` - uses PERSISTENT_DIR/training
4. Line 639-647: `populate_dataset_selector_for_training()` - uses PERSISTENT_DIR/training
5. Line 1478-1482: `train_model()` - uses get_models_metadata_path()
6. Line 1520-1524: `delete_model()` - uses get_models_metadata_path()

**Recommendation:**
All training callbacks need to:
1. Add `State('working-directory-store', 'data')`
2. Construct paths: `models_dir = os.path.join(base_dir, 'models')`
3. Construct paths: `training_dir = os.path.join(base_dir, 'training')`

## Required Changes

### Pattern for All Callbacks:
```python
@callback(
    Output(...),
    Input(...),
    State('working-directory-store', 'data'),  # ADD THIS
    prevent_initial_call=True
)
def my_callback(..., base_dir):  # ADD base_dir parameter
    # Use stored base directory or default to PERSISTENT_DIR
    if not base_dir:
        base_dir = PERSISTENT_DIR
    
    # Construct all paths dynamically
    metadata_file = os.path.join(base_dir, 'metadata.json')
    datasets_dir = os.path.join(base_dir, 'datasets')
    windows_dir = os.path.join(base_dir, 'windows')
    training_dir = os.path.join(base_dir, 'training')
    models_dir = os.path.join(base_dir, 'models')
    
    # Use dynamically constructed paths
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
```

## Impact Assessment

### High Priority (Breaks functionality with custom working dir):
1. **preprocessing_callbacks.py** - All data preprocessing operations
2. **feature_engineering_callbacks.py** - Training data generation
3. **training_callbacks.py** - Model training and management

### Files Modified:
- All callbacks that read/write data files
- All callbacks that access metadata.json
- All callbacks that load/save models

## Testing Required After Updates:
1. Set custom working directory (e.g., `D:\test_workspace`)
2. Upload dataset → verify saved to `{base_dir}/datasets/`
3. Preprocess data → verify metadata updated in `{base_dir}/metadata.json`
4. Generate windows → verify saved to `{base_dir}/windows/`
5. Feature engineering → verify training data in `{base_dir}/training/`
6. Train model → verify model saved to `{base_dir}/models/`
7. Code generation → verify loads from `{base_dir}/models/`
8. Switch working directory → verify all tabs use new location

## Estimated Changes:
- **preprocessing_callbacks.py**: ~15 callback updates
- **feature_engineering_callbacks.py**: ~3 callback updates
- **training_callbacks.py**: ~6 callback updates
- **Total**: ~24 callback functions need modification
