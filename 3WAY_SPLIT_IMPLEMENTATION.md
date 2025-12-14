# 3-Way Train-Validation-Test Split Implementation

**Status**: ✅ COMPLETED  
**Date**: Implementation completed  
**Scope**: Full 3-way data splitting with configurable ratios and CV fallback

## Overview

Implemented a comprehensive 3-way train-validation-test split system to replace the previous 2-way train-test split. The system supports:

- **Configurable split ratios** via UI sliders
- **Cross-validation fallback** when validation = 0%
- **Automatic test ratio calculation** with minimum 10% constraint
- **Backward compatibility** with existing 2-way split files

## Implementation Details

### 1. UI Updates (`layouts/preprocessing.py`)

**Changed Components:**

**Before:**

- Single slider: `train-test-split` (10%-90%, default 80%)
- Section title: "Train-Test Split Configuration"
- Button: "Perform Train-Test Split"

**After:**

- Three components:
  - `train-split` slider (40%-80%, step 0.05, default 60%)
  - `val-split` slider (0%-30%, step 0.05, default 20%)
    - Mark at 0% labeled "CV" to indicate cross-validation mode
  - `test-split-display` div (calculated, read-only)
- Section title: "Train-Validation-Test Split Configuration"
- Button: "Perform Train-Val-Test Split"
- Info message: "Configure data split ratios. Set validation to 0% to use cross-validation instead."

**Split Display Callback** (new):

- Inputs: `train-split`, `val-split` values
- Outputs:
  - `test-split-display`: Shows calculated test percentage
  - `split-ratio-info`: Shows split info or CV warning
- Logic:
  - Calculates: test = 100% - train - val
  - Enforces: test >= 10% minimum
  - Shows different messages:
    - val = 0%: "⚙️ Validation set is 0% - will use 5-fold cross-validation on training set"
    - val > 0%: Shows all three ratios in colored spans

### 2. Split Logic (`callbacks/preprocessing_callbacks.py`)

**Function: `perform_enhanced_train_val_test_split()`** (renamed from `perform_enhanced_train_test_split`)

**Signature Change:**

```python
# Before
def perform_enhanced_train_test_split(n_clicks, preprocessed_data, split_ratio, random_state)

# After
def perform_enhanced_train_val_test_split(n_clicks, preprocessed_data, train_ratio, val_ratio, random_state)
```

**Split Strategy:**

**When val_ratio = 0 (CV Mode):**

```python
# 2-way split only
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_ratio, random_state=random_state, stratify=y
)
X_val, y_val = None, None
```

**When val_ratio > 0 (3-way split):**

```python
# Step 1: Split off test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=test_ratio, random_state=random_state, stratify=y
)

# Step 2: Split remaining into train and validation
val_size_adjusted = val_ratio / (train_ratio + val_ratio)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
)
```

**Stored Data Structure:**

```python
split_data = {
    'X_train': X_train.tolist(),
    'X_val': X_val.tolist() if X_val is not None else [],
    'X_test': X_test.tolist(),
    'y_train': y_train.tolist(),
    'y_val': y_val.tolist() if y_val is not None else [],
    'y_test': y_test.tolist(),
    'feature_names': feature_names,
    'train_ratio': train_ratio,
    'val_ratio': val_ratio,
    'test_ratio': test_ratio,
    'random_state': random_state,
    'has_validation': val_ratio > 0
}
```

### 3. Visualization Updates

**3-Region Visualization** (when val > 0):

- **Training data**: Circle markers (blue/orange/green/etc.)
- **Validation data**: Square markers (same colors)
- **Test data**: Diamond markers (same colors)
- **Separators**: Two vertical lines
  - Green dashed line: Train | Val boundary
  - Red dashed line: Val | Test boundary
- **Title**: Shows all three ratios
- **Annotation**: Shows all three sample counts

**2-Region Visualization** (when val = 0):

- **Training data**: Circle markers
- **Test data**: Diamond markers
- **Separator**: Single vertical line (Train | Test)
- **Title**: Shows "(CV Mode)" indicator
- **Annotation**: Shows "⚙️ Using 5-fold CV"

### 4. Data Saving (`callbacks/preprocessing_callbacks.py`)

**Function: `save_preprocessed_training_data()`**

**Files Saved:**

**When val > 0 (3-way):**

- `{dataset_name}_train.csv` - Training features + labels
- `{dataset_name}_val.csv` - Validation features + labels
- `{dataset_name}_test.csv` - Test features + labels
- `{dataset_name}_metadata.json` - Split metadata

**When val = 0 (2-way):**

- `{dataset_name}_train.csv` - Training features + labels
- `{dataset_name}_test.csv` - Test features + labels
- `{dataset_name}_metadata.json` - Split metadata
- No validation file created

**Metadata Structure:**

```json
{
    "dataset_name": "string",
    "feature_names": ["feature1", "feature2", ...],
    "train_ratio": 0.6,
    "val_ratio": 0.2,
    "test_ratio": 0.2,
    "random_state": 42,
    "train_samples": 1200,
    "val_samples": 400,
    "test_samples": 400,
    "has_validation": true,
    "train_file": "path/to/train.csv",
    "val_file": "path/to/val.csv",
    "test_file": "path/to/test.csv",
    "created_at": "2024-01-15T10:30:00"
}
```

### 5. Training Integration (`callbacks/training_callbacks.py`)

**Function: `handle_training_actions()`**

**Changes:**

- Replaced window-based loading with file-based loading
- Uses `get_training_data_path()` helper for consistent paths
- Automatically detects most recent training dataset
- Loads validation file if it exists

**Loading Logic:**

```python
# Load training data
train_file = get_training_data_path(dataset_name, 'train')
test_file = get_training_data_path(dataset_name, 'test')
val_file = get_training_data_path(dataset_name, 'val')

# Check for validation data
has_validation = os.path.exists(val_file)

if has_validation:
    val_df = pd.read_csv(val_file)
    X_val = val_df.drop('label', axis=1).values
    y_val = val_df['label'].values
```

**Function Signatures Updated:**

```python
# Basic training
def perform_basic_training(model, X_train, X_test, y_train, y_test, model_type, X_val=None, y_val=None)

# Hyperparameter optimization
def perform_hyperparameter_optimization(model, X_train, X_test, y_train, y_test, model_type, X_val=None, y_val=None)
```

**Training Behavior:**

**When validation data provided:**

- Train on training set (no CV)
- Evaluate on validation set
- Report validation accuracy
- Final test on test set

**When validation data not provided (CV mode):**

- Train with 5-fold cross-validation
- Report CV mean accuracy
- Final test on test set

**Model Metadata Updates:**

```python
model_info = {
    'training_samples': len(X_train),
    'val_samples': len(X_val) if X_val is not None else 0,
    'test_samples': len(X_test),
    'val_accuracy': val_accuracy if val_accuracy else 0,
    'test_accuracy': test_accuracy,
    'cv_accuracy': cv_accuracy if use_cv else 0,
    'used_validation': not use_cv,
    # ... other fields
}
```

## Usage Guide

### For Users

**Default Setup (60/20/20 split):**

1. Preprocess data and extract features
2. Leave sliders at default positions:
   - Train: 60%
   - Validation: 20%
   - Test: 20% (auto-calculated)
3. Click "Perform Train-Val-Test Split"
4. Save split data
5. Train models (will use validation set)

**Cross-Validation Mode (0% validation):**

1. Set validation slider to 0%
2. Adjust train slider as desired (e.g., 80%)
3. Test will auto-adjust (e.g., 20%)
4. Notice CV indicator message
5. Click "Perform Train-Val-Test Split"
6. Save split data
7. Train models (will use 5-fold CV)

**Custom Split Ratios:**

1. Adjust train slider (40%-80%)
2. Adjust val slider (0%-30%)
3. Test auto-calculates (minimum 10%)
4. Click "Perform Train-Val-Test Split"
5. Save and train

### For Developers

**Accessing Split Data:**

```python
from config.config import get_training_data_path

# Load data
train_file = get_training_data_path(dataset_name, 'train')
val_file = get_training_data_path(dataset_name, 'val')
test_file = get_training_data_path(dataset_name, 'test')
metadata_file = get_training_data_path(dataset_name, 'metadata')

# Check if validation exists
has_val = os.path.exists(val_file)
```

**Creating 3-Way Split Programmatically:**

```python
from sklearn.model_selection import train_test_split

# Example: 60/20/20 split
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Step 1: Split off test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=test_ratio, stratify=y
)

# Step 2: Split remaining into train/val
val_adjusted = val_ratio / (train_ratio + val_ratio)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_adjusted, stratify=y_temp
)
```

## Backward Compatibility

The implementation maintains backward compatibility:

**Old 2-way split files:**

- Can still be loaded and used
- Training will use CV mode (no val file present)
- Metadata may have old format with `split_ratio` instead of `train_ratio`/`val_ratio`/`test_ratio`
- Code handles both formats gracefully

**Migration Strategy:**

- No automatic migration needed
- Users can re-split existing datasets with new 3-way system
- Old files continue working in CV mode

## Benefits

1. **Professional ML Practice**: Proper 3-way split is standard in machine learning
2. **Better Hyperparameter Tuning**: Validation set prevents test set leakage
3. **Flexible Configuration**: Users can choose split ratios based on data size
4. **CV Fallback**: Small datasets can still use cross-validation
5. **Clear Visualization**: Users see exactly how data is divided
6. **Backward Compatible**: Doesn't break existing workflows

## Testing Checklist

- [x] UI renders with 3 sliders
- [x] Test ratio auto-calculates correctly
- [x] Minimum test ratio (10%) enforced
- [x] CV warning shows when val=0%
- [x] 3-way split executes without errors
- [x] 2-way split (CV mode) executes without errors
- [x] Visualization shows 3 regions correctly
- [x] Visualization shows CV indicator
- [x] Files saved correctly (with/without val file)
- [x] Metadata includes all split info
- [x] Training loads validation data
- [x] Training uses validation when available
- [x] Training falls back to CV when no validation
- [ ] End-to-end workflow tested
- [ ] Performance verified with real data

## Files Modified

### Layouts

- `layouts/preprocessing.py` - UI components for 3-way split

### Callbacks

- `callbacks/preprocessing_callbacks.py` - Split logic, visualization, saving
- `callbacks/training_callbacks.py` - Training with validation support

### Configuration

- No changes needed - `config/config.py` already supports 'val' type

## Future Enhancements

1. **Dataset Selector**: Allow users to choose which training dataset to use
2. **Split Validation**: Warn if splits are imbalanced across classes
3. **Validation Metrics**: Enhanced display of validation vs test metrics
4. **Auto-Split**: Suggest optimal split ratios based on dataset size
5. **Stratification Info**: Show class distribution across splits

## Related Documentation

- `STORAGE_RESTRUCTURE_SUMMARY.md` - Storage organization
- `FEATURE_TERMINOLOGY_FIX.md` - Feature extraction updates
- `TODO.md` - Next steps (feature info display, results UI improvements)
