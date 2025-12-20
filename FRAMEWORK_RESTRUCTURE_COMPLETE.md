# Framework Restructuring Complete

## 🎯 Overview

The GUI application has been successfully upgraded from a 3-tab to a 5-tab structure, with proper separation of concerns following ML best practices.

## 📊 New Tab Structure

### Tab 1: 📊 Data Management
- **Purpose**: CSV upload, label assignment, sampling rate configuration
- **File**: `layouts/data_upload.py`
- **Status**: ✅ No changes needed (existing functionality)

### Tab 2: 🔬 Signal Preprocessing
- **Purpose**: Clean sensor data and create time windows
- **File**: `layouts/preprocessing.py`
- **Changes**: ✅ Removed feature engineering section
- **Features**:
  - Dataset selection and status display
  - Clean & smooth controls
  - Time window application
  - Interactive window selection graph
  - Manual window dragging
  - Sliding window generation
  - Window management (save, load, delete)

### Tab 3: ⚙️ Feature Engineering
- **Purpose**: Unified feature engineering with consistent settings
- **File**: `layouts/feature_engineering.py`
- **Status**: ✅ NEW - Complete UI created
- **Features**:
  - **Step 1**: Multi-label dataset selection
    - Select multiple activity labels at once
    - View windows per label
    - Select All / Clear buttons
  
  - **Step 2**: Global feature configuration
    - Feature extraction method (all/time-domain/frequency-domain/raw)
    - Normalization method (standard/minmax/robust/none)
    - Feature count display
  
  - **Step 3**: Train/Val/Test split configuration
    - Train split: 50-90% (slider)
    - Validation split: 5-30% (slider)
    - Test split: Auto-calculated
    - Random state for reproducibility
  
  - **Step 4**: Execute
    - Single button applies settings to ALL selected labels
    - Results display
    - Dataset statistics table

### Tab 4: 🎯 Model Training
- **Purpose**: Train and evaluate models
- **File**: `layouts/training.py`
- **Status**: ⚠️ Needs minor updates to load from engineered-dataset-store

### Tab 5: 📡 Device Testing
- **Purpose**: Real-time UART connection and testing
- **File**: `layouts/device_test.py`
- **Status**: ✅ Placeholder created
- **Planned Features**:
  - Serial port connection
  - Real-time sensor plotting
  - Live activity classification
  - Data recording
  - Performance metrics

## 🔧 Technical Implementation

### Files Created
1. ✅ `layouts/feature_engineering.py` (353 lines)
2. ✅ `layouts/device_test.py` (180 lines)
3. ✅ `callbacks/feature_engineering_callbacks.py` (placeholder with TODO)

### Files Modified
1. ✅ `layouts/preprocessing.py`
   - Removed: Feature engineering section (lines 626-938)
   - Updated: Header title to "Signal Preprocessing & Windowing"
   
2. ✅ `app.py`
   - Updated: 3 tabs → 5 tabs
   - Added: Imports for feature_engineering and device_test layouts
   - Added: Import for feature_engineering_callbacks

### Callback Structure

**callbacks/feature_engineering_callbacks.py** (to be implemented):
- `populate_activity_labels()`: Load available datasets from metadata
- `update_windows_per_label()`: Display window counts per label
- `handle_label_selection_buttons()`: Select All / Clear functionality
- `update_feature_count()`: Display feature count based on selection
- `calculate_test_split()`: Calculate test percentage (100% - train% - val%)
- `execute_feature_engineering()`: **CRITICAL CALLBACK**
  - Load ALL windows from ALL selected labels
  - Extract features using SAME method for ALL
  - Normalize using SINGLE scaler for ALL
  - Combine into single dataset
  - Split train/val/test on combined data
  - Save to engineered-dataset-store

## 🎯 Key Improvements

### Problem Solved: Feature Engineering Consistency
**Before**: Each dataset processed individually
- Risk of different normalization methods
- Risk of different feature selections
- Risk of different train/val/test ratios
- Difficult to ensure consistency

**After**: All datasets processed together
- ✅ Same normalization applied to ALL labels
- ✅ Same features extracted from ALL labels
- ✅ Same split ratios applied to ALL labels
- ✅ Single execution prevents human error
- ✅ Prevents data leakage (split AFTER combining)

### Workflow Benefits
1. **Clearer separation of concerns**
   - Tab 2: Signal processing only
   - Tab 3: Feature engineering only
   - Tab 4: Model training only

2. **Better ML practices**
   - Consistent preprocessing pipeline
   - Unified normalization scaler
   - Proper train/val/test splitting
   - Prevents data leakage

3. **Improved user experience**
   - Logical workflow progression
   - Clear tab labels with emojis
   - Step-by-step guidance
   - Single-button execution

## 📝 Next Steps

### Immediate (Required for functionality)
1. **Implement feature_engineering_callbacks.py**
   - Critical: `execute_feature_engineering()` callback
   - Load from `persistent_data/<label>/` directories
   - Use existing feature extraction code from utils/
   - Ensure single scaler for normalization
   - Save to dcc.Store for training tab

2. **Update training.py callbacks**
   - Modify to load from `engineered-dataset-store`
   - Remove per-dataset loading logic

3. **Test complete workflow**
   - Data Upload → Signal Preprocessing → Feature Engineering → Training

### Future Enhancements
1. **Device Testing Tab**
   - Implement serial port connection
   - Real-time plotting
   - Live classification

2. **Advanced Features**
   - Feature importance visualization
   - Data augmentation options
   - Export trained models for deployment

## 🚀 How to Run

```bash
cd d:\Workspaces\Master\ComputerScience\Thesis\GUI_app
python app.py
```

Open browser to: http://127.0.0.1:8050/

## ✅ Testing Checklist

- [x] App runs without errors
- [x] All 5 tabs appear
- [x] Tab 1 (Data Management) loads correctly
- [x] Tab 2 (Signal Preprocessing) loads correctly
- [x] Tab 3 (Feature Engineering) loads correctly
- [x] Tab 4 (Model Training) loads correctly
- [x] Tab 5 (Device Testing) loads correctly
- [ ] Feature engineering callbacks implemented
- [ ] Multi-label selection works
- [ ] Global settings apply uniformly
- [ ] Engineered dataset ready for training

## 📖 Architecture Notes

### Data Flow
```
Tab 1: Data Upload
  ↓ (CSV + labels)
Tab 2: Signal Preprocessing
  ↓ (cleaned windows)
Tab 3: Feature Engineering
  ↓ (engineered dataset with consistent settings)
Tab 4: Model Training
  ↓ (trained model)
Tab 5: Device Testing
  ↓ (real-time validation)
```

### Key Design Principle
**Consistency First**: The new structure ensures that all activity labels undergo identical preprocessing, preventing subtle bugs and ensuring fair model comparison.

### Storage Strategy
- **Windowed data**: `persistent_data/<label>/dragged_window_X.csv`
- **Engineered features**: `engineered-dataset-store` (dcc.Store in memory)
- **Trained models**: `models/` directory
- **Metadata**: `persistent_data/metadata.json`

## 🎨 UI/UX Improvements

1. **Visual Clarity**
   - Each tab has clear emoji icon
   - Descriptive tab names
   - Color-coded sections

2. **Workflow Guidance**
   - Step-by-step instructions
   - Progress indicators
   - Clear action buttons

3. **Error Prevention**
   - Global settings prevent inconsistency
   - Validation before execution
   - Clear feedback messages

---

**Status**: Framework restructuring complete ✅  
**Next Action**: Implement feature_engineering_callbacks.py  
**Priority**: HIGH (required for functionality)
