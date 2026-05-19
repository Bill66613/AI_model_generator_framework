# 📋 USER INPUT REQUIREMENTS FOR THESIS UPDATE

**Priority:** URGENT - Required for thesis completion  
**Timeline:** Complete within 1-2 weeks  
**Purpose:** Replace placeholder values and missing figures in Vietnamese thesis  

---

## 🚨 CRITICAL EXPERIMENTAL DATA NEEDED

### A. TRAINING PIPELINE RESULTS (MUST HAVE)

**Action Required:** Run complete training session and collect these exact metrics:

1. **Model Accuracy Results**
   ```
   Current Placeholders in thesis:
   - Random Forest: 94.5% (VERIFY OR REPLACE)
   - Neural Network: 96.2% (VERIFY OR REPLACE)  
   - SVM: 93.8% (VERIFY OR REPLACE)
   
   COLLECT: 
   - Run training in Tab 4 with HAR dataset
   - Screenshot final accuracy results
   - Save confusion matrix for each algorithm
   - Record training time for each model
   ```

2. **Per-Class Performance Metrics**
   ```
   COLLECT: For each activity class (6 activities)
   - Precision, Recall, F1-score per class
   - Support (number of samples) per class  
   - Confusion matrix showing misclassifications
   ```

3. **Model File Sizes**
   ```
   COLLECT: Check persistent_data/models/ folder
   - Random Forest: X.X KB
   - Neural Network: Y.Y KB
   - SVM: Z.Z KB
   ```

### B. DEVICE DEPLOYMENT PERFORMANCE (MUST HAVE)

**Action Required:** Deploy generated code on Seeed XIAO and measure:

1. **Inference Timing**
   ```
   Current Placeholder: 12-18ms on ARM Cortex-M4
   
   MEASURE:
   - Time per prediction (ms)
   - Feature extraction time (ms)
   - Model inference time (ms)  
   - Total end-to-end latency (ms)
   ```

2. **Memory Usage**
   ```
   Current Placeholders: 95-182KB Flash, 1.4-2.4% RAM
   
   MEASURE:
   - Flash memory used (KB)
   - RAM usage during inference (KB)
   - Static vs dynamic memory allocation
   ```

3. **Power Consumption** 
   ```
   Current Placeholder: 30mW average, 65+ hour battery
   
   MEASURE: (using multimeter or nRF Power Profiler)
   - Idle current (mA)
   - Sampling current (mA) 
   - Feature extraction current (mA)
   - Prediction current (mA)
   - Calculate battery life with typical Li-Po battery
   ```

### C. TRAINING-DEPLOYMENT PARITY VALIDATION (HIGH PRIORITY)

**Action Required:** Demonstrate the zero-padding fix impact:

1. **Before/After Accuracy Comparison**
   ```
   COLLECT:
   - Train model with zero-padding (old method): XX.X%
   - Train model with edge-replication (fixed method): YY.Y%  
   - Show improvement in accuracy
   ```

2. **Feature Distribution Analysis**
   ```
   COLLECT:
   - Screenshot showing acc_mag_min values with zero-padding (shows 0.0)
   - Screenshot showing acc_mag_min values with edge-replication (shows >0.0)
   - This proves the "physically impossible" zero magnitude issue was fixed
   ```

---

## 📸 UI SCREENSHOTS NEEDED (HIGH PRIORITY)

### Complete 6-Tab Workflow Documentation

**Setup Instructions:**
1. Use a real HAR dataset (6 activities: walking, running, sitting, standing, walking_upstairs, walking_downstairs)
2. Run complete pipeline from Tab 1 → Tab 6
3. Take high-quality screenshots (1920×1080 or higher)
4. Save as PNG with descriptive names

### Tab 1: Data Management
- [ ] **File:** `ui_tab1_data_upload.png` - CSV upload interface with file list
- [ ] **File:** `ui_tab1_data_visualization.png` - Time series plots of sensor data
- [ ] **File:** `ui_tab1_label_assignment.png` - Activity label assignment interface

### Tab 2: Signal Preprocessing  
- [ ] **File:** `ui_tab2_raw_data.png` - Raw sensor data plots
- [ ] **File:** `ui_tab2_filtering.png` - Butterworth filter configuration
- [ ] **File:** `ui_tab2_windowing.png` - **DRAGGABLE TIME WINDOWING** (KEY FEATURE!)

### Tab 3: Feature Engineering
- [ ] **File:** `ui_tab3_feature_modes.png` - 6 feature extraction modes selection
- [ ] **File:** `ui_tab3_feature_stats.png` - Feature statistics display
- [ ] **File:** `ui_tab3_train_split.png` - Train/validation/test split configuration

### Tab 4: Model Training
- [ ] **File:** `ui_tab4_algorithm_selection.png` - RF/SVM/NN algorithm options
- [ ] **File:** `ui_tab4_hyperparameters.png` - Hyperparameter grid configuration  
- [ ] **File:** `ui_tab4_training_results.png` - Final accuracy and performance metrics

### Tab 5: Code Generation
- [ ] **File:** `ui_tab5_platform_selection.png` - 11 code generators selection
- [ ] **File:** `ui_tab5_optimization.png` - Optimization mode (speed/memory/power/balanced)
- [ ] **File:** `ui_tab5_generated_code.png` - Preview of generated C++ code

### Tab 6: Device Testing
- [ ] **File:** `ui_tab6_serial_monitor.png` - Real-time serial communication
- [ ] **File:** `ui_tab6_predictions.png` - Live activity predictions display
- [ ] **File:** `ui_tab6_confusion_matrix.png` - Device testing confusion matrix

---

## 🔧 HARDWARE DEMONSTRATION (HIGH PRIORITY)

### Device in Operation
- [ ] **Photo:** Seeed XIAO nRF52840 connected via USB with sensor active
- [ ] **Video:** 30-60 second clip showing real-time activity recognition
- [ ] **Screenshot:** Serial monitor output showing continuous predictions

### Power Measurement Setup
- [ ] **Photo:** Multimeter or power profiler connected to measure current
- [ ] **Screenshot:** Power profiler software showing current consumption graphs
- [ ] **Data:** CSV or spreadsheet with measured power values

---

## 📊 COMPARATIVE ANALYSIS DATA (MEDIUM PRIORITY)

### Edge Impulse Comparison (If Possible)
- [ ] **Data:** Same HAR dataset uploaded to Edge Impulse
- [ ] **Results:** Edge Impulse accuracy vs our framework accuracy  
- [ ] **Analysis:** Model size, inference time comparison
- [ ] **Screenshots:** Edge Impulse training interface for comparison

### Multi-Platform Testing (Nice to Have)
- [ ] **ESP32 Deployment:** Same model deployed on ESP32 board
- [ ] **Performance Comparison:** XIAO vs ESP32 timing/memory
- [ ] **Photo:** Multiple devices running same model

---

## 📈 DATA COLLECTION CHECKLIST

### Session 1: Complete Training Pipeline
**Time Required:** 2-3 hours  
**Steps:**
1. [ ] Load HAR dataset in Tab 1
2. [ ] Process through Tab 2 (preprocessing + windowing)  
3. [ ] Configure features in Tab 3 (try multiple modes)
4. [ ] Train all 3 algorithms in Tab 4 
5. [ ] Screenshot all results and record metrics
6. [ ] Save model files (.joblib) for later analysis

### Session 2: Code Generation & Deployment  
**Time Required:** 2-3 hours
**Steps:**
1. [ ] Generate code for Seeed XIAO in Tab 5
2. [ ] Compile and flash to device
3. [ ] Test inference in Tab 6
4. [ ] Measure power consumption  
5. [ ] Record inference timing and accuracy

### Session 3: UI Documentation
**Time Required:** 1-2 hours  
**Steps:**
1. [ ] Run pipeline again with focus on screenshots
2. [ ] Capture high-quality images of each tab
3. [ ] Document unique features (draggable windowing)
4. [ ] Create video demonstration if possible

---

## 📝 DATA ORGANIZATION TEMPLATE

### Create This Folder Structure:
```
thesis_experimental_data/
├── training_results/
│   ├── accuracy_metrics.json
│   ├── confusion_matrices/
│   ├── training_logs/  
│   └── model_files/
├── deployment_results/
│   ├── inference_timing.csv
│   ├── memory_usage.txt
│   ├── power_measurements.csv
│   └── serial_outputs/
├── ui_screenshots/
│   ├── tab1_data/
│   ├── tab2_preprocessing/  
│   ├── tab3_features/
│   ├── tab4_training/
│   ├── tab5_codegen/
│   └── tab6_testing/
└── hardware_photos/
    ├── device_setup/
    ├── power_measurement/
    └── operation_videos/
```

---

## ⚡ PRIORITY RANKING

### 🔴 CRITICAL (Must complete this week)
1. **Training accuracy data** - Replace placeholder numbers  
2. **Device inference timing** - Real performance metrics
3. **UI screenshots of draggable windowing** - Unique feature demonstration

### 🟡 HIGH PRIORITY (Complete within 2 weeks)  
4. **Power consumption measurements** - Battery life calculations
5. **Complete UI workflow screenshots** - All 6 tabs documentation  
6. **Training-deployment parity demonstration** - Before/after fix

### 🟢 MEDIUM PRIORITY (Nice to have)
7. **Edge Impulse comparison** - Competitive analysis
8. **Multi-device deployment** - Cross-platform validation
9. **Video demonstrations** - Dynamic showcases

---

## 📞 SUPPORT AVAILABLE

### If You Need Help With:
- **Python Scripts:** I can provide code to automate data collection
- **Figure Generation:** I can create charts/graphs from your collected data  
- **LaTeX Integration:** I can help integrate figures into Vietnamese thesis
- **Data Analysis:** I can help interpret and present experimental results

### How to Share Data:
- **Screenshots:** Upload to shared folder or repository
- **Numerical Data:** Copy/paste into chat or share CSV files
- **Questions:** Ask specific questions about any measurement or procedure

---

## ✅ COMPLETION CHECKLIST

When you have completed data collection, verify:

- [ ] All placeholder accuracy numbers replaced with real measurements
- [ ] Device performance data (timing, memory, power) collected  
- [ ] High-quality UI screenshots of all unique features captured
- [ ] Training-deployment parity demonstrated with before/after comparison
- [ ] Hardware setup photos and operation videos recorded
- [ ] Data organized in structured folders for easy access
- [ ] Ready to proceed with thesis chapter updates and figure generation

**Estimated Total Time:** 6-10 hours spread over 1-2 weeks  
**Expected Outcome:** Complete experimental validation for thesis defense  
**Priority:** Complete training data and device measurements first, then UI documentation