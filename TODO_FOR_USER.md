# USER ACTION CHECKLIST — Thesis Update

**Created**: 2026-05-05
**Context**: PR #3 is being merged; thesis needs data collection and screenshots
**Your role**: Provide experimental data, screenshots, and real measurements
**My role**: Write LaTeX content, generate figures, integrate your data

---

## ✅ QUICK CHECKLIST (High-Level)

- [ ] **CRITICAL**: Retrain all models with new pipeline (4-6 hours)
- [ ] **CRITICAL**: Run device tests and export data (2-3 hours)
- [ ] **MEDIUM**: Capture UI screenshots (30 minutes)
- [ ] **OPTIONAL**: Cross-device benchmark if hardware available (1-2 hours)

---

## 📋 DETAILED TASK LIST

### CATEGORY 1: MODEL RETRAINING (CRITICAL — HIGHEST PRIORITY)

#### ✅ Task 1.1: Data Preparation
**Estimated time**: 30 minutes

**Steps**:
1. Open the GUI application: `uv run python app.py`
2. Go to **Data** tab
3. If your existing recordings are short (<1.5s per activity):
   - [ ] Collect new longer recordings (≥1.5s per window = 150 samples @ 100Hz)
   - [ ] Aim for 5-10 seconds per activity for each recording session
   - [ ] Perform 10+ recording sessions per activity
4. If existing data is sufficient:
   - [ ] Load your datasets (running, still, walking, walking_downstairs, walking_upstairs)

**Output**: Datasets loaded and ready

**Notes**: Longer recordings allow sliding window to generate more training samples (50+ windows/class recommended)

---

#### ✅ Task 1.2: Preprocessing with Kalman Filter
**Estimated time**: 15 minutes

**Steps**:
1. Go to **Preprocessing** tab
2. Load each dataset one by one
3. Configure preprocessing settings:
   - [ ] Enable **Outlier Removal** (3-sigma, default)
   - [ ] Enable **Low-Pass Filter**: Cutoff=5Hz, Order=2 (or your preferred settings)
   - [ ] Enable **Kalman Filter**: ✅
     - Process Noise (Q): `0.001` (default is good)
     - Measurement Noise (R): `0.1` (default is good)
4. Click **Clean and Smooth Data**
5. Review the plotted signal to ensure Kalman is working (should be smoother than raw)
6. Use **draggable window selection** to select activity segments
   - Try to select full-length segments (close to target window size 150 samples)
   - Use **Generate Sliding Windows** feature if available to create multiple overlapping windows
7. Save windows for each dataset

**Output**: Window CSV files in `persistent_data/windows/` with Kalman preprocessing applied

**Screenshot opportunity**: Capture screenshot of Preprocessing tab showing Kalman settings → save as `ui_kalman_settings.png`

---

#### ✅ Task 1.3: Feature Engineering
**Estimated time**: 10 minutes

**Steps**:
1. Go to **Feature Engineering** tab
2. Load all saved windows
3. Configure Feature Engineering:
   - [ ] Feature Method: **Orientation-Invariant Time-Domain** (33 features) OR **Orientation-Invariant + DFT** (53 features)
   - [ ] Window Size: 150 samples (1.5 seconds @ 100Hz)
   - [ ] Normalization: **StandardScaler** (recommended)
   - [ ] Data Augmentation (optional but recommended):
     - Enable: ✅
     - Static labels: `still` (if you have "still" activity)
     - Augmentation factor: 2-3x
4. Click **Extract Features and Prepare Training Data**
5. Verify output: Training/validation/test split should show sufficient samples

**Output**: Feature CSV files in `persistent_data/training/` + `*_fe_metadata.json`

**Expected**: If you had 5 activities × 10 windows each = 50 windows total → ~110 training samples, 24 test samples (sufficient for proof-of-concept)

---

#### ✅ Task 1.4: Model Training
**Estimated time**: 20-30 minutes (per model × 4 models = 1.5-2 hours total)

**Steps** (repeat for each model type):

1. Go to **Training** tab
2. Select model type:
   - [ ] **Neural Network** (MLP): Hidden layers [128, 64], epochs 100, early stopping
   - [ ] **Random Forest**: n_estimators=100, max_depth=None
   - [ ] **SVM**: kernel='rbf', C=1.0, gamma='scale'
   - [ ] **PyTorch MLP** (optional): Same as NN but with PyTorch backend
3. Click **Train Model**
4. Wait for training to complete (1-10 minutes depending on model)
5. **Record these metrics** (write down or screenshot):
   - Train accuracy: _____%
   - Validation accuracy: _____%
   - Test accuracy: _____%
   - **Deployment accuracy** (with confidence threshold 0.6): _____%
   - **Rejection rate**: _____%
6. Save model (should auto-save to `persistent_data/models/`)

**Output**: 3-4 trained model `.joblib` files

**Data needed for thesis**:
```
Model Performance Summary:

Neural Network:
- Train: __.__% | Val: __.__% | Test: __.__% | Deployment (conf≥0.6): __.__%
- Rejection rate: __.__%

Random Forest:
- Train: __.__% | Val: __.__% | Test: __.__% | Deployment (conf≥0.6): __.__%
- Rejection rate: __.__%

SVM:
- Train: __.__% | Val: __.__% | Test: __.__% | Deployment (conf≥0.6): __.__%
- Rejection rate: __.__%
```

**Thesis table input**: Copy this data to `THESIS_DATA_INPUT.md` (I'll create this file for you)

---

### CATEGORY 2: ON-DEVICE TESTING (CRITICAL)

#### ✅ Task 2.1: Code Generation
**Estimated time**: 10 minutes per model

**Steps** (for each trained model):
1. Go to **Code Generation** tab
2. Select trained model from dropdown
3. Configure deployment:
   - Framework: **Arduino**
   - Target Board: **Seeed XIAO nRF52840**
   - Optimization: **Balanced**
   - Quantization: **None** (for first test)
   - Confidence Threshold: **0.6**
   - Smoothing Window: **3**
   - **Enable IIR Filter**: ✅ (if you want on-device filtering)
   - **Enable Kalman Filter**: ✅ (to match training preprocessing)
4. Click **Generate Embedded Code**
5. Save generated files to organized folder

**Output**: Generated `.ino` sketch + `.h` header + `.cpp` implementation

---

#### ✅ Task 2.2: Flash and Test on Device
**Estimated time**: 30-45 minutes per model (2-3 hours total for 4 models)

**Steps** (for each model):
1. Open generated `.ino` file in Arduino IDE
2. Install required libraries if prompted (LSM6DS3, Wire)
3. Select board: **Seeed XIAO nRF52840**
4. Flash to device (Upload)
5. Open Serial Monitor (115200 baud)
6. Verify initialization messages (IMU initialized, model loaded, etc.)

**For each activity** (running, still, walking, walking_downstairs, walking_upstairs):
7. Perform the activity for ~30 seconds while wearing device in same orientation as training data
8. Observe Serial Monitor for predictions
9. Record results:
   - Activity performed: __________
   - Predicted labels (most common): __________
   - Correct predictions: _____ / _____ (count)
   - Inference time (from debug output): _____ms

**Alternatively**: Use **Device Test** tab in GUI (easier):
1. Keep device connected via USB
2. Go to **Device Test** tab in GUI (while `.ino` is running on device)
3. Click **Start Inference**
4. Perform each activity for 30 seconds
5. GUI will plot sensor data + predictions in real-time
6. Click **Export Data** to save CSV

**Output**: Device test CSV files OR manual accuracy count

**Data needed for thesis**:
```
On-Device Test Results (Seeed XIAO nRF52840):

Neural Network Model:
Activity          | Accuracy | Inference Time
------------------|----------|---------------
Running           | __.__% | ___ms
Still             | __.__% | ___ms
Walking           | __.__% | ___ms
Walking Downstairs| __.__% | ___ms
Walking Upstairs  | __.__% | ___ms
OVERALL           | __.__% | ___ms avg

Random Forest Model:
[same table structure]

SVM Model:
[same table structure]
```

---

### CATEGORY 3: UI SCREENSHOTS (MEDIUM PRIORITY)

#### ✅ Task 3.1: Preprocessing Tab (Kalman Settings)
**Estimated time**: 5 minutes

**Steps**:
1. Open GUI, go to Preprocessing tab
2. Enable Kalman filter with settings visible
3. Ensure Q=0.001, R=0.1 values are shown in UI
4. Capture screenshot (full tab, not just panel)
5. Save as: `academic-paper-vietnamese/figures/ui_kalman_settings.png`

**Purpose**: Show users how to configure Kalman preprocessing

---

#### ✅ Task 3.2: Data Upload Tab
**Estimated time**: 3 minutes

**Steps**:
1. Go to Data tab with at least one dataset loaded
2. Show dataset name, file path, sample count visible
3. Capture screenshot
4. Save as: `academic-paper-vietnamese/figures/ui_data_upload.png`

**Purpose**: Demonstrate data upload workflow

---

#### ✅ Task 3.3: Training Results Display
**Estimated time**: 3 minutes

**Steps**:
1. Go to Training tab after training a model
2. Show performance metrics panel (train/val/test accuracy, confusion matrix if visible)
3. Capture screenshot
4. Save as: `academic-paper-vietnamese/figures/ui_results_display.png`

**Purpose**: Show training pipeline output

---

#### ✅ Task 3.4: Preprocessing Draggable Windows
**Estimated time**: 3 minutes

**Steps**:
1. Go to Preprocessing tab with a dataset loaded
2. Show the draggable window selection interface (if visible in your version)
3. Capture screenshot showing the drag-to-select functionality
4. Save as: `academic-paper-vietnamese/figures/ui_preprocessing_draggable.png`

**Purpose**: Demonstrate interactive window selection feature

---

### CATEGORY 4: OPTIONAL TASKS (LOW PRIORITY)

#### 🔲 Task 4.1: Cross-Device Benchmark (If Hardware Available)
**Estimated time**: 1-2 hours

**Steps**:
1. If you have ESP32 or M5Stack device:
   - Generate code for that platform (Code Generation tab → select target board)
   - Flash and test one model (e.g., Random Forest)
   - Record accuracy and inference time
2. Compare with XIAO results

**Data needed**:
```
Cross-Device Comparison:

Random Forest Model:
Device             | Accuracy | Inference Time | RAM Used
-------------------|----------|----------------|----------
Seeed XIAO nRF52   | __.__% | ___ms         | ___KB
ESP32              | __.__% | ___ms         | ___KB
```

**Thesis impact**: Strengthens multi-device deployment claim

---

#### 🔲 Task 4.2: Ablation Study (Training Without Kalman)
**Estimated time**: 1 hour

**Steps**:
1. Retrain ONE model (e.g., Neural Network) WITHOUT Kalman filter:
   - Go to Preprocessing tab
   - Disable Kalman filter (keep LPF only)
   - Re-extract features and retrain
2. Compare test accuracy: With Kalman __.__% vs Without Kalman __.__%
3. Record difference

**Thesis impact**: Quantifies Kalman filter contribution

---

## 📁 FILE ORGANIZATION

After completing tasks, your directory should have:

```
persistent_data/
├── models/
│   ├── neural_network_har_model_YYYYMMDD_HHMMSS.joblib
│   ├── random_forest_har_model_YYYYMMDD_HHMMSS.joblib
│   └── svm_har_model_YYYYMMDD_HHMMSS.joblib
├── training/
│   ├── [dataset]_train.csv
│   ├── [dataset]_val.csv
│   ├── [dataset]_test.csv
│   └── [dataset]_fe_metadata.json
└── generated/
    └── [model_name]/
        ├── [sketch].ino
        ├── [header].h
        └── [impl].cpp

academic-paper-vietnamese/figures/
├── ui_kalman_settings.png           ← NEW (Task 3.1)
├── ui_data_upload.png                ← NEW (Task 3.2)
├── ui_results_display.png            ← NEW (Task 3.3)
├── ui_preprocessing_draggable.png    ← NEW (Task 3.4)
└── [other figures I will generate]
```

---

## 📊 DATA COLLECTION SUMMARY

Create a file `THESIS_DATA_INPUT.md` in the root directory with this structure (I'll provide template):

```markdown
# THESIS DATA COLLECTION — 2026-05-05

## Training Results

### Neural Network
- Train accuracy: __.__%
- Validation accuracy: __.__%
- Test accuracy: __.__%
- Deployment accuracy (conf≥0.6): __.__%
- Rejection rate: __.__%

### Random Forest
[same]

### SVM
[same]

## On-Device Test Results (Seeed XIAO nRF52840)

### Neural Network
| Activity | Accuracy | Inference Time |
|----------|----------|----------------|
| Running | __.__% | ___ms |
| Still | __.__% | ___ms |
| Walking | __.__% | ___ms |
| Walking Downstairs | __.__% | ___ms |
| Walking Upstairs | __.__% | ___ms |
| **OVERALL** | **__.__%** | **___ms avg** |

[same for RF and SVM]

## Kalman Filter Impact (Optional Ablation)

| Model | Without Kalman | With Kalman | Improvement |
|-------|----------------|-------------|-------------|
| NN | __.__% | __.__% | +__.__% |

## Screenshots Captured
- [x] ui_kalman_settings.png
- [x] ui_data_upload.png
- [x] ui_results_display.png
- [x] ui_preprocessing_draggable.png
```

---

## ⏱️ ESTIMATED TIME BREAKDOWN

| Category | Task | Time | Priority |
|----------|------|------|----------|
| **1. Retraining** | Data prep | 30 min | 🔴 CRITICAL |
| | Preprocessing + Kalman | 15 min | 🔴 CRITICAL |
| | Feature engineering | 10 min | 🔴 CRITICAL |
| | Training (×4 models) | 1.5-2h | 🔴 CRITICAL |
| **2. Device Testing** | Code generation | 40 min | 🔴 CRITICAL |
| | Flash + test (×4 models) | 2-3h | 🔴 CRITICAL |
| **3. Screenshots** | All UI screenshots | 30 min | 🟡 MEDIUM |
| **4. Optional** | Cross-device benchmark | 1-2h | 🟢 LOW |
| | Ablation study | 1h | 🟢 LOW |
| **TOTAL** | | **5-8h** | (core tasks) |

---

## 🚀 RECOMMENDED WORKFLOW

### Session 1 (2-3 hours): Core Retraining
1. Task 1.1: Data preparation (30 min)
2. Task 1.2: Preprocessing with Kalman (15 min)
3. Task 1.3: Feature engineering (10 min)
4. Task 1.4: Train all models (1.5-2 hours)
5. **Checkpoint**: Take a break, record training metrics

### Session 2 (2-3 hours): Device Testing
1. Task 2.1: Generate code for all models (40 min)
2. Task 2.2: Flash and test each model (2-3 hours)
3. **Checkpoint**: Record device test results

### Session 3 (30 min): Screenshots
1. Tasks 3.1-3.4: Capture all UI screenshots
2. Organize files
3. Fill in `THESIS_DATA_INPUT.md`

---

## ❓ WHAT IF I GET STUCK?

### Issue: Training accuracy is low (<70%)
**Possible causes**:
- Dataset too small → collect more data or use augmentation
- Window size mismatch → ensure consistent 150 samples
- Preprocessing mismatch → verify Kalman settings match

**Solution**: Check training logs, try RF (more robust to small datasets)

---

### Issue: Device predictions are all wrong
**Possible causes**:
- Orientation mismatch → device mounted differently than training
- Preprocessing mismatch → ensure on-device Kalman matches training
- Feature extraction bug → verify generated code compiles without warnings

**Solution**:
1. Check Serial Monitor for debug output
2. Verify `# PRED: class=X conf=Y.YYYY` lines
3. Compare feature values (Python vs C++) if possible

---

### Issue: Can't capture screenshots
**Solution**: Use screen capture tool (Windows: Win+Shift+S, Mac: Cmd+Shift+4, Linux: Screenshot tool)

---

## ✉️ HOW TO SEND ME THE DATA

Once you've completed the critical tasks:

1. **Create `THESIS_DATA_INPUT.md`** (using template above)
2. **Fill in all accuracy/timing numbers**
3. **Attach screenshots** to the GitHub issue or upload to a shared folder
4. **Optional**: Attach CSV exports from Device Test tab
5. **Reply to this issue** with: "Data collection complete — ready for thesis writing"

I will then:
- Integrate your data into Chapter 4 tables
- Generate confusion matrices from your test results
- Create Kalman comparison plots
- Write LaTeX content for Chapters 3-5
- Update THESIS_REPORT_INSTRUCTIONS.md with completion status

---

## 📋 FINAL CHECKLIST

Before considering this task "done", verify:

- [ ] At least 3 models trained (NN, RF, SVM) with new Kalman pipeline
- [ ] Training metrics recorded (train/val/test/deployment accuracy)
- [ ] On-device testing completed for all models on XIAO
- [ ] On-device accuracy + inference time recorded
- [ ] At least 2 UI screenshots captured (Kalman settings + one other)
- [ ] `THESIS_DATA_INPUT.md` created and filled with real numbers
- [ ] All files organized in correct directories

**When all checkboxes are ticked**, you're done! 🎉

Let me know if you have any questions about any of these tasks.

