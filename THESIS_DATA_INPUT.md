# THESIS DATA COLLECTION — Input Template

**Date**: _____________
**Collector**: Nguyen Truong Minh Hoang
**Purpose**: Real experimental data for thesis Chapter 4 (Results)

---

## 1. TRAINING RESULTS

### 1.1 Neural Network (MLP)
- **Architecture**: Input(33 or 53) → Hidden(128) → Hidden(64) → Output(5)
- **Training time**: _______ seconds
- **Train accuracy**: _______%
- **Validation accuracy**: _______%
- **Test accuracy**: _______%
- **Deployment accuracy** (with confidence threshold ≥0.6): _______%
- **Rejection rate**: _______%
- **Notes**: _______________________________________________________________

### 1.2 Random Forest
- **Architecture**: n_estimators=100, max_depth=None
- **Training time**: _______ seconds
- **Train accuracy**: _______%
- **Validation accuracy**: _______%
- **Test accuracy**: _______%
- **Deployment accuracy** (with confidence threshold ≥0.6): _______%
- **Rejection rate**: _______%
- **Notes**: _______________________________________________________________

### 1.3 Support Vector Machine (SVM)
- **Architecture**: kernel='rbf', C=1.0, gamma='scale'
- **Training time**: _______ seconds
- **Train accuracy**: _______%
- **Validation accuracy**: _______%
- **Test accuracy**: _______%
- **Deployment accuracy** (with confidence threshold ≥0.6): _______%
- **Rejection rate**: _______%
- **Notes**: _______________________________________________________________

### 1.4 PyTorch MLP (Optional)
- **Architecture**: _______________
- **Training time**: _______ seconds
- **Train accuracy**: _______%
- **Validation accuracy**: _______%
- **Test accuracy**: _______%
- **Deployment accuracy** (with confidence threshold ≥0.6): _______%
- **Rejection rate**: _______%
- **Notes**: _______________________________________________________________

---

## 2. ON-DEVICE TEST RESULTS

### 2.1 Device Information
- **Device**: Seeed XIAO nRF52840
- **IMU**: LSM6DS3 (built-in)
- **Firmware**: Generated from framework (Date: _____________)
- **Orientation**: _____________ (e.g., "USB port facing up, mounted on wrist")
- **Test date**: _____________

### 2.2 Neural Network — On-Device Performance

| Activity | Correct Predictions | Total Samples | Accuracy | Avg Inference Time | Notes |
|----------|---------------------|---------------|----------|-------------------|-------|
| Running | _____ | _____ | ____.__%  | _____ms | _____________ |
| Still | _____ | _____ | ____.__%  | _____ms | _____________ |
| Walking | _____ | _____ | ____.__%  | _____ms | _____________ |
| Walking Downstairs | _____ | _____ | ____.__%  | _____ms | _____________ |
| Walking Upstairs | _____ | _____ | ____.__%  | _____ms | _____________ |
| **OVERALL** | **_____** | **_____** | **____.__%** | **_____ms** | |

**Confusion observed**: _________________________________________________________
**Unknown predictions**: _____ out of _____ (____%)

### 2.3 Random Forest — On-Device Performance

| Activity | Correct Predictions | Total Samples | Accuracy | Avg Inference Time | Notes |
|----------|---------------------|---------------|----------|-------------------|-------|
| Running | _____ | _____ | ____.__%  | _____ms | _____________ |
| Still | _____ | _____ | ____.__%  | _____ms | _____________ |
| Walking | _____ | _____ | ____.__%  | _____ms | _____________ |
| Walking Downstairs | _____ | _____ | ____.__%  | _____ms | _____________ |
| Walking Upstairs | _____ | _____ | ____.__%  | _____ms | _____________ |
| **OVERALL** | **_____** | **_____** | **____.__%** | **_____ms** | |

**Confusion observed**: _________________________________________________________
**Unknown predictions**: _____ out of _____ (____%)

### 2.4 SVM — On-Device Performance

| Activity | Correct Predictions | Total Samples | Accuracy | Avg Inference Time | Notes |
|----------|---------------------|---------------|----------|-------------------|-------|
| Running | _____ | _____ | ____.__%  | _____ms | _____________ |
| Still | _____ | _____ | ____.__%  | _____ms | _____________ |
| Walking | _____ | _____ | ____.__%  | _____ms | _____________ |
| Walking Downstairs | _____ | _____ | ____.__%  | _____ms | _____________ |
| Walking Upstairs | _____ | _____ | ____.__%  | _____ms | _____________ |
| **OVERALL** | **_____** | **_____** | **____.__%** | **_____ms** | |

**Confusion observed**: _________________________________________________________
**Unknown predictions**: _____ out of _____ (____%)

---

## 3. PREPROCESSING IMPACT (Optional Ablation Study)

### 3.1 Kalman Filter Impact

**Model tested**: _____________ (e.g., Neural Network)

| Preprocessing Configuration | Test Accuracy | On-Device Accuracy | Notes |
|----------------------------|---------------|-------------------|-------|
| LPF only (no Kalman) | ____.__% | ____.__% | _____________ |
| LPF + Kalman (Q=0.001, R=0.1) | ____.__% | ____.__% | _____________ |
| **Improvement** | **+___.__%** | **+___.__%** | |

**Observations**: ______________________________________________________________
________________________________________________________________________________

---

## 4. MEMORY AND RESOURCE USAGE

### 4.1 Code Size (from Arduino IDE compilation)

| Model | Sketch Size | Global Variables | Flash Used | SRAM Used | Notes |
|-------|-------------|------------------|------------|-----------|-------|
| Neural Network | _____ bytes | _____ bytes | _____% | _____% | _____________ |
| Random Forest | _____ bytes | _____ bytes | _____% | _____% | _____________ |
| SVM | _____ bytes | _____ bytes | _____% | _____% | _____________ |

**Maximum capacity** (Seeed XIAO nRF52840):
- Flash: 1 MB (1,048,576 bytes)
- SRAM: 256 KB (262,144 bytes)

---

## 5. CROSS-DEVICE BENCHMARK (Optional)

### 5.1 Device Comparison

**Model tested**: _____________ (e.g., Random Forest)

| Device | Accuracy | Avg Inference Time | Flash Used | SRAM Used | Notes |
|--------|----------|-------------------|------------|-----------|-------|
| Seeed XIAO nRF52840 | ____.__% | _____ms | _____ bytes | _____ bytes | _____________ |
| ESP32 (if available) | ____.__% | _____ms | _____ bytes | _____ bytes | _____________ |
| M5Stack (if available) | ____.__% | _____ms | _____ bytes | _____ bytes | _____________ |

**Observations**: ______________________________________________________________
________________________________________________________________________________

---

## 6. TFLITE DEPLOYMENT (Optional)

### 6.1 TFLite Micro Conversion Results

**Model tested**: _____________ (e.g., Neural Network)

| Quantization | Model Size | Conversion Time | Accuracy (test set) | On-Device Accuracy | Notes |
|--------------|------------|-----------------|-------------------|-------------------|-------|
| None (float32) | _____ bytes | _____s | ____.__% | ____.__% | _____________ |
| INT8 | _____ bytes | _____s | ____.__% | ____.__% | _____________ |

### 6.2 RF/SVM TFLite via Surrogate

**Model**: Random Forest

| Method | Surrogate Agreement | Test Accuracy | On-Device Accuracy | Notes |
|--------|---------------------|---------------|-------------------|-------|
| Direct C++ | N/A | ____.__% | ____.__% | _____________ |
| TFLite (ONNX pipeline) | N/A | [FAILED] | N/A | onnx2tf doesn't support TreeEnsemble |
| TFLite (Keras surrogate) | ___.__% | ____.__% | ____.__% | _____________ |

**Observations**: ______________________________________________________________
________________________________________________________________________________

---

## 7. UI SCREENSHOTS CHECKLIST

- [ ] `ui_kalman_settings.png` — Preprocessing tab with Kalman filter settings visible
- [ ] `ui_data_upload.png` — Data tab with dataset loaded
- [ ] `ui_results_display.png` — Training tab showing model performance metrics
- [ ] `ui_preprocessing_draggable.png` — Preprocessing tab with window selection interface
- [ ] `ui_feature_engineering.png` — (Optional) Feature Engineering tab
- [ ] `ui_code_generation.png` — (Optional) Code Generation tab with settings

**Screenshot location**: `academic-paper-vietnamese/figures/`

---

## 8. EXPORTED DATA FILES

### 8.1 Test Set Predictions (for confusion matrix generation)

**Files to provide**:
- [ ] `test_predictions_nn.csv` — Columns: `true_label, predicted_label, confidence`
- [ ] `test_predictions_rf.csv`
- [ ] `test_predictions_svm.csv`

**Format example**:
```
true_label,predicted_label,confidence
running,running,0.9234
still,still,0.8765
walking,walking_downstairs,0.6543
...
```

### 8.2 Device Test Logs (Optional)

**Files to provide**:
- [ ] `device_test_nn_[activity].csv` — Raw sensor data + predictions from Device Test tab
- [ ] `device_test_rf_[activity].csv`
- [ ] `device_test_svm_[activity].csv`

**Format**: As exported by Device Test tab (aX, aY, aZ, gX, gY, gZ, prediction)

---

## 9. SIGNAL QUALITY DATA (for Kalman comparison plots)

### 9.1 Raw vs Filtered Signals

**Files to provide** (Optional but helpful):
- [ ] `signal_raw.csv` — One activity, 3 seconds, raw sensor data
- [ ] `signal_lpf_only.csv` — Same segment, LPF only
- [ ] `signal_kalman.csv` — Same segment, LPF + Kalman

**Format**: Same as sensor CSV (aX, aY, aZ, gX, gY, gZ)

**Purpose**: I'll generate comparison plot showing Kalman noise reduction

---

## 10. NOTES AND OBSERVATIONS

### 10.1 Training Process

**Issues encountered**: _________________________________________________________
________________________________________________________________________________
________________________________________________________________________________

**Unexpected results**: _________________________________________________________
________________________________________________________________________________
________________________________________________________________________________

### 10.2 On-Device Testing

**Issues encountered**: _________________________________________________________
________________________________________________________________________________
________________________________________________________________________________

**Orientation sensitivity**: ____________________________________________________
________________________________________________________________________________

**Battery life observations** (if tested): ____________________________________
________________________________________________________________________________

### 10.3 General Comments

________________________________________________________________________________
________________________________________________________________________________
________________________________________________________________________________
________________________________________________________________________________

---

## ✅ COMPLETION CHECKLIST

Before sending this file to me, verify:

- [ ] Section 1 (Training Results) — All model accuracies filled in
- [ ] Section 2 (On-Device Results) — At least NN + RF tested on XIAO
- [ ] Section 4 (Memory Usage) — Code sizes from Arduino IDE compilation
- [ ] Section 7 (Screenshots) — At least 2 screenshots captured
- [ ] Section 8 (Exported Files) — Test prediction CSVs exported (for confusion matrices)
- [ ] File saved and ready to share

**Estimated completion date**: _____________

**Ready to send?**: ☐ Yes ☐ Not yet (missing: ___________________________)

---

**Next steps after you provide this data**:
1. I'll integrate numbers into Chapter 4 LaTeX tables
2. I'll generate confusion matrices from test prediction CSVs
3. I'll create Kalman comparison plots (if signal CSVs provided)
4. I'll write updated LaTeX content for Chapters 3-5
5. I'll update THESIS_REPORT_INSTRUCTIONS.md with completion status

