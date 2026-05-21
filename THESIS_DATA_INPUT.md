# THESIS DATA COLLECTION — Input Template

**Date**: 2026-05-20
**Collector**: Nguyen Truong Minh Hoang
**Purpose**: Real experimental data for thesis Chapter 4 (Results)

---

## 1. TRAINING RESULTS

### 1.1 Neural Network (MLP)

- **Architecture**: Input(63) → Hidden(100) → Hidden(50) → Output(3)
- **Training time**: 0.22 seconds
- **Train accuracy**: 99.17%
- **Validation accuracy**: 100% (best_val_accuracy, early stopped at epoch 13)
- **Test accuracy**: 100%
- **Deployment accuracy** (with confidence threshold ≥0.6): _______%
- **Rejection rate**: _______%
- **Notes**: Orientation-invariant + DFT feature set (63 features), 3 classes: running/still/walking. sklearn MLPClassifier, early stopping enabled.
Per-Class Performance (test set):
| Activity | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Running  | 1.0000    | 1.0000 | 1.0000   | 56      |
| Still    | 1.0000    | 1.0000 | 1.0000   | 51      |
| Walking  | 1.0000    | 1.0000 | 1.0000   | 49      |
| Macro Avg| 1.0000    | 1.0000 | 1.0000   | 156     |

### 1.2 Random Forest

- **Architecture**: n_estimators=50, max_depth=10
- **Training time**: 0.12 seconds
- **Train accuracy**: 100%
- **Validation accuracy**: N/A (no held-out val split for RF)
- **Test accuracy**: 100%
- **Deployment accuracy** (with confidence threshold ≥0.6): _______%
- **Rejection rate**: _______%
- **Notes**: 63 orientation-invariant + DFT features, 3 classes: running/still/walking.
Per-Class Performance (test set):
| Activity | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Running  | 1.0000    | 1.0000 | 1.0000   | 56      |
| Still    | 1.0000    | 1.0000 | 1.0000   | 51      |
| Walking  | 1.0000    | 1.0000 | 1.0000   | 49      |
| Macro Avg| 1.0000    | 1.0000 | 1.0000   | 156     |

### 1.3 Support Vector Machine (SVM)

- **Architecture**: kernel='rbf', C=1.0, gamma='scale'
- **Training time**: 0.09 seconds
- **Train accuracy**: 100%
- **Validation accuracy**: N/A (no held-out val split for SVM)
- **Test accuracy**: 100%
- **Deployment accuracy** (with confidence threshold ≥0.6): _______%
- **Rejection rate**: _______%
- **Notes**: 63 orientation-invariant + DFT features, 3 classes: running/still/walking.
Per-Class Performance (test set):
| Activity | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Running  | 1.0000    | 1.0000 | 1.0000   | 56      |
| Still    | 1.0000    | 1.0000 | 1.0000   | 51      |
| Walking  | 1.0000    | 1.0000 | 1.0000   | 49      |
| Macro Avg| 1.0000    | 1.0000 | 1.0000   | 156     |

### 1.4 PyTorch MLP (Optional)

- **Architecture**: Input(63) → Hidden layers → Output(3), orientation-invariant + DFT features
- **Training time**: 2.54 seconds
- **Train accuracy**: 99.86%
- **Validation accuracy**: 100% (best_val_accuracy, early stopped at epoch 17)
- **Test accuracy**: 100%
- **Deployment accuracy** (with confidence threshold ≥0.6): _______%
- **Rejection rate**: _______%
- **Notes**: PyTorch custom MLP, Adam optimizer, early stopping. 63 features, 3 classes.
Per-Class Performance (test set):
| Activity | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Running  | 1.000     | 1.000  | 1.000    | 56      |
| Still    | 1.000     | 1.000  | 1.000    | 51      |
| Walking  | 1.000     | 1.000  | 1.000    | 49      |
| Macro Avg| 1.000     | 1.000  | 1.000    | 156     |

### 1.5 PyTorch 1D-CNN (Optional)

- **Architecture**: 1D-CNN (window_size=150, n_channels=6), orientation-invariant + DFT features
- **Training time**: 10.29 seconds
- **Train accuracy**: 100%
- **Validation accuracy**: 100% (best_val_accuracy, early stopped at epoch 33)
- **Test accuracy**: 100%
- **Deployment accuracy** (with confidence threshold ≥0.6): 99.4%, mean confidence 98.8%
- **Rejection rate**: 0.6%
- **Notes**: PyTorch 1D-CNN, Adam optimizer, early stopping. Raw 6-axis input (window_size=150 @ 100Hz = 1.5s), 3 classes.
Per-Class Performance (test set):
| Activity | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Running  | 1.000     | 1.000  | 1.000    | 56      |
| Still    | 1.000     | 1.000  | 1.000    | 51      |
| Walking  | 1.000     | 1.000  | 1.000    | 49      |
| Macro Avg| 1.000     | 1.000  | 1.000    | 156     |
| Weighted | 1.000     | 1.000  | 1.000    | 156     |

---

### 1.6 Neural Network (MLP) — 5 Activities

- **Architecture**: Input(63) → Hidden(100) → Hidden(50) → Output(5)
- **Training time**: 0.42 seconds
- **Train accuracy**: 99.72%
- **Validation accuracy**: 98.25% (best_val_accuracy, early stopped at epoch 25)
- **Test accuracy**: 98.69%
- **Deployment accuracy** (with confidence threshold ≥0.6): _______%
- **Rejection rate**: _______%
- **Notes**: Same feature set (63 features), 5 classes: running/still/walking/walking_downstairs/walking_upstairs.
Per-Class Performance (test set):
| Activity           | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| Running            | 1.0000    | 1.0000 | 1.0000   | 56      |
| Still              | 1.0000    | 1.0000 | 1.0000   | 51      |
| Walking            | 1.0000    | 0.9796 | 0.9897   | 49      |
| Walking Downstairs | 0.9487    | 0.9737 | 0.9610   | 38      |
| Walking Upstairs   | 0.9714    | 0.9714 | 0.9714   | 35      |
| Macro Avg          | 0.9840    | 0.9849 | 0.9844   | 229     |

### 1.7 Random Forest — 5 Activities

- **Architecture**: n_estimators=50, max_depth=10
- **Training time**: 0.15 seconds
- **Train accuracy**: 99.81%
- **Validation accuracy**: N/A
- **Test accuracy**: 98.25%
- **Deployment accuracy** (with confidence threshold ≥0.6): _______%
- **Rejection rate**: _______%
- **Notes**: 63 features, 5 classes.
Per-Class Performance (test set):
| Activity           | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| Running            | 1.0000    | 1.0000 | 1.0000   | 56      |
| Still              | 1.0000    | 1.0000 | 1.0000   | 51      |
| Walking            | 0.9792    | 0.9592 | 0.9691   | 49      |
| Walking Downstairs | 0.9250    | 0.9737 | 0.9487   | 38      |
| Walking Upstairs   | 1.0000    | 0.9714 | 0.9855   | 35      |
| Macro Avg          | 0.9808    | 0.9809 | 0.9807   | 229     |

### 1.8 Support Vector Machine (SVM) — 5 Activities

- **Architecture**: kernel='rbf', C=1.0, gamma='scale'
- **Training time**: 0.15 seconds
- **Train accuracy**: 97.55%
- **Validation accuracy**: N/A
- **Test accuracy**: 97.38%
- **Deployment accuracy** (with confidence threshold ≥0.6): _______%
- **Rejection rate**: _______%
- **Notes**: 63 features, 5 classes.
Per-Class Performance (test set):
| Activity           | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| Running            | 1.0000    | 1.0000 | 1.0000   | 56      |
| Still              | 1.0000    | 1.0000 | 1.0000   | 51      |
| Walking            | 0.9783    | 0.9184 | 0.9474   | 49      |
| Walking Downstairs | 0.8810    | 0.9737 | 0.9250   | 38      |
| Walking Upstairs   | 1.0000    | 0.9714 | 0.9855   | 35      |
| Macro Avg          | 0.9719    | 0.9727 | 0.9716   | 229     |

### 1.9 PyTorch MLP — 5 Activities

- **Architecture**: Input(63) → Hidden layers → Output(5), orientation-invariant + DFT features
- **Training time**: 1.94 seconds
- **Train accuracy**: 99.81%
- **Validation accuracy**: 98.69% (best_val_accuracy, early stopped at epoch 41)
- **Test accuracy**: 99.13%
- **Deployment accuracy** (with confidence threshold ≥0.6): _______%
- **Rejection rate**: _______%
- **Notes**: PyTorch MLP, 5 classes.
Per-Class Performance (test set):
| Activity           | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| Running            | 1.0000    | 1.0000 | 1.0000   | 56      |
| Still              | 1.0000    | 1.0000 | 1.0000   | 51      |
| Walking            | 1.0000    | 0.9796 | 0.9897   | 49      |
| Walking Downstairs | 0.9500    | 1.0000 | 0.9744   | 38      |
| Walking Upstairs   | 1.0000    | 0.9714 | 0.9855   | 35      |
| Macro Avg          | 0.9900    | 0.9902 | 0.9899   | 229     |

### 1.10 PyTorch 1D-CNN — 5 Activities

- **Architecture**: 1D-CNN (window_size=150, n_channels=6), raw 6-axis input
- **Training time**: 10.09 seconds
- **Train accuracy**: 99.25%
- **Validation accuracy**: 99.56% (best_val_accuracy, early stopped at epoch 24)
- **Test accuracy**: 99.13%
- **Deployment accuracy** (with confidence threshold ≥0.6): _______%
- **Rejection rate**: _______%
- **Notes**: PyTorch 1D-CNN, 5 classes.
Per-Class Performance (test set):
| Activity           | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| Running            | 1.0000    | 0.9821 | 0.9910   | 56      |
| Still              | 1.0000    | 1.0000 | 1.0000   | 51      |
| Walking            | 0.9796    | 0.9796 | 0.9796   | 49      |
| Walking Downstairs | 0.9744    | 1.0000 | 0.9870   | 38      |
| Walking Upstairs   | 1.0000    | 1.0000 | 1.0000   | 35      |
| Macro Avg          | 0.9908    | 0.9923 | 0.9915   | 229     |

### Edge Impulse NN — 3 Activities

- **Architecture**: NN (window_size=150, n_channels=6): 126-40-20-Dropout rate 0.2-3
- **Training time**:
- **Train accuracy**: 100.0%
- **Validation accuracy**: 100.0% (best_val_accuracy, early stopped at epoch 30)
- **Test accuracy**: 92.31%
- **Deployment accuracy** (with confidence threshold ≥0.6): _______%
- **Rejection rate**: _______%
- **Notes**: NN, 3 classes.

Metrics for Classifier
| Metric | Value |
| Area under ROC Curve | 1.00 |
| Weighted average Precision | 0.97 |
| Weighted average Recall | 0.96 |
| Weighted average F1 score | 0.96 |

Confusion matrix
| | running | still | walking | uncertain |
| running | 100% | 0% | 0% | 0% |
| still | 0% | 100% | 0% | 0% |
| walking | 0% | 0% | 80% | 20% |
| f1 score | 1.00 | 1.00 | 0.89 |

---

## 2. ON-DEVICE TEST RESULTS

### 2.1 Device Information

- **Device**: Seeed XIAO nRF52840
- **IMU**: LSM6DS3 (built-in)
- **Firmware**: Generated from framework (Date: _____________)
- **Orientation**: _____________ (e.g., "USB port facing up, mounted on wrist")
- **Test date**: _____________

### 2.2 Pytorch MLP — On-Device Performance

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

### 2.2 Pytorch CNN — On-Device Performance

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

**Problem: Not quantized with quantization int8 option -> inference performane low, hanging symptom**

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

### Edge Impulse NN — 3 Activities — On-Device Performance

Deployment target: C++ Library
Inference engine: EON Compiler
Deployment claim:
Model optimizations can increase on-device performance but may reduce accuracy. Performance estimate for Nordic nRF52840 DK (Cortex-M4F 64MHz).
Quantized (int8)

| | Spectral features | Classifier | Total |
| Latency | 8 ms. | 1 ms. | 9 ms. |
| Ram | 4.3K | 1.5K | 4.3K |
| Flash | - | 20.0K | - |
| Accuracy |  |  | 88.46% |

Actual result:
Still and Walking are good; Running is not recognizable

---

## 3. PREPROCESSING IMPACT (Optional Ablation Study)

### 3.1 Kalman Filter Impact

**Model tested**: _____________ (e.g., Neural Network)

| Preprocessing Configuration | Test Accuracy | On-Device Accuracy | Notes |
|----------------------------|---------------|-------------------|-------|
| LPF only (no Kalman) | ____.__% | ____.__% | _____________ |
| LPF + Kalman (Q=0.001, R=0.1) | ____.__% | ____.__% | _____________ |
| **Improvement** | **+_**.**%** | **+_**.**%** | |

**Observations**: ______________________________________________________________
________________________________________________________________________________

---

## 4. MEMORY AND RESOURCE USAGE

### 4.1 Code Size (from Arduino IDE compilation)

| Model | Sketch Size | Global Variables | Flash Used | SRAM Used | Notes |
|-------|-------------|------------------|------------|-----------|-------|
| Pytorch CNN | 145728 bytes | 126440 bytes | 17% | 53% | _____________ |
| Pytorch MLP | 177168 bytes | 49640 bytes | 21% | 20% | _____________ |
| Neural Network | 178248 bytes | 58416 bytes | 21% | 24% | _____________ |
| Random Forest | 187936 bytes | 49640 bytes | 23% | 20% | _____________ |
| SVM | 132952 bytes | 49632 bytes | 16% | 20% | _____________ |

**Maximum capacity** (Seeed XIAO nRF52840):

- Flash:  MB (811,008 bytes)
- SRAM:  KB (187,928 bytes)

### From Edge Impulse

Class: 3 activities
Model: NN
Sketch Size: 417856 bytes (51%) of program storage space. Maximum is 811008 bytes.
Global Variables: 74432 bytes (31%) of dynamic memory, leaving 163136 bytes for local variables. Maximum is 237568 bytes.

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
| TFLite (Keras surrogate) | _**.**% | ____.__% | ____.__% | _____________ |

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
