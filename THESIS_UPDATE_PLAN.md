# THESIS UPDATE PLAN — Session 2026-05-05

**Status**: Initial draft for review
**PR Context**: PR #3 (bugfix/tflite_deployment) will be merged — 21 files changed, 2 new findings, TFLite improvements
**Goal**: Update thesis with latest changes, create structured plan, identify user support needed

---

## EXECUTIVE SUMMARY

The thesis is ~80% complete with most chapters written in Vietnamese. The main gaps are:

1. **Technical Content**: Need to incorporate 2 new findings from PR #3 (Kalman filter, CNN metadata fix)
2. **Data Collection**: Placeholder numbers in Chapter 4 need real experimental results
3. **User Assets**: Missing UI screenshots, confusion matrices, deployment accuracy data
4. **Structure**: Good overall narrative; needs polish and consistency checks

---

## NEW TECHNICAL FINDINGS FROM PR #3

### Finding 15: CNN Code Generation Metadata Mismatch ✅ FIXED

**Problem**: CNN models showed wrong feature count in filenames (`f33` instead of `f6`) and confusing UI display mixing raw windows with FE method labels.

**Root Cause**: Fallback logic loaded FE metadata from wrong session; CNN models don't use extracted features but system treated them as if they did.

**Solution**:
- Code generation callback now detects CNN models and uses channel placeholders (`ch0-ch5`)
- Filename correctly shows `f6` (6 input channels per timestep)
- UI distinguishes "raw windows" from "extracted features"

**Thesis Impact**:
- Ch.3 §Code Generation: Add note about multi-architecture support (CNN vs feature-based)
- Ch.5 §Pipeline Correctness: Example of metadata consistency across architectures

### Finding 16: Kalman Filter with Exact Deployment Parity ✅ IMPLEMENTED

**Problem**: Existing IIR filter had parity gap (Python `filtfilt` = non-causal, C++ `lfilter` = causal) causing phase shifts in features.

**Solution**:
- Implemented constant-velocity Kalman filter per channel
- Kalman is inherently causal → Python and C++ produce **identical results**
- Tunable Q (process noise) and R (measurement noise) parameters
- Generated C++ code has exact same math as Python training

**Thesis Impact**:
- Ch.3 §Preprocessing: Add new subsection on Kalman filter (first zero-parity-gap preprocessing)
- Ch.5 §Training-Deployment Parity: Highlight this as parity breakthrough (no filtfilt vs lfilter gap)
- Differentiate from Edge Impulse (they don't offer Kalman filtering)

### Other PR #3 Improvements

- **TFLite for RF/SVM**: Keras surrogate via knowledge distillation when ONNX→TF fails
- **Quantization robustness**: Synthetic calibration data when representative data unavailable
- **ONNX deployment**: Reuses base class platform-specific IMU code for consistency
- **Device Test**: Better prediction persistence (last_activity label on every line)

---

## CHAPTER-BY-CHAPTER UPDATE PLAN

### Chapter 1: Giới thiệu (Introduction) — STATUS: ✅ 95% COMPLETE

**Current State**: Reframed for multi-device deployment; XIAO as reference platform; 6 objectives listed

**Updates Needed**:
- ✅ No major changes required (narrative is correct)
- Consider mentioning Kalman filter as 7th objective if it's a major contribution

**User Support Needed**: None

---

### Chapter 2: Công trình liên quan (Related Work) — STATUS: ✅ 100% COMPLETE

**Current State**: Written in Vietnamese, 11 new refs added (augmentation, confidence, parity)

**Updates Needed**:
- ✅ No changes required (already comprehensive)
- Optional: Add reference to Kalman filtering in IMU preprocessing if available

**User Support Needed**: None

---

### Chapter 3: Phương pháp luận (Methodology) — STATUS: ⚠️ 85% COMPLETE

**Current State**:
- Reframed for multi-device
- Has sections on FE, padding strategy, augmentation, confidence threshold
- Multi-device generator matrix added

**Updates Needed**:

1. **NEW SECTION: §3.X Bộ lọc Kalman cho Tiền xử lý Tín hiệu (Kalman Filter for Signal Preprocessing)**
   - Mô tả constant-velocity model: state [position, velocity], observation [position]
   - Công thức toán học: F, H, Q, R matrices
   - Parameters: process_noise (Q), measurement_noise (R), sampling rate
   - **Key selling point**: Causal (forward-only) → ZERO parity gap vs Python training
   - So sánh với IIR: filtfilt (non-causal) vs lfilter (causal) has phase shift; Kalman không có

2. **UPDATE: §3.Y Tạo mã triển khai (Code Generation)**
   - Add note: framework hỗ trợ hai kiến trúc (feature-based: NN/RF/SVM và end-to-end: CNN)
   - CNN: raw window input (150×6), không FE, không scaling
   - Feature-based: FE → scaling → model weights
   - Validator phải architecture-aware (Finding 9 + Finding 15)

3. **UPDATE: §3.Z Triển khai TFLite Micro**
   - For RF/SVM: ONNX→TF pipeline hoặc Keras surrogate via knowledge distillation
   - TreeEnsembleClassifier (RF) không được hỗ trợ bởi onnx2tf → fallback to surrogate
   - Surrogate: small Keras MLP trained on RF/SVM's soft predictions

**User Support Needed**:
- [ ] Kalman filter before/after signal plots (show noise reduction + parity)
- [ ] Diagram: Python Kalman vs C++ Kalman (identical output)

---

### Chapter 4: Kết quả (Results) — STATUS: ⚠️ 60% COMPLETE (PLACEHOLDER NUMBERS)

**Current State**:
- Has structure for results
- Multi-device deployment matrix table
- **CRITICAL**: Placeholder accuracy numbers, no real data

**Updates Needed**:

1. **RETRAIN ALL MODELS** with fixed pipeline:
   - Use edge-value replication (not zero-padding) ✅ Already fixed
   - Use longer recordings (≥1.5s per window)
   - Use sliding window to generate 50+ windows/class
   - Include Kalman filter preprocessing

2. **NEW TABLE: §4.X Tác động của Bộ lọc Kalman**
   | Model | Without Kalman | With Kalman | Improvement |
   |-------|----------------|-------------|-------------|
   | NN    | [PLACEHOLDER] % | [PLACEHOLDER] % | [PLACEHOLDER] % |
   | RF    | [PLACEHOLDER] % | [PLACEHOLDER] % | [PLACEHOLDER] % |
   | SVM   | [PLACEHOLDER] % | [PLACEHOLDER] % | [PLACEHOLDER] % |

3. **NEW TABLE: §4.Y So sánh Triển khai TFLite**
   | Model Type | Direct C++ | TFLite (ONNX) | TFLite (Surrogate) |
   |------------|-----------|---------------|-------------------|
   | NN/MLP     | ✅ | ✅ | N/A |
   | CNN        | ✅ | ✅ | N/A |
   | RF         | ✅ | ❌ (onnx2tf không hỗ trợ) | ✅ (80-90% agreement) |
   | SVM        | ✅ | ❌ | ✅ |

4. **UPDATE: Device Test Results**
   - On-device accuracy với/không Kalman
   - Inference time với Kalman overhead
   - Confusion matrix (actual data, not placeholder)

**User Support Needed**:
- [ ] **CRITICAL**: Retrain models with new pipeline
- [ ] **CRITICAL**: Collect device test data (XIAO + optional ESP32/M5Stack)
- [ ] Generate confusion matrices from test set
- [ ] Measure inference time with Kalman filter enabled
- [ ] CSV export from Device Test tab (for plotting)

---

### Chapter 5: Thảo luận (Discussion) — STATUS: ✅ 90% COMPLETE

**Current State**:
- Has section on training-deployment parity
- Has section on augmentation design
- Has section on confidence threshold
- Separated architectural multi-device support from single-platform benchmark

**Updates Needed**:

1. **NEW SUBSECTION: §5.X Đột phá Tương đồng Tiền xử lý: Bộ lọc Kalman**
   - Giải thích tại sao IIR có parity gap (filtfilt ≠ lfilter)
   - Kalman là bộ lọc đầu tiên với zero parity gap
   - So sánh với Edge Impulse (họ không có Kalman filtering)
   - Ý nghĩa: causal preprocessing + exact parity → reliable deployment

2. **UPDATE: §5.Y Kiến trúc Đa mô hình (Multi-Architecture Support)**
   - CNN vs feature-based models: khác nhau về input, processing, validation
   - Finding 15 as case study: metadata consistency across architectures
   - So sánh với EI: họ có separate "processing blocks", chúng tôi có unified pipeline

3. **UPDATE: §5.Z Triển khai TFLite cho RF/SVM**
   - ONNX→TF pipeline fragile (10+ undeclared deps, onnx2tf doesn't support tree ops)
   - Keras surrogate as pragmatic fallback
   - Direct C++ generation vẫn là recommended path (faithful, no approximation, smaller binary)
   - Agreement metric (80-90%) shows surrogate quality

**User Support Needed**: None (analysis of existing data)

---

### Chapter 6: Kết luận (Conclusion) — STATUS: ✅ 95% COMPLETE

**Current State**:
- 7 contributions listed (including augmentation, confidence, multi-device)
- Future work section
- Multi-device contribution explicit

**Updates Needed**:

1. **ADD to contributions list**:
   - **#8**: Kalman filter with exact deployment parity — first zero-gap preprocessing filter
   - **#9** (optional): Multi-architecture code generation (CNN + feature-based in unified pipeline)

2. **UPDATE: Future work**:
   - Cross-device benchmark (currently only XIAO quantified)
   - Multi-subject dataset validation (UCI HAR, WISDM)
   - Temperature scaling for confidence calibration
   - Additional preprocessing: Savitzky-Golay with causal implementation

**User Support Needed**: None

---

## FIGURES AND VISUALIZATIONS NEEDED

### HIGH PRIORITY (Required for defense)

1. **`figures/kalman_signal_comparison.png`**
   - 3 subplots: Raw signal, IIR filtered, Kalman filtered
   - Show noise reduction + phase preservation
   - Caption: "So sánh các phương pháp lọc tín hiệu IMU"

2. **`figures/kalman_parity_verification.png`**
   - Python vs C++ Kalman output (overlapping lines → identical)
   - Caption: "Xác minh tương đồng hoàn toàn giữa Python và C++ Kalman filter"

3. **`figures/confusion_matrix_nn.png`** ← ALREADY IN TODO
   - From retrained NN model (after Kalman + edge-replication fixes)

4. **`figures/ui_kalman_settings.png`**
   - Screenshot of Preprocessing tab with Kalman filter settings
   - Shows Q, R parameter inputs

### MEDIUM PRIORITY (Nice to have)

5. **`figures/cnn_vs_feature_pipeline.png`**
   - Diagram showing two pipelines: CNN (raw window → Conv → predict) vs Feature-based (window → FE → scale → predict)
   - Illustrates Finding 15 context

6. **`figures/tflite_deployment_strategies.png`**
   - Flowchart: Model type → Deployment path (Direct C++, TFLite ONNX, TFLite Surrogate)

7. **`figures/deployment_accuracy.png`** ← ALREADY IN TODO
   - Bar chart: accuracy before/after fixes (zero-padding, Kalman, double-scaling)

### EXISTING FIGURES (Already in TODO)

8. **`figures/system_architecture.png`**
9. **`figures/workflow_diagram.png`**
10. **`figures/ui_data_upload.png`**
11. **`figures/ui_preprocessing_draggable.png`**
12. **`figures/ui_results_display.png`**
13. **`figures/padding_comparison.png`**

---

## USER SUPPORT NEEDED — STRUCTURED LIST

### CATEGORY 1: DATA COLLECTION (CRITICAL PATH)

#### Task 1.1: Retrain Models with Fixed Pipeline
**What you need to do**:
1. Run Feature Engineering tab with:
   - Edge-value replication ✅ (already fixed)
   - Kalman filter enabled (Q=0.001, R=0.1)
   - Longer recordings if possible (≥1.5s per window)
   - Use sliding window feature to generate 50+ windows/class
2. Train all 4 models: NN, RF, SVM, CNN (if applicable)
3. Record training metrics for each

**Expected output**: New `.joblib` model files in `persistent_data/models/`

**Timeline**: 1-2 hours (depending on data collection)

---

#### Task 1.2: Collect On-Device Test Data
**What you need to do**:
1. Flash retrained models to Seeed XIAO (one at a time)
2. Perform each activity (running, still, walking, walking_downstairs, walking_upstairs) for ~30 seconds each
3. Use Device Test tab to record predictions + sensor data
4. Export CSV files for each model + activity combination

**Expected output**:
- CSV files: `device_test_{model}_{activity}.csv`
- Accuracy numbers for each model
- Inference time measurements (from Serial output)

**Timeline**: 2-3 hours (flash + test + export)

---

#### Task 1.3: (OPTIONAL) Cross-Device Benchmark
**What you need to do**:
1. If you have ESP32 or M5Stack available, flash one model to it
2. Repeat activity tests
3. Compare accuracy/inference time vs XIAO

**Expected output**: Cross-device benchmark table

**Timeline**: 1-2 hours (if hardware available)

---

### CATEGORY 2: UI SCREENSHOTS (MEDIUM PRIORITY)

#### Task 2.1: Preprocessing Tab Screenshots
**What you need to do**:
1. Open Preprocessing tab
2. Enable Kalman filter, set Q=0.001, R=0.1
3. Screenshot the settings panel
4. Screenshot a before/after signal plot (if available in UI)

**Expected output**: `figures/ui_kalman_settings.png`

**Timeline**: 10 minutes

---

#### Task 2.2: Data Upload Screenshot
**What you need to do**: Take screenshot of Data tab with dataset loaded

**Expected output**: `figures/ui_data_upload.png`

**Timeline**: 5 minutes

---

#### Task 2.3: Results Display Screenshot
**What you need to do**: Screenshot Training tab with model performance metrics displayed

**Expected output**: `figures/ui_results_display.png`

**Timeline**: 5 minutes

---

#### Task 2.4: Preprocessing Draggable Windows Screenshot
**What you need to do**: Screenshot Preprocessing tab showing draggable window selection

**Expected output**: `figures/ui_preprocessing_draggable.png`

**Timeline**: 5 minutes

---

### CATEGORY 3: PROGRAMMATIC FIGURE GENERATION (I CAN HELP)

#### Task 3.1: Confusion Matrix
**What I need from you**: Test set predictions CSV (after retraining)

**What I'll do**: Generate confusion matrix plot

**Expected output**: `figures/confusion_matrix_nn.png`

---

#### Task 3.2: Kalman Signal Comparison
**What I need from you**: Raw sensor CSV + Kalman-filtered CSV

**What I'll do**: 3-subplot comparison (raw, IIR, Kalman)

**Expected output**: `figures/kalman_signal_comparison.png`

---

#### Task 3.3: Architecture Diagrams
**What I'll do**: Create system architecture, pipeline flowcharts using LaTeX TikZ or Python matplotlib

**Expected output**:
- `figures/system_architecture.png`
- `figures/cnn_vs_feature_pipeline.png`
- `figures/tflite_deployment_strategies.png`

---

### CATEGORY 4: LATEX CONTENT (I WILL WRITE)

#### Task 4.1: Chapter 3 Kalman Section
**Status**: Ready to write (LaTeX content)

**Vietnamese text needed**: Yes

**Depends on**: Task 2.1 (Kalman screenshot), Task 3.2 (signal comparison plot)

---

#### Task 4.2: Chapter 4 Results Tables
**Status**: Waiting for data (Task 1.1, 1.2)

**Placeholders to fill**:
- Model accuracies (with/without Kalman)
- On-device accuracy + inference time
- TFLite surrogate agreement metrics

---

#### Task 4.3: Chapter 5 Discussion Updates
**Status**: Ready to write (analysis)

**Depends on**: Understanding Kalman impact from results

---

## SUMMARY: CRITICAL PATH

```
┌─────────────────────────────────────┐
│ CRITICAL: Data Collection (YOU)    │
│ ├─ Retrain models (1-2h)           │
│ ├─ Device tests (2-3h)             │
│ └─ Export CSVs                     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Figure Generation (ME + YOU)        │
│ ├─ Confusion matrix (I generate)   │
│ ├─ Kalman plots (I generate)       │
│ └─ UI screenshots (you capture)    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ LaTeX Writing (ME)                  │
│ ├─ Ch.3: Kalman section            │
│ ├─ Ch.4: Results with real data    │
│ ├─ Ch.5: Discussion updates        │
│ └─ Ch.6: Contributions update      │
└─────────────────────────────────────┘
```

**Estimated Total Time**:
- **Your work**: 4-6 hours (mostly data collection + device testing)
- **My work**: 6-8 hours (LaTeX writing + figure generation)
- **Total**: 10-14 hours

---

## NEXT STEPS — IMMEDIATE ACTIONS

1. **Review this plan** — Does it make sense? Any missing aspects?
2. **Prioritize user tasks** — Can you do Task 1.1 (retrain) soon? Task 1.2 (device test)?
3. **Identify blockers** — What do you need help with?
4. **Define timeline** — When can you provide the critical data (retraining + device tests)?

Once you confirm the plan and timeline, I'll:
1. Update `TECHNICAL_FINDINGS.md` with Findings 15-16
2. Start writing LaTeX content for chapters (using placeholders where data is needed)
3. Generate diagrams and figures I can create without data
4. Create a `TODO_FOR_USER.md` checklist you can follow

---

## DOCUMENT STATUS

- [x] Initial plan created
- [ ] User reviewed and approved
- [ ] User timeline confirmed
- [ ] LaTeX content written
- [ ] Figures generated
- [ ] Real data collected and integrated
- [ ] Final review and compilation check

