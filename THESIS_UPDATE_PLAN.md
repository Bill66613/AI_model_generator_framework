# KẾ HOẠCH CẬP NHẬT LUẬN VĂN - THESIS UPDATE PLAN

**Ngày tạo:** 2026-05-05  
**Phiên bản:** 1.0  
**Sinh viên:** Nguyễn Trường Minh Hoàng (MSSV: 2270757)  
**Đề tài:** Xây dựng Framework Tạo Mô hình AI cho Ứng dụng Theo dõi Chuyển động Con người

---

## 📊 TÌNH TRẠNG HIỆN TẠI

### Framework Code Status
- ✅ **11+ Code Generators** hoàn thiện (RF, SVM, NN, CNN, ARM Cortex-M, MicroPython, Zephyr, etc.)
- ✅ **14 Technical Findings** đã được ghi nhận và sửa chữa
- ✅ **6-tab End-to-End Workflow** từ dữ liệu thô đến triển khai thiết bị
- ✅ **Multi-device Deployment Matrix** hỗ trợ 8+ vi xử lý
- ✅ **Production-ready** với ~15,000+ dòng code Python

### Thesis Status Gap Analysis
| Vietnamese Thesis (Official) | English Draft | Framework Reality |
|-----|-----|-----|
| ⚠️ Cần cập nhật từ v5.1 | ✅ Complete (~15,500 words) | ✅ Production-ready |
| ❌ Thiếu 14 technical findings | ✅ Has comprehensive results | ✅ Has all 14 findings documented |
| ❌ Thiếu figures/screenshots | ❌ Placeholder figures | ✅ Framework can generate figures |
| ⚠️ Placeholder numbers | ❌ Needs real data | ✅ Can collect real experimental data |

---

## 🎯 MỤC TIÊU CẬP NHẬT

### Primary Goals
1. **Sync Vietnamese thesis with English draft quality**
2. **Incorporate all 14 technical findings into thesis chapters**
3. **Generate missing figures, screenshots, and experimental data**
4. **Emphasize multi-device deployment capability (not just XIAO)**
5. **Create reproducible experimental results section**

### Success Metrics
- Vietnamese thesis quality matches English draft (95% complete → 100%)
- All placeholder values replaced with real experimental data
- All figures generated and integrated
- Technical findings properly positioned in relevant chapters
- Thesis ready for defense with concrete evidence

---

## 📝 DANH SÁCH CẬP NHẬT CHI TIẾT

### A. CHƯƠNG TRÌNH CẬP NHẬT CÁC CHƯƠNG

#### Chapter 1: Giới thiệu (Introduction) 
**Status:** ✅ Reframed for multi-device deployment

**Required Updates:**
- [ ] Verify 6 research objectives reflect framework reality
- [ ] Update problem statement with specific competitive advantages vs Edge Impulse/SensiML
- [ ] Add concrete statistics (11 generators, 156 features, 8+ MCUs supported)

#### Chapter 3: Phương pháp luận (Methodology)
**Status:** ✅ Reframed + multi-device generator matrix

**Required Updates:**  
- [ ] Add detailed 14 technical findings integration
- [ ] Update feature extraction section with 6 modes (33-156 features)
- [ ] Add training-deployment parity validation methodology
- [ ] Include code generation factory pattern explanation

#### Chapter 4: Kết quả (Results)
**Status:** ⚠️ Reframed + placeholder numbers

**Critical Missing Data:**
- [ ] Real accuracy numbers (placeholder: RF 94.5%, NN 96.2%, SVM 93.8%)
- [ ] Real inference timing (placeholder: 12-18ms on ARM Cortex-M4)
- [ ] Real memory footprint (placeholder: 95-182KB flash)
- [ ] Real power consumption (placeholder: 30mW average)
- [ ] Confusion matrix from trained models
- [ ] Before/after accuracy comparison for zero-padding fix

#### Chapter 5: Thảo luận (Discussion) 
**Status:** ✅ Reframed

**Required Updates:**
- [ ] Add section "Training-Deployment Parity Analysis" incorporating Finding 1,2,6,7,8
- [ ] Add section "Multi-Algorithm Code Generation" showcasing 11 generators
- [ ] Add competitive analysis vs Edge Impulse with concrete numbers

#### Chapter 6: Kết luận (Conclusion)
**Status:** ✅ Reframed

**Required Updates:**
- [ ] Update contributions to emphasize 14 technical findings
- [ ] Add future work based on framework's extensible architecture

### B. FIGURES VÀ VISUALIZATIONS CẦN TẠO

#### High Priority Figures
1. **System Architecture Diagram**
   - 6-tab workflow: Data → Preprocess → Feature Engineering → Training → Code Gen → Device Test
   - Multi-platform deployment matrix
   - File: `academic-paper-vietnamese/figures/system_architecture.png`

2. **Confusion Matrix** 
   - From actual trained Neural Network model (96.2% accuracy)
   - 6x6 matrix for HAR activities
   - File: `academic-paper-vietnamese/figures/confusion_matrix_nn.png`

3. **Training-Deployment Parity Flowchart**
   - Shows Python feature extraction → C++ code generation → validation
   - Highlights Finding 1,2,6,7,8 fixes
   - File: `academic-paper-vietnamese/figures/parity_validation.png`

4. **Multi-Device Deployment Matrix**
   - Table/chart showing 11 generators × 8+ MCUs
   - Memory/performance characteristics per platform
   - File: `academic-paper-vietnamese/figures/deployment_matrix.png`

#### Medium Priority Figures  
5. **Feature Extraction Modes Comparison**
   - Bar chart: 6 modes with feature counts (33, 53, 90, 156, 66, 6)
   - File: `academic-paper-vietnamese/figures/feature_modes.png`

6. **Power Consumption Profile**
   - Bar chart: Idle, Sampling, Feature Extraction, Prediction phases
   - File: `academic-paper-vietnamese/figures/power_profile.png`

7. **Dash UI Screenshots**
   - Screenshots of all 6 tabs showing interactive preprocessing
   - File: `academic-paper-vietnamese/figures/ui_screenshots/`

### C. EXPERIMENTAL DATA CẦN THU THẬP

#### Critical Real Data Needed

1. **Model Training Performance**
   ```
   REQUIRED: Run full training pipeline and collect:
   - Actual accuracy: RF, SVM, NN for 6-class HAR
   - Training time: seconds per algorithm  
   - Model size: KB after serialization
   - Per-class precision/recall/F1 metrics
   ```

2. **Edge Deployment Characteristics**
   ```
   REQUIRED: Deploy on Seeed XIAO and measure:
   - Inference latency: ms per prediction
   - Memory footprint: Flash KB, RAM KB
   - Power consumption: mW (idle, active, prediction)
   - Battery life estimation: hours
   ```

3. **Training-Deployment Parity Validation**
   ```
   REQUIRED: Before/after accuracy for zero-padding fix:
   - Accuracy with zero-padding: XX.X%
   - Accuracy with edge-replication: YY.Y%  
   - Feature distribution analysis
   ```

4. **Cross-Platform Deployment**
   ```
   NICE-TO-HAVE: Deploy same model on:
   - Seeed XIAO nRF52840 (primary)
   - ESP32 (comparison platform)
   - Memory/timing comparison data
   ```

---

## 🤝 USER INPUT REQUIREMENTS

### A. SCREENSHOTS DẦN THU THẬP (High Priority)

**Tab 1: Data Management**
- [ ] CSV upload interface with dataset list
- [ ] Data visualization plots (time series)
- [ ] Label assignment interface

**Tab 2: Signal Preprocessing** 
- [ ] Raw sensor data plots
- [ ] Butterworth filter settings
- [ ] **DRAGGABLE WINDOWING** interface (UNIQUE FEATURE vs Edge Impulse)

**Tab 3: Feature Engineering**
- [ ] Feature extraction mode selection (6 modes)
- [ ] Feature statistics display  
- [ ] Train/val/test split configuration

**Tab 4: Model Training**
- [ ] Algorithm selection (RF, SVM, NN, PyTorch)
- [ ] Hyperparameter grid interface
- [ ] Training progress and results

**Tab 5: Code Generation**
- [ ] Platform selection (11 generators)
- [ ] Optimization mode selection
- [ ] Generated code preview

**Tab 6: Device Testing**
- [ ] Serial monitor with real-time predictions
- [ ] Confusion matrix display
- [ ] Performance metrics

### B. DEVICE DEPLOYMENT RESULTS (High Priority)

**Real Hardware Testing on Seeed XIAO:**
- [ ] Video/photos of device in operation
- [ ] Serial output showing predictions
- [ ] Current consumption measurements (multimeter/power profiler)
- [ ] Comparison of Python vs C++ predictions on same test data

### C. TRAINING RESULTS (Critical)

**Complete Training Session Data:**
- [ ] Run training on HAR dataset with all 3 algorithms
- [ ] Collect confusion matrices, accuracy metrics
- [ ] Save model files (.joblib) for analysis
- [ ] Compare before/after accuracy for technical fixes

### D. COMPARATIVE ANALYSIS DATA (Medium Priority)

**Framework vs Edge Impulse:**
- [ ] Same dataset trained on both platforms
- [ ] Accuracy comparison
- [ ] Model size comparison
- [ ] Deployment code complexity comparison

---

## 📅 TIMELINE VÀ PRIORITIES

### Week 1: Critical Experimental Data
- **Days 1-2:** Run complete training pipeline, collect real accuracy/timing data
- **Days 3-4:** Deploy on Seeed XIAO, measure inference performance
- **Days 5-7:** Take comprehensive UI screenshots of all 6 tabs

### Week 2: Figure Generation & Chapter Updates
- **Days 1-3:** Generate all required figures using collected data
- **Days 4-5:** Update Vietnamese thesis chapters with real numbers
- **Days 6-7:** Integrate technical findings into relevant chapters

### Week 3: Polish & Validation
- **Days 1-3:** Complete thesis compilation and proofreading
- **Days 4-5:** Cross-validation with English draft quality
- **Days 6-7:** Final PDF generation and review

---

## 📋 IMMEDIATE ACTION CHECKLIST

### For User (You) To Do This Week:
- [ ] **Run Training Pipeline:** Execute full HAR training with RF/SVM/NN, record all metrics
- [ ] **Deploy and Test:** Flash generated code to Seeed XIAO, collect performance data
- [ ] **Take Screenshots:** Capture UI of all 6 tabs during a complete workflow
- [ ] **Power Measurements:** Use multimeter/power profiler to measure device consumption
- [ ] **Comparative Test:** If possible, run same dataset on Edge Impulse for comparison

### For AI Assistant (Me) To Do:
- [ ] **Generate Figures:** Create system architecture, deployment matrix, feature comparison charts
- [ ] **Update Vietnamese Chapters:** Incorporate technical findings and real experimental data
- [ ] **LaTeX Integration:** Ensure all figures compile properly with Vietnamese thesis
- [ ] **Technical Validation:** Cross-check framework claims against actual codebase
- [ ] **Defense Preparation:** Identify strongest technical contributions for thesis defense

---

## 🎯 SUCCESS CRITERIA

### Minimum Viable Update (Must Have)
1. ✅ Real experimental data replaces all placeholder numbers
2. ✅ UI screenshots show complete 6-tab workflow  
3. ✅ Technical findings properly integrated into chapters
4. ✅ Vietnamese thesis compiles to PDF without errors
5. ✅ Framework capabilities accurately represented

### Excellent Update (Should Have)
6. ✅ Cross-platform deployment comparison (XIAO + ESP32)
7. ✅ Competitive analysis vs Edge Impulse with real data
8. ✅ Power consumption measurements and battery life estimates
9. ✅ All 11 code generators showcased in deployment matrix
10. ✅ Video demonstration of real-time HAR on device

### Outstanding Update (Nice to Have)
11. ✅ Multi-subject validation beyond single-person dataset
12. ✅ Additional activity classes beyond basic 6 activities
13. ✅ Extended inference accuracy analysis (confidence thresholds, smoothing)
14. ✅ Published demo video and reproducible setup guide

---

## 📞 SUPPORT REQUESTS

### Technical Assistance Needed:
- **LaTeX Compilation:** Help with Vietnamese character encoding, figure integration
- **Python Code:** Automated figure generation from trained models
- **Data Analysis:** Statistical comparison of before/after accuracy improvements

### Content Review Needed:
- **Technical Accuracy:** Verify framework descriptions match actual implementation
- **Academic Writing:** Ensure Vietnamese technical terminology is consistent
- **Defense Strategy:** Identify strongest points for thesis defense

---

**Status:** Plan created - Ready for execution  
**Next Action:** User to collect experimental data and screenshots  
**Timeline:** 3 weeks to completion  
**Priority:** Update Vietnamese thesis to match framework reality and English draft quality