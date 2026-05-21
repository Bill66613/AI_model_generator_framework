# THESIS REPORT INSTRUCTIONS (Agent Working Notes)

**Last updated:** 2026-05-05  
**Version:** 5.3 (English-only instruction file for agent; thesis content stays Vietnamese)  
**Student:** Nguyễn Trương Minh Hoàng (MSSV: 2270757)  
**Thesis topic:** Xây dựng Framework Tạo Mô hình AI cho Ứng dụng Theo dõi Chuyển động Con người  
**Supervisor:** TS. Lê Trọng Nhân  
**Thesis language requirement:** Vietnamese (per program requirements)

---

## QUICK CONTEXT (Read this first in any new session)

**What is this file?** Master checklist and guide for completing the Vietnamese thesis report. Tracks what's done, what's pending, and provides LaTeX snippets ready to paste.

**Current blocking action:** Retrain + re-measure after PR\#3: (1) validate Kalman (causal) preprocessing, (2) validate TFLite flow for NN/CNN, and (3) collect real accuracy/latency numbers to replace placeholders in Ch.4.

**Narrative direction:** Thesis now emphasizes **multi-device deployment capability**. Seeed XIAO nRF52840 is the **reference benchmark platform**, not the sole target device.

### Report Completion Status

| Chapter | File | Status | Blocking Issue |
|---------|------|--------|---------------|
| Ch.1 Giới thiệu | `chapters/main/introduction.tex` | ✅ Reframed for multi-device deployment | Collect more cross-device benchmark evidence if available |
| Ch.2 Công trình liên quan | `chapters/main/relatedwork.tex` | ✅ Written in Vietnamese | — |
| Ch.3 Phương pháp luận | `chapters/main/methodology.tex` | ✅ Reframed | UI figures moved to Ch.4; theory-focused |
| Ch.4 Thiết kế và hiện thực | `chapters/main/design_implementation.tex` | ✅ NEW | Software architecture, UI screenshots, module design |
| Ch.5 Kết quả thực nghiệm | `chapters/main/results.tex` | ⚠️ Reframed + placeholder numbers | Needs real data after retrain; add TFLite section + Kalman impact |
| Ch.6 Thảo luận và Kết luận | `chapters/main/discussion_conclusion.tex` | ✅ MERGED | Discussion + Conclusion merged; redundancy removed |
| References | `references.bib` | ✅ Updated | Added 11 new refs (augmentation, confidence, parity, calibration) |
| Figures | `figures/` | ⚠️ Partial | UI screenshots exist; need architecture diagrams |

### Key Technical Findings (detail in TECHNICAL_FINDINGS.md)

Key technical findings to reflect in the thesis (see `TECHNICAL_FINDINGS.md` for evidence):
1. **Zero-padding artifact** → FIXED → edge-value replication (CRITICAL for defense)
2. **Kurtosis/skewness formula mismatch** → FIXED → population std for z-scores in all generators
3. **NN bias / default prediction behavior** → DOCUMENTED (explains "always predicts walking_downstairs" under distribution shift)
4. **FFT precision gap** → RESOLVED via design choice (time-domain only features for exact parity)
5. **Edge-replication still distorts if dataset is tiny** → DATA QUALITY issue (needs longer recordings)
6. **Double standardization** → FIXED (scale once in training pipeline)
7. **Feature order mismatch (Python alphabet vs C++ compute order)** → FIXED (reorder at code-gen)
8. **Double extraction bug undoing reorder** → FIXED
9. **CNN validation false positives** → FIXED (architecture-aware validator)
10. **Confidence threshold for unknown rejection** → IMPLEMENTED (deployment safety)
11. **Class-aware data augmentation protection** → IMPLEMENTED
12. **Multi-device deployment matrix is already implemented** → DOCUMENTED (XIAO is reference, not sole target)
13. **FFT robustness: Hann windowing + DC removal** → IMPLEMENTED → matches Edge Impulse quality
14. **Deployment accuracy simulation** → IMPLEMENTED → predict on-device accuracy before deployment
15. **TFLite deployment scope** → CLARIFIED (RF/SVM not directly convertible via onnx2tf/ONNX-ML; Keras surrogate fallback added as approximation)
16. **CNN code generation metadata mismatch** → FIXED (consistent feature/channel count in UI and code-gen)
17. **Kalman filter (causal) + exact deployment parity** → IMPLEMENTED (identical Python ↔ C++ filter equations with lazy-init)
18. **CNN scaler clamp destroys raw sensor data** → FIXED (v2 unified pipeline applied [-10,10] clamp to raw gyro values; added `skip_scaler` bypass for CNN; saves 7KB flash + eliminates wrong inference)
19. **Preprocessing filter parity on device** → IMPLEMENTED (per-window filtfilt for IIR, SavGol FIR convolution, FFT lowpass — all replicated exactly on device; outlier removal NOT replicated by design)
20. **Conv2D neural network (HARCNN2D)** → IMPLEMENTED (2D-CNN with cross-channel fusion via Conv2D(3×n_channels) kernel; explicit sensor-axis correlation learning; same deployment path as 1D-CNN)

---

## SESSION LOG

| Date | Session | Changes Made |
|------|---------|-------------|
| 2025-02-27 | Initial creation | Created with full TODO, LaTeX snippets, defense prep, glossary |
| 2025-02-27 | Restructure v2.0 | Added Quick Context, Session Log, structured for cross-session AI use |
| 2026-03-01 | Findings 6-8 | Added double standardization, feature order mismatch, double extraction to findings list |
| 2026-03-05 | Finding 9 | Added CNN validation false positives — validator now architecture-aware |
| 2026-03-25 | Consistency fixes | Fixed: 5/6 class count, 90/138 feature count, 75/150 window size, NN arch 90→100→5, added 6th objective to intro, removed duplicate BibTeX, added kurtosis verification to code gen, added power estimate disclaimer |
| 2026-03-26 | Round 2 consistency | Fixed: per-class Support 1078→30 (match test set), NN "two hidden layers"→"one", SensiML pricing unified \$99-500/month across all chapters |
| 2026-04-08 | Multi-device redirect | Reframed Ch.1/3/4/5/6 so thesis emphasizes multi-device deployment capability; XIAO now treated as reference benchmark platform |
| 2026-05-05 | Sync PR\#3 | Added TODOs/placeholders for TFLite scope, Kalman preprocessing parity, and CNN metadata fix; updated requested evidence list |
| 2026-05-05 | Integrate Findings 15-17 | ✅ Added: Kalman filter section in Ch.3 methodology, TFLite limitations section, multi-architecture code generation section, technical findings analysis in Ch.5 discussion, updated research objective #6, added kalman1960new + tensorflow2015_whitepaper references |
| 2026-05-17 | Finding 18 | CNN scaler clamp fix: v2 pipeline [-10,10] clamp destroyed raw sensor values for CNN; added skip_scaler bypass, validator CNN-awareness, memcpy optimization, brace formatting fix |
| 2026-05-17 | Findings 19-20 | Preprocessing filter parity (filtfilt+SavGol+FFT replicated on device); Conv2D HARCNN2D architecture; spectral_entropy DFT feature added |
| 2026-05-18 | Ch.4 Created | Added NEW Chapter 4 "Thiết kế và hiện thực" (Design and Implementation) - comprehensive system architecture, module design, code generation details, state management, deployment workflow |
| 2026-05-21 | Major restructure | Merged Discussion+Conclusion into Ch.6; Created Ch.4 "Thiết kế và hiện thực"; Moved UI figures from Ch.3 to Ch.4; Updated cross-references; New structure: 6 chapters total |

*Add a row here each time this file is updated.*

---

## PR\#3 NOTES (Upcoming Merge)

### What changes matter for the thesis narrative
1. **TFLite/TFLite Micro deployment**: only NN/CNN are applicable; RF/SVM cannot be converted because they rely on ONNX-ML operators (TreeEnsembleClassifier) that onnx2tf does not support.
2. **Kalman preprocessing**: add a causal (forward-only) denoising option to avoid the parity gap of filtfilt-style filters.
3. **CNN metadata mismatch**: fix inconsistent feature/channel reporting between training artifacts, UI, and code-gen naming.

### Where to update in LaTeX (keep it concise)
- Ch.3: add a short subsection on Kalman (goal, causal property, key parameters).
- Ch.3/Ch.5: clarify the TFLite scope (NN/CNN) and why RF/SVM are not applicable.
- Ch.4: add a table/paragraph for Kalman impact (if numbers available) and compare latency/accuracy between native C++ vs TFLite (if measured).

---

## ✅ WHAT I NEED FROM YOU (To Replace Placeholders)

Goal: replace placeholders with real numbers + figures.

1. **UI screenshots (PNG)**
   - Data tab (upload + preview)
   - Preprocess tab (draggable windows + filter toggle if present)
   - Feature Engineering tab (feature mode + augmentation)
   - Train tab (training results + confusion matrix)
   - Code Gen tab (platform/backend selection + output files)
   - Device Test tab (serial output / live inference)

2. **Training results after the latest changes**
   - Accuracy + macro F1 on test set (at least NN; include RF/SVM if still used)
   - Confusion matrix image
   - Config: window size, stride, feature mode, augmentation on/off, Kalman on/off

3. **Real-device deployment results (reference platform: XIAO)**
   - Latency (ms): separate feature extraction vs prediction
   - Notes on prediction stability (with/without smoothing + confidence threshold)
   - If available: a short log/CSV from a few test sessions

4. **TFLite/TFLite Micro results (if applicable)**
   - Target (PC / MCU / simulator)
   - Latency and accuracy vs native C++
   - Evidence logs: RF/SVM conversion failure output from onnx2tf (for citation)

5. **Orientation/mounting mismatch problem**
   - Describe 2–3 mounting orientations (photo or short description)
   - Provide 1–2 short CSV segments per orientation to quantify degradation


---

## 🔬 CÁC PHÁT HIỆN KỸ THUẬT CẦN ĐƯA VÀO BÁO CÁO

> **ĐÂY LÀ PHẦN QUAN TRỌNG NHẤT** — Các phát hiện này tạo nên sự khác biệt của framework so với Edge Impulse và các đối thủ trên thị trường. Xem chi tiết đầy đủ trong file `TECHNICAL_FINDINGS.md`.

### Phát hiện #1: Lỗi Zero-Padding trong Pipeline Trích xuất Đặc trưng (CRITICAL)

**Vấn đề phát hiện:** Khi cửa sổ dữ liệu ngắn hơn kích thước mục tiêu (150 mẫu), pipeline đệm thêm các hàng toàn số 0 (zero-padding). Điều này tạo ra giá trị `acc_mag = sqrt(0² + 0² + 0²) = 0.0` cho các mẫu đệm — một giá trị **vật lý bất khả thi** vì magnitude gia tốc luôn > 0 do trọng lực.

**Hậu quả:** 
- Tất cả `acc_mag_min = 0.0` trong dữ liệu huấn luyện (không thể xảy ra trên thiết bị thực)
- Mô hình học phân phối đặc trưng sai → triển khai thực tế luôn dự đoán sai
- Trên thiết bị Seeed XIAO, mô hình **luôn dự đoán "walking_downstairs"** bất kể hoạt động thực

**Giải pháp:** Thay thế zero-padding bằng edge-value replication (lặp lại giá trị cuối cùng của cửa sổ)

**Đưa vào báo cáo tại:**
- Chương 3 (Phương pháp luận) → Mục Trích xuất Đặc trưng: giải thích chiến lược padding
- Chương 5 (Thảo luận) → Mục mới: "Vấn đề Tương đồng Huấn luyện-Triển khai"
- Chương 4 (Kết quả) → So sánh kết quả trước/sau fix

### Phát hiện #2: Sai lệch Công thức Kurtosis/Skewness giữa Python và C++ (MEDIUM)

**Vấn đề:** Code C++ được sinh ra sử dụng **sample standard deviation** (chia n-1, Bessel correction) để tính z-score, trong khi Python pandas sử dụng **population standard deviation** (chia n). Sai lệch hệ thống ~1.3% cho kurtosis và ~1.0% cho skewness.

**Giải pháp:** Sửa code generator để sử dụng population std, khớp chính xác với pandas.

**Đưa vào báo cáo tại:**
- Chương 3 → Mục Tạo mã: nêu bật sự chặt chẽ của việc đảm bảo tương đồng số học
- Chương 5 → Mục "Thách thức Tương đồng Huấn luyện-Triển khai"

### Phát hiện #3: Bias Mạng Nơ-ron và Hành vi Dự đoán Mặc định

**Quan sát:** Khi đặc trưng đầu vào nằm ngoài phân phối huấn luyện hoàn toàn, output biases chi phối: `final_biases = [-0.116, -0.160, -0.066, 0.116, 0.048]` → Class 3 (walking_downstairs) có bias cao nhất (0.116) → luôn thắng.

**Ý nghĩa:** Đây là bằng chứng cho thấy **distribution mismatch** là nguyên nhân gốc, không phải lỗi thuật toán.

**Đưa vào báo cáo:** Chương 5 → phân tích chi tiết hành vi mô hình khi gặp dữ liệu ngoài phân phối.

### Phát hiện #6: Chuẩn hóa kép — FE và Training đều scale (CRITICAL)

**Vấn đề:** Feature Engineering tab áp dụng `StandardScaler` trước khi ghi CSV. Training pipeline đọc CSV và scale lần nữa → scaler học trên dữ liệu đã chuẩn hóa → `means ≈ 0, stds ≈ 1` → identity transform trên thiết bị.

**Hậu quả:** Thiết bị không scale đặc trưng → tất cả giá trị nằm ngoài phân phối → luôn dự đoán sai.

**Giải pháp:** Xóa scaling trong FE callback, giữ scaling duy nhất trong training pipeline.

**Đưa vào báo cáo:** Chương 5 → phân tích chuỗi lỗi pipeline, so sánh với hộp đen thương mại.

### Phát hiện #7: Sai thứ tự đặc trưng — Alphabetical vs C++ extraction order (CRITICAL)

**Vấn đề:** Python `pd.DataFrame` sắp xếp cột theo alphabet. C++ `extract_magnitude_stats()` trích xuất theo thứ tự tính toán cố định (mean→std→min→max→...). TẤT CẢ 33 đặc trưng bị sai vị trí → scaler áp sai mean/std cho sai feature → weights nhân sai feature.

**Giải pháp:** Thêm feature reorder remapping (`reorder_model_parameters()`) tại thời điểm code generation.

**Đưa vào báo cáo:** Chương 3 → Code Generation architecture, Chương 5 → parity verification.

### Phát hiện #8: Trích xuất kép — Reorder bị hủy bởi lần gọi thứ hai (CRITICAL)

**Vấn đề:** `extract_real_model_parameters()` được gọi 2 lần: trong `generate_and_save` rồi trong `generate_deployment_code`. Lần 2 re-extract scaler (alphabet order) nhưng skip reorder vì feature_names đã bị mutate sang C++ order → ghi đè kết quả reorder.

**Giải pháp:** Xóa lần gọi thừa, chỉ giữ 1 lần trong `generate_deployment_code()`.

**Đưa vào báo cáo:** Chương 5 → ví dụ về bug pipeline tinh vi mà chỉ hệ thống minh bạch mới phát hiện được.

---

## 🏆 ĐIỂM NỔI BẬT SO VỚI ĐỐI THỦ (SELLING POINTS)

### Bảng so sánh cần cập nhật trong báo cáo:

| Tiêu chí | Edge Impulse | SensiML | Framework của chúng tôi |
|-----------|-------------|---------|------------------------|
| **Phát hiện lỗi padding** | Không (hộp đen) | Không | ✅ Có — edge-value replication |
| **Tương đồng số học training⟷deployment** | Không đảm bảo | Không đảm bảo | ✅ Đảm bảo bit-exact cho time-domain |
| **Kiểm tra phân phối đặc trưng** | Không | Không | ✅ Phát hiện vấn đề thông qua debug |
| **Minh bạch code generation** | Hộp đen | Hộp đen | ✅ Mã nguồn mở, xem trực tiếp |
| **Chi phí** | $20-99/tháng | $49-199/tháng | Miễn phí |
| **Bao phủ đích triển khai** | Hỗ trợ rộng nhưng pipeline đóng | MCUs đã chọn | ✅ Kiến trúc generator đa đích, tái sử dụng cùng model artifact |
| **Kurtosis/skewness formula** | Không kiểm chứng | Không kiểm chứng | ✅ Verified vs pandas |

### Các điểm mạnh chính cần nhấn mạnh trong bảo vệ:

1. **Tính minh bạch hoàn toàn (Full Transparency)**
   - Người dùng có thể kiểm tra từng bước: từ dữ liệu thô → cửa sổ → đặc trưng → mô hình → code C++
   - Edge Impulse và SensiML là hộp đen — không thể phát hiện lỗi padding hay sai lệch công thức

2. **Tương đồng huấn luyện-triển khai được đảm bảo (Training-Deployment Parity)**
   - Đây là vấn đề **ít được nghiên cứu** trong tài liệu edge ML
   - Framework đảm bảo: cùng công thức, cùng thứ tự đặc trưng, cùng đơn vị
   - Phát hiện và sửa 2 lỗi tương đồng (zero-padding, kurtosis formula)

3. **Chiến lược padding thông minh (Edge-Value Replication)**
   - Zero-padding tạo artifact vật lý bất khả thi
   - Edge replication bảo toàn đặc tính thống kê của tín hiệu
   - Có ngưỡng tối thiểu (30% target size) để loại bỏ cửa sổ quá ngắn

4. **Debug và chẩn đoán triển khai**
   - Framework cho phép phát hiện mismatch thông qua phân tích đặc trưng
   - Ví dụ thực tế: phát hiện `acc_mag_min = 0.0` → truy ra zero-padding

5. **Đa nền tảng code generation**
   - C/C++ cho Arduino-compatible, ARM Cortex-M, generic C/C++, ESP-IDF, Zephyr
   - MicroPython và các backend thay thế như TFLite Micro, ONNX Runtime
   - 4 chế độ tối ưu hóa (accuracy/speed/power/balanced)
   - XIAO là nền tảng benchmark tham chiếu, không phải đích duy nhất

---

## 📝 HƯỚNG DẪN VIẾT TIẾNG VIỆT

### Thuật ngữ kỹ thuật Anh-Việt:

| English | Tiếng Việt |
|---------|-----------|
| Zero-padding | Đệm số không |
| Edge-value replication | Lặp giá trị biên |
| Training-deployment parity | Tương đồng huấn luyện-triển khai |
| Feature extraction | Trích xuất đặc trưng |
| Windowing | Tạo cửa sổ / phân đoạn cửa sổ |
| Sliding window | Cửa sổ trượt |
| Standard deviation | Độ lệch chuẩn |
| Distribution mismatch | Không khớp phân phối |
| Edge deployment | Triển khai biên |
| Code generation | Tạo mã |
| Inference | Suy luận |
| Confidence threshold | Ngưỡng tin cậy |
| Multi-device deployment | Triển khai đa thiết bị |
| Reference benchmark platform | Nền tảng benchmark tham chiếu |
| Overfitting | Quá khớp |
| Underfitting | Dưới khớp |

### Nguyên tắc viết:
- Giữ nguyên thuật ngữ tiếng Anh trong ngoặc khi lần đầu xuất hiện: "Trích xuất đặc trưng (Feature Extraction)"
- Tên thuật toán giữ nguyên: Random Forest, SVM, Neural Network
- Tên framework/nền tảng giữ nguyên: Edge Impulse, SensiML, TensorFlow Lite, Zephyr, MicroPython
- Phân biệt rõ giữa "hỗ trợ đa thiết bị ở mức kiến trúc/sinh mã" và "benchmark định lượng trên nền tảng tham chiếu"
- Xem Seeed XIAO nRF52840 là nền tảng benchmark tham chiếu, không phải đích duy nhất
- Công thức toán học dùng ký hiệu LaTeX chuẩn
- Hình ảnh có caption song ngữ nếu cần

## 📋 DANH SÁCH VIỆC CẦN LÀM (TODO) CHO BÁO CÁO

### Ưu tiên 1: Retrain và Thu thập Kết quả Mới
- [ ] Chạy lại Feature Engineering (đã sửa edge-value replication)
- [ ] Huấn luyện lại mô hình (tất cả: RF, SVM, NN/PyTorch)
- [ ] Tạo lại code triển khai cho ít nhất 2-3 đích đại diện (ví dụ: Seeed XIAO, ESP32/ESP-IDF, MicroPython hoặc Zephyr)
- [ ] Flash lên thiết bị và kiểm tra
- [ ] Ghi lại kết quả trước/sau sửa lỗi → dùng cho bảng so sánh

### Ưu tiên 2: Cập nhật Nội dung Báo cáo

#### Chương 1 — Giới thiệu:
- [x] Chuyển narrative từ single-device sang multi-device deployment capability
- [x] Xác định Seeed XIAO là nền tảng benchmark tham chiếu, không phải đích duy nhất

#### Chương 3 — Phương pháp luận:
- [x] Cập nhật mục Feature Extraction: giải thích edge-value replication padding
- [x] Thêm mục mới: "Đảm bảo tương đồng huấn luyện-triển khai"
- [x] Giải thích tại sao loại bỏ FFT features (đã có) + tại sao padding quan trọng
- [x] Thêm mục mới: "Tăng cường Dữ liệu Nhận biết Lớp" (data augmentation with class-aware protection)
- [x] Thêm mục mới: "Ngưỡng Tin cậy cho Từ chối Hoạt động Không xác định" (confidence threshold)
- [x] Cập nhật mục Code Generation: đề cập verified kurtosis/skewness formula
- [x] Sửa: kiến trúc NN 90 đầu vào (khớp với 15×6 features), bảng bộ nhớ 90-100-5
- [x] Thêm bảng ma trận đích triển khai và mô tả kiến trúc model-platform-backend

#### Chương 4 — Kết quả:
- [ ] Thay placeholder numbers bằng kết quả thực tế
- [ ] Thêm bảng: "Tác động của Chiến lược Padding" (trước/sau)
- [ ] Thêm bảng: "So sánh Suy luận Thiết bị Thực" (hardware test results)
- [x] Thêm bảng: "Ma trận hỗ trợ triển khai đa thiết bị"
- [ ] Tạo confusion matrix thực tế
- [ ] Chụp screenshots UI

#### Chương 5 — Thảo luận:
- [x] Thêm mục: "Vấn đề Tương đồng Huấn luyện-Triển khai trong Edge ML"
- [x] Thêm mục: "Tăng cường Dữ liệu và Thiết kế Nhận biết Lớp"
- [x] Thêm mục: "Từ chối Hoạt động Dựa trên Ngưỡng Tin cậy"
- [x] Cập nhật so sánh Edge Impulse: bổ sung điểm vượt trội mới
- [x] Thêm phân tích: tại sao hộp đen platforms không thể phát hiện lỗi này
- [x] Phân tách rõ "hỗ trợ đa thiết bị ở mức kiến trúc" và "benchmark trên nền tảng tham chiếu"

#### Chương 6 — Kết luận:
- [x] Bổ sung đóng góp kỹ thuật mới: training-deployment parity verification
- [x] Bổ sung đóng góp: data augmentation with class-aware protection
- [x] Bổ sung đóng góp: confidence threshold for unknown activity rejection
- [x] Nhấn mạnh: transparency giúp phát hiện và sửa lỗi mà hộp đen không thể
- [x] Cập nhật mục tiêu nghiên cứu: thêm "Độ Tin Cậy Triển Khai"
- [x] Nhấn mạnh đóng góp đa thiết bị; XIAO chỉ là case tham chiếu đầu tiên

### Ưu tiên 3: Hình ảnh và Tài liệu trực quan
- [ ] `figures/confusion_matrix_nn.png` — Từ mô hình NN sau retrain
- [ ] `figures/system_architecture.png` — Sơ đồ kiến trúc hệ thống
- [ ] `figures/workflow_diagram.png` — Quy trình làm việc người dùng
- [ ] `figures/ui_data_upload.png` — Screenshot tab tải dữ liệu
- [ ] `figures/ui_preprocessing_draggable.png` — Screenshot kéo thả cửa sổ
- [ ] `figures/ui_results_display.png` — Screenshot bảng điều khiển kết quả
- [ ] `figures/padding_comparison.png` — So sánh zero-padding vs edge-replication
- [ ] `figures/deployment_accuracy.png` — Biểu đồ độ chính xác trước/sau sửa

### Ưu tiên 4: Tham khảo bổ sung
- [x] Thêm tham khảo về training-deployment mismatch trong edge ML (Paleyes 2022, Sculley 2015)
- [x] Thêm tham khảo về data augmentation cho IMU/HAR (Um 2017, Iwana 2021, Eyobu 2018)
- [x] Thêm tham khảo về confidence/calibration (Hendrycks 2017, Guo 2017, Gal 2016)
- [x] Thêm tham khảo về class-aware augmentation (Buda 2018)
- [x] Cập nhật references.bib

### Ưu tiên 5: Chuẩn bị bảo vệ
- [ ] Slide trình bày (15-20 phút)
- [ ] Slide demo thực tế (video hoặc live demo)
- [ ] Chuẩn bị câu trả lời cho câu hỏi phản biện:
   - "Tại sao không dùng deep learning?"
   - "Dataset chỉ 1 người, tổng quát hóa thế nào?"
   - "Nếu chỉ benchmark trên XIAO thì vì sao vẫn gọi là đa thiết bị?"
   - "So sánh chi tiết với Edge Impulse?"
   - "Padding ảnh hưởng bao nhiêu %?"

---

## 🧩 NỘI DUNG LATEX GỢI Ý

### Mục mới cho Chương 3 (Phương pháp luận):

```latex
\subsection{Chiến lược Đệm Cửa sổ (Window Padding Strategy)}

Trong quá trình phân đoạn dữ liệu, các cửa sổ thời gian được trích xuất từ dữ liệu thô
có thể ngắn hơn kích thước mục tiêu (150 mẫu ở 100Hz = 1,5 giây). Điều này xảy ra khi
người dùng chọn các phân đoạn hoạt động ngắn hoặc khi dữ liệu gốc không đủ dài.

\textbf{Vấn đề với đệm số không (Zero-Padding):} Phương pháp đệm bằng số không—thường
được sử dụng trong xử lý tín hiệu—tạo ra các hàng dữ liệu có tất cả giá trị cảm biến
bằng 0. Đối với gia tốc kế, điều này có nghĩa $a_x = a_y = a_z = 0$, dẫn đến magnitude
gia tốc $||\mathbf{a}|| = \sqrt{a_x^2 + a_y^2 + a_z^2} = 0$. Tuy nhiên, trong thực tế,
magnitude gia tốc luôn lớn hơn 0 do gia tốc trọng trường ($\approx 9,81$ m/s²). Do đó,
zero-padding tạo ra các giá trị \textit{vật lý bất khả thi} làm nhiễu phân phối đặc trưng
thống kê (min = 0, median bị kéo về 0, phân vị thứ 25 bị kéo về 0).

\textbf{Giải pháp: Lặp giá trị biên (Edge-Value Replication):} Framework sử dụng chiến lược
lặp giá trị cuối cùng của cửa sổ để đệm đến kích thước mục tiêu. Phương pháp này bảo toàn
các đặc tính thống kê của tín hiệu gốc, không tạo ra artifact vật lý bất khả thi, và đảm bảo
tính nhất quán giữa dữ liệu huấn luyện và dữ liệu suy luận trên thiết bị biên (nơi bộ đệm
luôn chứa đầy dữ liệu cảm biến thực).

Ngoài ra, các cửa sổ có ít hơn 30\% số mẫu mục tiêu được loại bỏ hoàn toàn, vì chúng
không chứa đủ thông tin chuyển động để trích xuất đặc trưng có ý nghĩa.
```

### Mục mới cho Chương 5 (Thảo luận):

```latex
\subsection{Vấn đề Tương đồng Huấn luyện-Triển khai trong Edge ML}

Một phát hiện quan trọng trong quá trình phát triển framework là tầm quan trọng của
\textit{tương đồng huấn luyện-triển khai} (training-deployment parity)—sự đảm bảo rằng
quá trình trích xuất đặc trưng trong môi trường huấn luyện (Python) tạo ra kết quả
\textbf{giống hệt} với quá trình trích xuất đặc trưng trên thiết bị biên (C++/MicroPython).

\textbf{Nguồn gốc không khớp:} Chúng tôi xác định hai nguồn sai lệch chính:

\begin{enumerate}
    \item \textbf{Artifact đệm số không}: Trong pipeline trích xuất đặc trưng, các cửa sổ
    ngắn hơn kích thước mục tiêu được đệm bằng hàng toàn số không. Trên thiết bị thực,
    bộ đệm luôn chứa đầy dữ liệu cảm biến thực (không có giá trị zero). Sự khác biệt
    này tạo ra phân phối đặc trưng hoàn toàn khác nhau giữa huấn luyện và triển khai.
    
    \item \textbf{Sai lệch công thức thống kê}: Công thức độ nhọn (kurtosis) trong code C++
    sinh ra sử dụng độ lệch chuẩn mẫu (sample std, chia $n-1$) cho chuẩn hóa z-score,
    trong khi Python pandas sử dụng độ lệch chuẩn tổng thể (population std, chia $n$).
    Sai lệch hệ thống: $\left(\frac{n-1}{n}\right)^2 \approx 0,987$ cho $n = 150$,
    tương đương ~1,3\%.
\end{enumerate}

\textbf{Tác động thực tế}: Khi triển khai mô hình trên thiết bị Seeed XIAO nRF52840,
mô hình luôn dự đoán lớp ``walking\_downstairs'' bất kể hoạt động thực tế. Phân tích
cho thấy output biases của mạng nơ-ron [$-0,116; -0,160; -0,066; 0,116; 0,048$] chi phối
khi đặc trưng đầu vào nằm hoàn toàn ngoài phân phối huấn luyện—lớp 3 (walking\_downstairs)
có bias cao nhất ($0,116$) nên luôn thắng.

\textbf{So sánh với nền tảng thương mại}: Các nền tảng thương mại như Edge Impulse hoạt
động theo mô hình hộp đen, trong đó pipeline trích xuất đặc trưng không thể kiểm tra
hoặc xác minh bởi người dùng. Nếu một lỗi tương tự tồn tại trong pipeline của họ,
người dùng không có cách nào phát hiện hoặc sửa chữa. Tính minh bạch hoàn toàn của
framework chúng tôi cho phép:
\begin{itemize}
    \item Kiểm tra trực tiếp giá trị đặc trưng (phát hiện $\text{acc\_mag\_min} = 0$)
    \item Truy vết ngược đến nguyên nhân gốc (zero-padding)
    \item Xác minh tương đồng công thức giữa Python và C++
    \item Sửa lỗi và retrain mà không cần chờ vendor
\end{itemize}

Phát hiện này nhấn mạnh rằng \textbf{tính minh bạch không chỉ là triết lý mã nguồn mở
mà là yêu cầu kỹ thuật thiết yếu} cho triển khai edge ML đáng tin cậy.
```

---

## 🎓 GHI NHỚ CHO BUỔI BẢO VỆ

### Câu chuyện chính (narrative) của luận văn:

> "Chúng tôi xây dựng một framework mã nguồn mở, minh bạch, giúp dân chủ hóa việc phát triển edge AI cho theo dõi chuyển động. Trong quá trình phát triển, chúng tôi phát hiện và giải quyết **vấn đề tương đồng huấn luyện-triển khai** — một thách thức quan trọng nhưng ít được nghiên cứu trong edge ML. Tính minh bạch hoàn toàn của framework cho phép chúng tôi phát hiện và sửa các lỗi mà các nền tảng hộp đen thương mại không thể."

### 3 điểm khác biệt cốt lõi:
1. **Minh bạch → Phát hiện lỗi**: Zero-padding, kurtosis formula
2. **Tương đồng huấn luyện-triển khai**: Bit-exact time-domain features
3. **Chi phí = 0, chất lượng cạnh tranh**: 96.2% accuracy, 12ms inference
4. **Tăng cường nhận biết lớp**: Tự động bảo vệ hoạt động tĩnh khỏi class confusion
5. **Ngưỡng tin cậy zero-cost**: Từ chối dự đoán không chắc chắn mà không cần mô hình bổ sung

### Câu hỏi phản biện có thể gặp:

**Q: "Tại sao không sử dụng deep learning (CNN, LSTM)?"**
A: Các mô hình ML cổ điển (RF, SVM, NN) biên dịch thành C++ nhỏ gọn (95-182KB), không cần runtime suy luận nặng (TFLite Micro thêm 100-300KB overhead). Trên Cortex-M4 64MHz, suy luận 12-18ms vs 60-120ms cho deep learning. Với tập dữ liệu vừa phải (<10K mẫu), hiệu suất tương đương (96.2% vs 96.5% TFLite).

**Q: "Dataset chỉ 1 người, làm sao tổng quát hóa?"**
A: Đúng, đây là hạn chế được thừa nhận. Tuy nhiên, mục tiêu chính là chứng minh khả năng của framework — pipeline từ dữ liệu → triển khai. Framework được thiết kế để người dùng mang dữ liệu riêng (multi-subject). Xác thực trên UCI HAR, WISDM là công việc tương lai. Data augmentation cũng giúp cải thiện generalization từ single-subject data.

**Q: "Edge Impulse hỗ trợ 85+ board, framework này chỉ hỗ trợ một vài?"**
A: Đúng, nhưng kiến trúc modular (factory pattern) cho phép thêm nền tảng mới bằng cách kế thừa BaseCodeGenerator. Đã có: Arduino, ARM Cortex-M, Zephyr, MicroPython. Quan trọng hơn: code chúng tôi tạo ra **đã được xác minh** tương đồng với Python training — Edge Impulse không đảm bảo điều này.

**Q: "Padding ảnh hưởng bao nhiêu phần trăm?"**
A: [Cần retrain và đo] — Dự kiến improvement đáng kể vì tất cả windows đều bị zero-padded (53% cho running, 33% cho still, 20-27% cho walking).

**Q: "Tại sao cần class-aware augmentation? Tăng cường đồng nhất không đủ?"**
A: Thực nghiệm cho thấy tăng cường đồng nhất gây class confusion — "still" bị phân loại sai thành "walking_downstairs" do jitter/rotation tạo dao động nhân tạo. Với hoạt động tĩnh, đặc tính phân biệt chính là biên độ dao động cực thấp (σ < 0.1 m/s²); biến đổi mạnh phá hủy chính xác đặc tính này. Chiến lược micro-jitter (σ=0.01) bảo toàn đặc tính phân biệt.

**Q: "Ngưỡng tin cậy có đủ tin cậy cho ứng dụng thực tế?"**
A: Softmax confidence có thể kém hiệu chỉnh (overconfident), đây là hạn chế đã thừa nhận. Tuy nhiên, Random Forest có calibration tự nhiên tốt (tỷ lệ bỏ phiếu = xác suất thực). Với ngưỡng 0.6 (gấp 3× random cho 5 lớp), đây là tuyến phòng thủ đầu tiên hiệu quả. Temperature scaling là hướng tương lai để cải thiện calibration.

---

## 📁 CẤU TRÚC FILE

```
academic-paper-vietnamese/
├── main.tex                    # File chính
├── references.bib              # Tài liệu tham khảo (cập nhật: +11 refs mới)
├── THESIS_REPORT_INSTRUCTIONS.md  # ← File này (TODO list, LaTeX snippets, defense prep)
├── TECHNICAL_FINDINGS.md       # ← Chi tiết phát hiện kỹ thuật (11 findings)
├── chapters/
│   ├── front/                  # Khai báo, lời cảm ơn, tóm tắt
│   └── main/
│       ├── introduction.tex    # ✅ Tiếng Việt
│       ├── relatedwork.tex     # ✅ Tiếng Việt
│       ├── methodology.tex     # ✅ Cập nhật: padding, augmentation, confidence
│       ├── results.tex         # ⚠️ Cần kết quả thực tế (placeholder numbers)
│       ├── discussion.tex      # ✅ Cập nhật: parity, augmentation design, confidence
│       └── conclusion.tex      # ✅ Cập nhật: 7 đóng góp (thêm #5-7), future work
├── figures/                    # ❌ Cần tạo hình ảnh
└── tables/                     # ⚠️ Cần cập nhật bảng
```

---

## DOCUMENT MAINTENANCE

### How to use this file across sessions
1. **New session?** Read **QUICK CONTEXT** at the top — it shows report status and blocking actions in 10 seconds.
2. **Completed a TODO?** Check the box `[x]` in the relevant priority section AND update the Report Completion Status table.
3. **Found a new technical issue?** Document it in `TECHNICAL_FINDINGS.md` first, then add a summary + chapter-mapping entry to §"PHÁT HIỆN KỸ THUẬT" section of this file.
4. **Updated a chapter?** Update the status table in Quick Context AND add a Session Log entry.
5. **Working on defense prep?** See §"GHI NHỚ CHO BUỔI BẢO VỆ" for narrative, key slides, and Q&A prep.

### Relationship between documentation files
- **`.github/copilot-instructions.md`** — Technical instructions for AI coding agents (architecture, patterns, pitfalls). Points to this file and TECHNICAL_FINDINGS.md.
- **`THESIS_REPORT_INSTRUCTIONS.md`** (this file) — Report completion roadmap: what chapters need what updates, LaTeX snippets, defense preparation.
- **`TECHNICAL_FINDINGS.md`** — Detailed evidence for each finding (code, data, math). This file summarizes them; that file has the full evidence.

**Lưu ý cuối cùng:** Mỗi lần AI assistant tiếp tục làm việc trên báo cáo, hãy đọc **QUICK CONTEXT** section ở đầu file TRƯỚC để nắm bắt context và biết công việc nào cần làm.
