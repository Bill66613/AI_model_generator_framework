# PHÁT HIỆN KỸ THUẬT QUAN TRỌNG
# Technical Findings — Framework vs Commercial Platforms

**Last updated:** 2026-03-05  
**Version:** 5.0 (added finding 10: Confidence threshold for unknown activity rejection)

---

## QUICK CONTEXT (Read this first in any new session)

This file documents **10 critical technical findings** discovered during framework development and device testing. These findings are the **core differentiators** of the thesis vs commercial platforms (Edge Impulse, SensiML) and form the strongest defense arguments.

| # | Finding | Severity | Status | Code Files Affected | Thesis Chapters |
|---|---------|----------|--------|---------------------|-----------------|
| 1 | Zero-padding artifact in FE pipeline | CRITICAL | ✅ FIXED | `callbacks/feature_engineering_callbacks.py` | Ch.3 §FE, Ch.4 §results, Ch.5 §parity |
| 2 | Kurtosis/skewness sample-vs-population std | MEDIUM | ✅ FIXED | `deployment/base_generator.py`, `deployment/micropython_generator.py` | Ch.3 §CodeGen, Ch.5 §parity |
| 3 | NN bias → default prediction behavior | INFO | DOCUMENTED | (no code change needed) | Ch.5 §analysis |
| 4 | FFT precision mismatch → time-only features | DESIGN | RESOLVED | `utils/feature_extraction.py` | Ch.3 §FE rationale |
| 5 | Edge-replication padding still distorts features + tiny dataset | CRITICAL | ⚠️ DATA QUALITY | `callbacks/feature_engineering_callbacks.py`, `callbacks/preprocessing_callbacks.py` | Ch.4 §accuracy, Ch.5 §deployment |
| 6 | Double standardization — FE + Training both scaling | CRITICAL | ✅ FIXED | `callbacks/feature_engineering_callbacks.py`, `callbacks/training_callbacks.py` | Ch.5 §parity |
| 7 | Feature order mismatch — alphabetical vs C++ computation order | CRITICAL | ✅ FIXED | `deployment/code_generator_factory.py`, `deployment/neural_network_generator.py` | Ch.3 §CodeGen, Ch.5 §parity |
| 8 | Double extraction — reorder undone by 2nd extract call | CRITICAL | ✅ FIXED | `deployment/code_generator_factory.py` | Ch.5 §parity |
| 9 | CNN validation false positives — architecture-unaware validator | MEDIUM | ✅ FIXED | `deployment/validation.py` | Ch.3 §CodeGen, Ch.5 §validation |
| 10 | Confidence threshold for unknown activity rejection | FEATURE | ✅ IMPLEMENTED | `deployment/base_generator.py`, `*_generator.py` (all) | Ch.3 §CodeGen, Ch.4 §robustness |

**Action required:** Collect longer recordings (≥1.5s per window), use sliding window to generate 50+ windows/class, retrain, collect before/after accuracy data for Ch.4.

### Cross-Reference Index

- **Finding 1 (zero-padding)** → Code: `callbacks/feature_engineering_callbacks.py` → Thesis: Ch.3 padding strategy, Ch.5 training-deployment parity → LaTeX snippet: §THESIS_REPORT_INSTRUCTIONS.md "Chiến lược Đệm Cửa sổ"
- **Finding 2 (kurtosis)** → Code: `deployment/base_generator.py` (2 locations), `deployment/micropython_generator.py` (2 locations) → Thesis: Ch.3 code gen, Ch.5 numerical parity
- **Finding 3 (NN bias)** → Data: output biases `[-0.116, -0.160, -0.066, +0.116, +0.048]` → Thesis: Ch.5 out-of-distribution analysis
- **Finding 4 (FFT)** → Decision: use only time-domain features → Thesis: Ch.3 feature selection rationale
- **Parity checklist** (§5.3) → Validates all 8 aspects of training-deployment consistency

- **Finding 5 (padding distortion + tiny dataset)** → Code: `callbacks/feature_engineering_callbacks.py` padding logic, `callbacks/preprocessing_callbacks.py` sliding window → Thesis: Ch.4 accuracy results, Ch.5 deployment analysis → Related: Finding 1 (preceded this, was the first padding issue)
- **Finding 6 (double standardization)** → Code: `callbacks/feature_engineering_callbacks.py` (removed), `callbacks/training_callbacks.py` (keep DataFrame) → Thesis: Ch.5 training-deployment parity chain
- **Finding 7 (feature order mismatch)** → Code: `deployment/code_generator_factory.py` (reorder functions), `deployment/neural_network_generator.py` (weight reorder) → Thesis: Ch.3 code generation, Ch.5 parity
- **Finding 8 (double extraction)** → Code: `deployment/code_generator_factory.py` line 589 removed → Thesis: Ch.5 pipeline correctness
- **Finding 9 (CNN validation)** → Code: `deployment/validation.py` (CNN-aware validation branch + `_validate_cnn_code` + `_estimate_cnn_resources`) → Thesis: Ch.3 multi-architecture support, Ch.5 validation framework
- **Finding 10 (Confidence threshold)** → Code: `deployment/base_generator.py` (har_predict wrapper + softmax), `deployment/neural_network_generator.py`, `deployment/random_forest_generator.py`, `deployment/svm_generator.py`, `deployment/cnn_generator.py`, `deployment/micropython_generator.py` → Thesis: Ch.3 code generation robustness, Ch.4 real-world deployment

---

## SESSION LOG

| Date | Session | Changes Made |
|------|---------|-------------|
| 2025-02-27 | Initial creation | Documented all 4 findings from device deployment debugging session |
| 2025-02-27 | Restructure v2.0 | Added Quick Context, Cross-Reference Index, Session Log for cross-session AI continuity |
| 2025-02-27 | Finding 5 added | Documented edge-replication distortion and tiny dataset root cause analysis |
| 2026-03-01 | Findings 6-8 added | Three critical deployment bugs: double standardization, feature order mismatch, double extraction in code gen pipeline |

*Add a row here each time this file is updated.*

---

**Mục đích:** Ghi lại chi tiết các phát hiện kỹ thuật quan trọng phát sinh trong quá trình phát triển và kiểm thử framework. Các phát hiện này tạo nên sự khác biệt cốt lõi giữa framework của chúng tôi với các nền tảng thương mại (Edge Impulse, SensiML) và là **luận điểm mạnh cho bảo vệ luận văn**.

---

## 1. LỖI ZERO-PADDING TRONG PIPELINE TRÍCH XUẤT ĐẶC TRƯNG

### 1.1 Mô tả vấn đề

**Tên kỹ thuật:** Artifact đệm số không trong trích xuất đặc trưng (Zero-Padding Artifact in Feature Extraction)

**Bối cảnh:** Khi thu thập dữ liệu cảm biến và phân đoạn cửa sổ thời gian (time windowing), người dùng chọn các đoạn hoạt động có thể ngắn hơn kích thước cửa sổ mục tiêu. Ví dụ:
- Kích thước mục tiêu: 150 mẫu (1.5 giây ở 100Hz)
- Cửa sổ thực tế: 70-120 mẫu (tùy thuộc vào hoạt động)

**Phương pháp cũ (SAI):**
```python
# Zero-padding: đệm cửa sổ ngắn bằng hàng toàn số 0
padding_needed = target_window_samples - len(df_window)
padding_df = pd.DataFrame(0, index=range(padding_needed), columns=df_window.columns)
df_padded = pd.concat([df_window, padding_df], ignore_index=True)
```

**Tại sao sai:** 
- Hàng padding có `aX = aY = aZ = 0`, dẫn đến `acc_magnitude = sqrt(0² + 0² + 0²) = 0.0`
- Trong thực tế, `acc_magnitude ≥ 9.81 m/s²` (do trọng lực) khi thiết bị đứng yên
- Giá trị `acc_magnitude = 0.0` là **vật lý bất khả thi**

### 1.2 Dữ liệu chứng minh

**Dữ liệu huấn luyện (bị nhiễm zero-padding):**
```
File: persistent_data/training/running_still_walking_and_2_more_train.csv
110 mẫu huấn luyện, TẤT CẢ đều có:
- acc_mag_min = 0.000  (BẤT KHẢ THI — gia tốc kế luôn > 0)
- acc_mag_median ≈ 0.0 cho nhiều windows (bị kéo xuống do 30-50% số hàng là 0)
- acc_mag_q25 = 0.0 cho hầu hết windows
- gyro_mag_min = 0.0 (tương tự)
```

**Kích thước cửa sổ thực tế (tất cả đều ngắn hơn 150):**
```
Hoạt động          | Số mẫu thực | Tỉ lệ zero-padding
--------------------|-------------|--------------------
Running             | 70          | 53% (80 hàng 0)
Still               | 101         | 33% (49 hàng 0)
Walking             | 110         | 27% (40 hàng 0)
Walking Downstairs  | 120         | 20% (30 hàng 0)
Walking Upstairs    | 120         | 20% (30 hàng 0)
```

**Trên thiết bị thực (Seeed XIAO nRF52840):**
- Bộ đệm 150 mẫu luôn **CHỨA ĐẦY** dữ liệu cảm biến thực
- Không có giá trị 0 nào (trừ khi thiết bị trong trạng thái rơi tự do — micro-gravity)
- acc_mag_min ≈ 8-11 m/s² thay vì 0.0

### 1.3 Hậu quả triển khai

**Triệu chứng:** Mô hình Neural Network triển khai trên Seeed XIAO **luôn dự đoán "walking_downstairs"** bất kể hoạt động thực tế (đứng yên, đi bộ, chạy, v.v.).

**Phân tích nguyên nhân:**
1. Mô hình được huấn luyện với phân phối đặc trưng bị nhiễm zero-padding
2. Trên thiết bị thực, phân phối đặc trưng hoàn toàn khác (không có giá trị 0)
3. Đặc trưng đầu vào nằm **ngoài phân phối huấn luyện** hoàn toàn
4. Khi đặc trưng out-of-distribution, hidden layer activations ≈ 0
5. Output biases chi phối: `[-0.116, -0.160, -0.066, +0.116, +0.048]`
6. Class 3 (walking_downstairs) có bias cao nhất (+0.116) → luôn thắng

### 1.4 Giải pháp đã triển khai

**File sửa:** `callbacks/feature_engineering_callbacks.py`

**Phương pháp mới: Edge-Value Replication (Lặp giá trị biên)**
```python
# Lấy hàng cuối cùng của cửa sổ thực
last_row = df_window.iloc[[-1]]
# Lặp lại hàng cuối để đệm đến kích thước mục tiêu
padding_df = pd.concat([last_row] * padding_needed, ignore_index=True)
df_padded = pd.concat([df_window, padding_df], ignore_index=True)
```

**Bổ sung ngưỡng tối thiểu:**
```python
min_window_samples = max(10, int(target_window_samples * 0.3))
if len(df_window) < min_window_samples:
    # Loại bỏ cửa sổ quá ngắn (< 30% target)
    discarded_count += 1
    continue
```

**Ưu điểm:**
- Bảo toàn đặc tính thống kê của tín hiệu (min, max, range không bị ảnh hưởng bởi giá trị 0 giả)
- `acc_mag_min` giờ phản ánh giá trị vật lý thực (> 0 do trọng lực)
- Phân phối đặc trưng huấn luyện khớp với phân phối trên thiết bị thực
- Cửa sổ quá ngắn bị loại bỏ thay vì bị đệm quá nhiều

---

## 2. SAI LỆCH CÔNG THỨC KURTOSIS/SKEWNESS

### 2.1 Mô tả vấn đề

**Tên kỹ thuật:** Không khớp công thức thống kê giữa Python training và C++ deployment (Statistical Formula Mismatch)

**Công thức Python (pandas):**
Pandas `.kurtosis()` sử dụng excess kurtosis với Fisher's definition:
$$\text{kurtosis} = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum\left(\frac{x_i - \bar{x}}{s_{\text{pop}}}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}$$

Trong đó $s_{\text{pop}} = \sqrt{\frac{1}{n}\sum(x_i - \bar{x})^2}$ (population standard deviation)

**Công thức C++ cũ (SAI):**
```cpp
// Sử dụng sample std (Bessel correction) thay vì population std
float sample_var = variance * n / (n - 1.0f + 0.001f);
float sample_std = sqrtf(sample_var);    // ← SAI: dùng sample std
float z = (mag[i] - mean) / sample_std;  // z-score sai
```

### 2.2 Phân tích toán học

Khi dùng sample std thay vì population std cho z-score:

**Cho skewness:**
$$\text{error\_ratio} = \left(\frac{n-1}{n}\right)^{3/2}$$
Với n = 150: error = $(149/150)^{1.5} \approx 0.990$ → sai lệch ~1.0%

**Cho kurtosis:**  
$$\text{error\_ratio} = \left(\frac{n-1}{n}\right)^{2}$$
Với n = 150: error = $(149/150)^2 \approx 0.987$ → sai lệch ~1.3%

### 2.3 Giải pháp đã triển khai

**Files sửa:** 
- `deployment/base_generator.py` — 2 vị trí (magnitude stats + per-axis stats)
- `deployment/micropython_generator.py` — 2 vị trí

**Code mới (ĐÚNG):**
```cpp
// Sử dụng population std cho z-scores, khớp chính xác pandas
float pop_std = sqrtf(variance > 0.0f ? variance : 0.0001f);
float z = (mag[i] - mean) / (pop_std + 0.0001f);
```

**Ảnh hưởng:** Tất cả code generators đều đã được sửa:
- BaseCodeGenerator (kế thừa bởi: NeuralNetworkGenerator, RandomForestGenerator, SVMGenerator, CNNGenerator, ARMCortexMGenerator, ZephyrGenerator)
- MicroPythonCodeGenerator

---

## 3. HÀNH VI DỰ ĐOÁN MẶC ĐỊNH CỦA MẠNG NƠ-RON

### 3.1 Mô tả

Khi tất cả đặc trưng đầu vào nằm ngoài phân phối huấn luyện, mạng nơ-ron có hành vi "mặc định" được xác định bởi output biases.

### 3.2 Dữ liệu

**Kiến trúc mạng:** Input(33) → Hidden1(128, ReLU) → Hidden2(64, ReLU) → Output(5, Softmax)

**Output biases từ file .cpp:**
```
Class 0 (RUNNING):           -0.116
Class 1 (STILL):             -0.160
Class 2 (WALKING):           -0.066
Class 3 (WALKING_DOWNSTAIRS): +0.116  ← CAO NHẤT
Class 4 (WALKING_UPSTAIRS):  +0.048
```

**Cơ chế:**
1. StandardScaler: `z = (x - mean) / std`
2. Khi x nằm ngoài phân phối, z rất lớn hoặc rất nhỏ
3. Clamp [-10, 10] giới hạn giá trị
4. Hidden layers với ReLU tạo activations gần 0 (do input distribution sai)
5. Output = bias terms chi phối → Class 3 (bias +0.116) luôn thắng

### 3.3 Ý nghĩa khoa học

- **Bằng chứng:** Đây là bằng chứng mạnh rằng vấn đề là **distribution mismatch**, không phải lỗi thuật toán hay hardware
- **Chẩn đoán:** Có thể dự đoán class nào sẽ bị bias bằng cách kiểm tra output biases
- **Phòng ngừa:** Framework nên thêm chức năng kiểm tra phân phối đặc trưng (distribution check) giữa training data và sample device data trước khi triển khai

---

## 4. VẤN ĐỀ TƯƠNG ĐỒNG FFT (ĐÃ GIẢI QUYẾT TRƯỚC ĐÓ)

### 4.1 Mô tả

Framework đã chủ động loại bỏ đặc trưng miền tần số (FFT) do vấn đề tương đồng:
- Python NumPy FFT: double-precision, FFTPACK/FFTW backend
- C++ embedded FFT: single-precision, approximation hoặc ARM CMSIS-DSP
- Sai lệch 5-15% giữa hai implementation
- Gây phân loại sai 8-12% khi triển khai

### 4.2 Giải pháp

Sử dụng CHỈ đặc trưng miền thời gian (time-domain features), đảm bảo:
- Tính toán dạng đóng, xác định (closed-form deterministic)
- Bit-exact giữa Python và C++ (sau khi sửa kurtosis formula)
- Zero-crossing rate và mean-crossing rate làm proxy cho thông tin tần số

**Đã ghi trong báo cáo:** Chương 3, mục Feature Extraction → "Lý do Loại trừ Đặc trưng FFT"

---

## 5. EDGE-REPLICATION VẪN LÀM SAI LỆCH ĐẶC TRƯNG + DỮ LIỆU QUÁ ÍT

### 5.1 Mô tả vấn đề

Sau khi sửa lỗi zero-padding (Finding 1) và kurtosis (Finding 2), model đã cải thiện: nhận đúng "still" và phần nào "running". Tuy nhiên, model KHÔNG BAO GIỜ dự đoán "walking" (Class 2) hay "walking_upstairs" (Class 4) — tất cả đều rơi vào "walking_downstairs" (Class 3).

**Hai nguyên nhân gốc:**
1. **Edge-replication padding vẫn gây sai lệch lớn** — mặc dù tốt hơn zero-padding, nó vẫn thay đổi nghiêm trọng các đặc trưng thống kê
2. **Dataset quá nhỏ** — chỉ 109 mẫu huấn luyện / 33 đặc trưng (21 mẫu/lớp)

### 5.2 Dữ liệu chứng minh

#### Sai lệch đặc trưng do padding (ví dụ: cửa sổ running 70 mẫu → pad đến 150)

| Đặc trưng | Gốc (70 mẫu) | Sau pad (150) | Sai lệch |
|-----------|---------------|---------------|----------|
| `acc_mag_std` | 7.661 | 5.269 | **-31.2%** |
| `acc_mag_iqr` | 5.739 | 2.326 | **-59.5%** |
| `acc_mag_kurtosis` | 1.253 | 4.970 | **+296.5%** |
| `acc_mag_mean_crossing_rate` | 0.100 | 0.047 | **-53.3%** |
| `gyro_mag_mean` | 209.303 | 151.771 | **-27.5%** |
| `gyro_mag_median` | 181.094 | 101.430 | **-44.0%** |
| `gyro_mag_iqr` | 171.228 | 73.236 | **-57.2%** |
| `gyro_mag_kurtosis` | -0.666 | 2.404 | **+460.9%** |
| `acc_jerk_mag_mean` | 2.249 | 1.042 | **-53.7%** |

**Cơ chế gây sai lệch:** Edge-replication lặp lại giá trị cuối cùng → vùng padding có variance = 0, jerk = 0, không có mean-crossing. Điều này:
- Giảm `std`, `iqr`, `rms` (vì thêm giá trị hằng số)  
- Tăng `kurtosis` mạnh (phân phối bị nhọn ở giá trị cuối)
- Giảm `mean_crossing_rate` mạnh (vùng padding không cross mean)
- Giảm `jerk_mean` mạnh (jerk = 0 trong vùng constant)

#### Dataset quá nhỏ

| Chỉ số | Giá trị | Yêu cầu tối thiểu |
|--------|---------|-------------------|
| Tổng mẫu huấn luyện | 109 | ≥ 500 |
| Mẫu/lớp | ~21 | ≥ 50 (tốt nhất 100+) |
| Số đặc trưng | 33 | — |
| Tỷ lệ mẫu/đặc trưng | 3.3:1 | ≥ 10:1 |
| Kích thước cửa sổ gốc | 70-121 mẫu | 150 (target) |
| Phần trăm padding | 19-53% | 0% (lý tưởng) |

#### Các lớp walking gần giống nhau

| Lớp | `acc_mag_mean` | `gyro_mag_mean` | `acc_jerk_mag_mean` |
|-----|----------------|-----------------|---------------------|
| walking | 10.937 | 106.037 | 0.670 |
| walking_downstairs | 10.657 | 102.573 | 0.748 |
| walking_upstairs | 10.325 | 75.799 | 0.649 |

Walking và walking_downstairs chênh lệch chỉ ~3% trên hầu hết đặc trưng → với 21 mẫu/lớp, model không thể học biên phân lớp ổn định.

### 5.3 Phân tích hệ quả

**Tại sao model hoạt động 95.8% trên test set nhưng kém trên thiết bị:**
1. Test set (24 mẫu) cũng là dữ liệu padding → cùng phân phối sai lệch với train
2. Thiết bị thu thập 150 mẫu thật → đặc trưng nằm NGOÀI phân phối training
3. Model quá ít dữ liệu → overfitting vào đặc trưng PADDING, không phải hoạt động thật
4. Cụ thể: trên thiết bị, `std` cao hơn, `kurtosis` thấp hơn, `jerk` cao hơn → model "confuse" và chọn lớp mặc định (walking_downstairs)

**So sánh với Edge Impulse:** Edge Impulse yêu cầu tối thiểu 3 phút dữ liệu mỗi lớp (~18,000 mẫu ở 100Hz) — framework của chúng tôi cần cảnh báo rõ hơn khi dữ liệu không đủ.

### 5.4 Giải pháp

**Cần thực hiện (theo thứ tự ưu tiên):**

1. **Thu thập dữ liệu dài hơn**: Ghi ít nhất 3-5 giây mỗi hoạt động (300-500 mẫu ở 100Hz). Dùng data acquisition sketch kết hợp serial recording.

2. **Dùng sliding window trong tab Preprocessing**: Framework ĐÃ có tính năng này (`callbacks/preprocessing_callbacks.py` → `generate_sliding_windows_from_current()`). Từ 1 recording 5 giây, trích xuất nhiều cửa sổ 1.5 giây chồng lấp (overlap 50%) → ~7 cửa sổ/recording. Nhân với 10 recording = 70 cửa sổ/lớp.

3. **HOẶC giảm kích thước cửa sổ**: Đặt window size = 70 mẫu (0.7 giây) → loại bỏ hoàn toàn padding. Tradeoff: ít ngữ cảnh hơn cho phân loại.

4. **Thử Random Forest**: Xử lý dataset nhỏ tốt hơn NN, ít overfitting hơn, robust hơn với sai lệch phân phối.

5. **Cân nhắc hợp nhất walking classes**: Nếu walking/walking_downstairs/walking_upstairs không phân biệt được, gộp thành 1 lớp "walking" → tăng số mẫu/lớp + giảm complexity.

**Mã nguồn không cần sửa** — vấn đề là chất lượng dữ liệu, không phải lỗi code. Framework nên bổ sung cảnh báo khi dataset quá nhỏ hoặc padding quá nhiều (cải tiến UX).

---

## 6. CHUẨN HÓA KÉP — FE VÀ TRAINING ĐỀU SCALE (Double Standardization)

### 6.1 Mô tả vấn đề

**Tên kỹ thuật:** Chuẩn hóa kép gây scaler gần đồng nhất (Double Standardization → Identity Scaler)

**Bối cảnh:** Pipeline Feature Engineering (FE) áp dụng `StandardScaler` lên feature matrix trước khi ghi ra CSV. Sau đó, Training pipeline đọc CSV và áp dụng `StandardScaler` lần nữa → model.scaler học trên dữ liệu **đã chuẩn hóa** → scaler.mean_ ≈ 0, scaler.scale_ ≈ 1.

**Chuỗi lỗi:**
```
FE tab: features → StandardScaler.fit_transform() → writes scaled CSV
Training tab: reads CSV → StandardScaler.fit_transform() AGAIN → scaler learns means≈0, stds≈1
Code gen: exports scaler means=[0.07, -0.02, ...], stds=[0.98, 1.01, ...] → identity transform
Device: z = (x - 0.07) / 0.98 ≈ x → NO SCALING → all activations wrong
```

### 6.2 Dữ liệu chứng minh

**Scaler parameters trong C++ trước sửa (từ model cũ):**
```
feature_means[0] = 0.07   (should be 11.79 for acc_mag_mean)
feature_stds[0]  = 0.98   (should be 2.49)
```

**Sau sửa (model mới):**
```
feature_means[0] = 11.791  ✅
feature_stds[0]  = 2.491   ✅
```

### 6.3 Giải pháp

**File sửa:** `callbacks/feature_engineering_callbacks.py`
- **XÓA** block `StandardScaler/MinMaxScaler/RobustScaler` trong FE callback
- FE tab giờ ghi raw features ra CSV
- Scaling chỉ xảy ra MỘT LẦN trong training pipeline (`EdgeMLModel.preprocess_data()`)

**File sửa:** `callbacks/training_callbacks.py`
- Giữ `pd.DataFrame` (không gọi `.values`) khi truyền vào model.train()
- Đảm bảo `feature_names` được lưu trong model dict (trước đó bị `None` do `.values` mất tên cột)

---

## 7. SAI THỨ TỰ ĐẶC TRƯNG — ALPHABETICAL VS C++ COMPUTATION ORDER (Feature Order Mismatch)

### 7.1 Mô tả vấn đề

**Tên kỹ thuật:** Không khớp thứ tự đặc trưng giữa model training và C++ extraction

**Bối cảnh:** Python `pd.DataFrame(list_of_dicts)` sắp xếp cột theo **thứ tự alphabet**. Model sklearn được huấn luyện trên thứ tự này. Nhưng C++ `extract_magnitude_stats()` trích xuất theo **thứ tự tính toán** cố định.

**Thứ tự alphabet (Python training):**
```
[0] acc_jerk_mag_max        → value 3.724
[1] acc_jerk_mag_mean       → value 1.011
[2] acc_jerk_mag_std        → value 0.848
...
[7] acc_mag_mean            → value 11.791
```

**Thứ tự C++ extraction:**
```
[0] acc_mag_mean            → value 11.791
[1] acc_mag_std             → value 3.837
[2] acc_mag_min             → value 4.975
...
```

**Hậu quả:** TẤT CẢ 33 đặc trưng bị sai lệch vị trí → scaler áp dụng sai mean/std cho sai feature → weights nhân với sai feature → kết quả dự đoán hoàn toàn ngẫu nhiên.

### 7.2 Giải pháp

Thêm **feature reorder remapping** vào `deployment/code_generator_factory.py`:

```python
def get_cpp_feature_order(feature_names):
    """Determine C++ extraction order from feature names."""
    # Groups: acc_mag → gyro_mag → jerk_mag
    # Stats per group: mean, std, min, max, range, median, q25, q75,
    #                  iqr, skewness, kurtosis, rms, energy,
    #                  zero_crossings, mean_crossing_rate
    ...

def reorder_model_parameters(enhanced_data, reorder_indices, cpp_order):
    """Reorder scaler means/stds, NN input weights, RF tree indices, SVM SV columns."""
    ...
```

Áp dụng reorder tại thời điểm code generation: scaler means/stds, input weight matrix rows (cho NN), tree feature indices (cho RF), support vector columns (cho SVM).

---

## 8. TRÍCH XUẤT KÉP — REORDER BỊ HỦY BỞI LẦN GỌI THỨ HAI (Double Extraction Bug)

### 8.1 Mô tả vấn đề

**Tên kỹ thuật:** Gọi `extract_real_model_parameters()` hai lần hủy kết quả reorder

**Bối cảnh:** Trong pipeline code generation:
1. `generate_and_save_deployment_code()` gọi `extract_real_model_parameters()` → reorder ✅ → `feature_names` thành C++ order
2. Rồi gọi `generate_deployment_code()` → gọi `extract_real_model_parameters()` LẦN NỮA
3. Lần gọi thứ 2: `feature_names` đã ở C++ order → `compute_feature_reorder_indices()` trả `None` → **KHÔNG reorder**
4. NHƯNG: scaler means/stds được trích xuất LẠI từ scaler object (thứ tự alphabet!) → **GHI ĐÈ** giá trị đã reorder

**Kết quả:** Generated code có `feature_means[0] = 3.724` (alphabetical: acc_jerk_mag_max) thay vì `11.791` (C++ order: acc_mag_mean).

### 8.2 Giải pháp

**File sửa:** `deployment/code_generator_factory.py`
- **XÓA** lệnh gọi `extract_real_model_parameters()` trong `generate_and_save_deployment_code()` (line 589)
- Giữ lại lệnh gọi duy nhất trong `generate_deployment_code()` (line 505)
- Thêm comment giải thích tại sao KHÔNG được gọi hai lần

### 8.3 Xác minh

Sau sửa, test code generation pipeline:
```
feature_means[0] = 11.791  ✅ (acc_mag_mean, C++ extraction order)
Feature order remapped: model order → C++ extraction order
Applied feature reorder to input weight matrix ✅
```

---

## 9. Validation không nhận diện kiến trúc CNN — False Positive

### 9.1 Vấn đề

`DeploymentValidator.validate_generated_code()` áp dụng kiểm tra dành cho model dựa trên **feature extraction** (NN, RF, SVM) cho tất cả model types, kể cả **CNN**. CNN sử dụng cửa sổ cảm biến thô (raw sensor windows) — **không có feature extraction, không có scaling** — nên 5 kiểm tra đều cho kết quả sai:

| Check | Kỳ vọng (feature-based) | CNN thực tế | Kết quả |
|-------|------------------------|-------------|---------|
| `feature_means[NUM_FEATURES]` | Có | Không có — CNN không scale | ❌ False positive |
| `feature_stds[NUM_FEATURES]` | Có | Không có | ❌ False positive |
| `extract_magnitude_stats` in source | Có (nếu acc_mag_*) | Không — CNN dùng raw data | ❌ False positive |
| `features == NULL` check | Có | Không có biến features | ⚠️ False warning |
| `std < 0.0001f` div-by-zero | Có | Không scaling | ⚠️ False warning |

**Nguyên nhân gốc:** Validator thiết kế theo kiến trúc feature-based (extract → scale → predict). CNN có kiến trúc hoàn toàn khác (collect window → Conv1D → MaxPool → Dense → predict).

### 9.2 Giải pháp

Thêm **model-type detection** vào validator:

1. **Phát hiện CNN**: Kiểm tra `model_type in ('pytorch_cnn', 'cnn')` hoặc `'har_predict_from_window' in source_code`
2. **CNN-specific checks** thay thế cho feature-based checks:
   - `WINDOW_SIZE` defined ✓
   - `N_CHANNELS` defined ✓
   - `NUM_CLASSES` matches ✓
   - `conv1d` function present ✓
   - `har_predict_from_window` function present ✓
   - CNN weight arrays present ✓
   - Kiểm tra nghịch: **không nên** có `feature_means`/`extract_features` trong CNN code
3. **Skip scaling validation** cho CNN (không có scaler)
4. **CNN resource estimation** riêng: RAM = sensor buffer + layer activations (no FE buffers)

### 9.3 Kết quả

Trước sửa:
```
❌ Missing feature_means array in source
❌ Missing feature_stds array in source
❌ Model trained with magnitude features but C++ uses per-axis extraction!
⚠️ Missing NULL pointer check for features array
⚠️ Missing division-by-zero protection in feature scaling
```

Sau sửa:
```
✅ WINDOW_SIZE=150 ✓
✅ N_CHANNELS=6 ✓
✅ NUM_CLASSES=5 ✓
✅ Conv1D layer function present ✓
✅ Window-based prediction function present ✓
✅ CNN weight arrays present (10 arrays) ✓
✅ CNN architecture: raw window input (no feature extraction needed) ✓
```

### 9.4 Ý nghĩa cho thesis

- Framework hỗ trợ **hai kiến trúc triển khai** (feature-based và end-to-end CNN) — commercial platforms thường chỉ hỗ trợ 1
- Validator phải **architecture-aware** — không thể dùng cùng checklist cho mọi model type
- CNN trên MCU: inference nhanh hơn (không mất thời gian FE) nhưng RAM cao hơn (lưu toàn bộ window)

**Files changed:** `deployment/validation.py`

---

## TỔNG HỢP: TẠI SAO FRAMEWORK NÀY VƯỢT TRỘI

### So sánh chi tiết với Edge Impulse

| Khía cạnh | Edge Impulse | Framework chúng tôi | Lợi thế |
|-----------|-------------|---------------------|---------|
| **Pipeline transparency** | Hộp đen (black-box) | Mã nguồn mở 100% | Chúng tôi: phát hiện và sửa lỗi |
| **Padding strategy** | Không rõ, không kiểm chứng | Edge-value replication, đã kiểm chứng | Chúng tôi: tránh artifact |
| **Formula verification** | Không thể kiểm tra | Python ↔ C++ đã xác minh | Chúng tôi: bit-exact |
| **Feature distribution check** | Không | Có thể qua debug output | Chúng tôi: phát hiện mismatch |
| **FFT consistency** | Có thể sai lệch, không biết | Loại bỏ FFT, chỉ time-domain | Chúng tôi: không có rủi ro |
| **Chi phí** | $20-99/tháng | $0 (mã nguồn mở) | Chúng tôi: miễn phí |
| **Debugging deployment** | Hạn chế | Full source code access | Chúng tôi: debug toàn diện |

### Vấn đề "Ít được nghiên cứu" trong tài liệu

Training-deployment parity trong edge ML là một vấn đề **ít được nghiên cứu chuyên sâu** trong tài liệu học thuật:
- Hầu hết bài báo báo cáo accuracy trên test set (Python) mà không xác minh trên device thực
- Các nền tảng thương mại xử lý đây là chi tiết triển khai nội bộ (không công khai)
- Framework của chúng tôi là một trong số ít hệ thống công khai ghi nhận VÀ giải quyết vấn đề này

### Danh sách các biện pháp đảm bảo tương đồng trong framework

1. ✅ **Loại bỏ FFT features** — tránh sai lệch triển khai FFT
2. ✅ **Edge-value replication** — thay thế zero-padding
3. ✅ **Verified kurtosis/skewness** — population std, khớp pandas
4. ✅ **Cùng thứ tự đặc trưng** — Python feature_extraction.py ↔ C++ extract_magnitude_stats()
5. ✅ **Cùng đơn vị cảm biến** — m/s² cho gia tốc, deg/s cho gyroscope
6. ✅ **Cùng tốc độ lấy mẫu** — 100Hz trong cả thu thập và suy luận
7. ✅ **Cùng kích thước cửa sổ** — 150 mẫu, từ metadata model
8. ✅ **StandardScaler nhất quán** — cùng mean/std values, cùng clamp range
9. ✅ **Loại bỏ chuẩn hóa kép** — scaling chỉ xảy ra 1 lần trong training pipeline
10. ✅ **Feature order remapping** — reorder scaler/weights tại code-gen để khớp C++ order
11. ✅ **Idempotent parameter extraction** — `extract_real_model_parameters` chỉ gọi 1 lần
12. ✅ **Architecture-aware validation** — validator phân biệt CNN (raw window) vs feature-based (NN/RF/SVM)

---

## 10. GHI CHÚ CHO TÁC GIẢ VÀ AI ASSISTANT

### Khi retrain model:
1. Xóa dữ liệu huấn luyện cũ (`persistent_data/training/*.csv`)
2. Xóa cửa sổ cũ (`persistent_data/windows/*.csv`)
3. Chạy lại Feature Engineering từ đầu
4. Kiểm tra: `acc_mag_min` trong training data PHẢI > 0
5. Huấn luyện và tạo code mới
6. So sánh accuracy trước/sau sửa → ghi vào báo cáo

### Khi viết báo cáo:
1. Tham khảo file `THESIS_REPORT_INSTRUCTIONS.md` cho TODO list
2. Dùng mẫu LaTeX gợi ý trong file đó
3. Nhớ viết bằng tiếng Việt
4. Giữ thuật ngữ tiếng Anh trong ngoặc lần đầu xuất hiện
5. So sánh với Edge Impulse phải công bằng — ghi nhận điểm mạnh của họ (85+ boards, UX tốt)

### Khi chuẩn bị bảo vệ:
1. Demo trực tiếp trên Seeed XIAO (trước/sau fix nếu có thể)
2. Slide key: "Phát hiện lỗi mà hộp đen không thể" 
3. Biểu đồ so sánh phân phối đặc trưng (training vs device)
4. Video thiết bị hoạt động chính xác sau fix

---

---

## Finding 10: Confidence Threshold for Unknown Activity Rejection

**Severity:** FEATURE  
**Status:** ✅ IMPLEMENTED  
**Date:** 2026-03-06  
**Files:** All code generators (`deployment/*_generator.py`, `deployment/base_generator.py`)

### §10.1 Mô tả vấn đề

During device testing (Session 21), all 4 trained models showed severe class confusion:
- SVM predicted `walking_downstairs` 100% of the time (26/26 windows)
- NN predicted `walking` 0% of the time, heavy bias toward `walking_downstairs`
- RF and MLP showed similar patterns

Root cause: with orientation-invariant features (magnitude-based), walking/walking_downstairs/walking_upstairs have nearly identical feature distributions (acc_mag: [9.74-11.62], gyro_mag: [65-122] for all three). The model was always forced to pick one class even when it had no confidence.

Commercial platforms like Edge Impulse include an "anomaly detection" layer for uncertain inputs. Our framework lacked any uncertainty rejection mechanism.

### §10.2 Giải pháp

Implemented confidence threshold across ALL generated code:

**Approach:** Each model computes per-class probabilities, and if the max probability is below `CONFIDENCE_THRESHOLD` (default 0.6), the prediction returns -1 ("unknown").

**Confidence computation by model type:**
- **Neural Network (MLP):** Softmax over output logits → max(softmax) as confidence
- **Random Forest:** Vote proportion = votes[class] / total_trees → max proportion as confidence  
- **SVM:** Softmax over OvR decision scores → max(softmax) as confidence
- **CNN:** Softmax over final dense layer logits → max(softmax) as confidence

**API changes (C/C++):**
```c
// Before:
int har_predict(float features[NUM_FEATURES]);
int har_predict_from_window(float window[WINDOW_SIZE][N_CHANNELS]);

// After:
int har_predict(float features[NUM_FEATURES], float* confidence);  // returns -1 if below threshold
int har_predict_from_window(float window[WINDOW_SIZE][N_CHANNELS], float* confidence);  // same
```

**MicroPython API changes:**
```python
# Before:
predicted = har_predict(features)  # returns int

# After:
predicted, confidence = har_predict(features)  # returns (int, float), -1 if below threshold
```

### §10.3 Hậu quả / Phân tích

- Predicted class -1 maps to `get_activity_name(-1)` → "unknown" string
- Device Test tab already handles arbitrary activity name strings in CSV format
- Default threshold 0.6 means model must be ≥60% confident to make a prediction
- For 5-class problem, random chance is 20%, so 60% is 3× random — a reasonable bar
- RF confidence is naturally interpretable (e.g., 80/100 trees agree = 0.8 confidence)
- NN/SVM confidence via softmax may overestimate — calibration could improve this in future

### §10.4 Thesis Significance

**Differentiator vs Edge Impulse:** Edge Impulse's confidence rejection requires a separate "anomaly detection" block trained on additional data. Our approach extracts confidence directly from the model's existing outputs — zero additional training cost, zero additional memory.

**Defense argument:** Shows the framework handles real deployment challenges (class overlap, out-of-distribution inputs) gracefully, which is a hallmark of production-ready systems. This is a key feature for safety-critical HAR applications.

---

## APPENDIX: DOCUMENT MAINTENANCE

### How to use this file across sessions
1. **New session?** Read the **QUICK CONTEXT** table at the top first — it summarizes everything in 10 seconds.
2. **Made a code change?** Check if it affects any finding listed in the Cross-Reference Index. If yes, update the relevant section AND the status in the Quick Context table.
3. **New finding?** Add it as §N+1 following the same structure: Description → Evidence → Impact → Solution. Update the Quick Context table and Cross-Reference Index.
4. **Retrained model?** Add a row to the Session Log and update any accuracy numbers in §1-§4.

### Structure of each finding section
Each finding follows a consistent pattern for ease of scanning:
- **§N.1 Mô tả vấn đề** — What went wrong and why
- **§N.2 Dữ liệu chứng minh** — Hard evidence (code, numbers, data)
- **§N.3 Hậu quả / Phân tích** — Real-world impact
- **§N.4 Giải pháp** — What was changed and where

*File này nên được cập nhật sau mỗi lần retrain hoặc phát hiện thêm vấn đề mới.*
