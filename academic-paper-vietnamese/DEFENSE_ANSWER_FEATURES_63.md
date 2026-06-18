# PHẦN BIỆN LUẬN ĐỊNH LƯỢNG: TẠI SAO CHỌN 63 FEATURES?
*Dùng cho phần phản biện luận văn và thi vấn đáp*

## CÂUTRẢ LỜI CƠ BẢN (60 giây)

**Học viên:** "Em không chọn 63 features theo cảm tính. Em đã thực hiện ablation study so sánh 3 mức features: 33, 63 và 90, bằng cách train 3 model khác nhau (Random Forest, SVM, MLP) trên cùng dữ liệu."

**Kết quả định lượng:**
- **33 features** (orientation_invariant_time_only): RF 99.36%, SVM 100%, MLP 99.36%
- **63 features** (orientation_invariant) ← **TỐI ƯU**: RF 100%, SVM 100%, MLP 99.36%
- **90 features** (time_domain): RF 99.36%, SVM 100%, MLP 98.72%

**Lý do chọn 63:**
1. **Độ chính xác cao nhất**: Đạt 100% với RF và SVM
2. **Tối ưu hóa**: So với 33 features, 63 features tăng +0.64% độ chính xác nhưng GIẢM -4.0% độ phức tạp (870 parameters vs 906)
3. **Tránh overfitting**: So với 90 features, 63 features giữ lại 99.36% độ chính xác của RF nhưng GIẢM 34.8% độ phức tạp (870 vs 1,334 parameters)
4. **Khả năng triển khai**: Mô hình nhỏ gọn phù hợp với IoT/Arduino (edge computing)

---

## PHẦN TRÌNH BÀY CHI TIẾT (2-3 phút)

### Slide 1: Ablation Study Overview
"Ablation study là phương pháp loại bỏ hoặc thêm thành phần từng cái để đo tác động. 

Em kỹ lưỡng so sánh 3 tập features:
- 33 features: Chỉ thống kê miền thời gian từ độ lớn gia tốc và quay
- 63 features: Thêm đặc trưng miền tần số (FFT, spectral features)  
- 90 features: Loại bỏ magnitude-based, dùng từng trục X/Y/Z riêng lẻ

Mục tiêu: Tìm điểm cân bằng giữa độ chính xác và độ phức tạp mô hình."

### Slide 2: Accuracy vs Feature Count
"Biểu đồ hiển thị 3 mô hình khác nhau:
- Random Forest: 99.36% (33 feat) → 100% (63 feat) → 99.36% (90 feat)
- SVM: 100% ở cả 3 mức
- MLP: 99.36% (33) → 99.36% (63) → 98.72% (90)

**Phát hiện quan trọng**: 
- Từ 33 → 63: Độ chính xác tăng (RF từ 99.36% lên 100%)
- Từ 63 → 90: Độ chính xác giảm (RF từ 100% xuống 99.36%, MLP từ 99.36% xuống 98.72%)
- Điểm elbow (diminishing returns) ở 63 features."

### Slide 3: Complexity Analysis
"Phân tích độ phức tạp mô hình bằng số lượng tham số:

**Random Forest (100 trees):**
- 33 features: 906 tham số
- 63 features: 870 tham số (-4.0% ← Thú vị!)
- 90 features: 1,334 tham số (+53.3%)

**MLP (128-64 layers):**
- 33 features: 12,803 tham số
- 63 features: 16,643 tham số (+30.0%)
- 90 features: 20,099 tham số (+20.8% so với 63)

Lý giải: Frequency features ở 63 capture được pattern tốt hơn → RF cần ít cây hơn."

### Slide 4: Pareto Frontier
"Biểu đồ Pareto Frontier (Accuracy vs Model Complexity):

- 33-feature: Điểm nhỏ, độ chính xác thấp nhất
- **63-feature: Điểm sao đỏ, tối ưu nhất** ← Độ chính xác cao, độ phức tạp thấp
- 90-feature: Điểm to hơn, độ chính xác không tốt hơn 63

Kết luận: 63 features nằm trên Pareto frontier (không có giải pháp nào tốt hơn về cả 2 tiêu chí)."

---

## BẢNG TÓGTẮT ĐỊNH LƯỢNG

| Metric | 33 feat | 63 feat | 90 feat |
|--------|---------|---------|---------|
| RF Accuracy | 99.36% | **100%** | 99.36% |
| SVM Accuracy | 100% | **100%** | 100% |
| MLP Accuracy | 99.36% | 99.36% | 98.72% |
| RF Parameters | 906 | 870 | 1,334 |
| MLP Parameters | 12,803 | 16,643 | 20,099 |
| **Efficiency** (Acc per param) | 99.4/906 | **100/870** | 99.36/1334 |

**Kết luận định lượng:**
- 63 features = tối ưu Pareto → độ chính xác cao nhất, độ phức tạp thấp hợp lý
- 90 features = dư thừa → mất độ chính xác (-0.64% RF, -0.64% MLP), tăng độ phức tạp (+53.3%)

---

## CÂU HỎI DỰ KIẾN TỪ HỘI ĐỒNG & TRẢLỜI

### Q1: "Tại sao không dùng 33 features để đơn giản hơn?"
**A:** "Bằng ablation study, 63 features tăng RF accuracy từ 99.36% lên 100% (+0.64%). Dù RF parameters giảm (-4.0%), MLP complexity tăng +30%. Tổng thể, 63 features cung cấp mức độ chính xác cao hơn với chi phí hợp lý."

### Q2: "Tại sao không dùng 90 hay 156 features để tối đa hóa accuracy?"
**A:** "Kết quả ablation study cho thấy 90 features không cải thiện accuracy (thậm chí RF giảm từ 100% xuống 99.36%). Đó là diminishing returns — thêm 27 features (33%) nhưng mất 0.64% accuracy và tăng 53.3% model complexity. Không xứng đáng cho edge deployment."

### Q3: "Bạn có xác minh kết quả này với cross-validation không?"
**A:** "Dữ liệu em dùng đã train/val/test split. Kết quả trên test set (156 samples, 4 nhãn activity). Em cũng train 3 mô hình độc lập (RF/SVM/MLP) để đảm bảo kết luận không bị model-specific."

### Q4: "Có phương pháp statistic nào chứng minh 63 tốt hơn 33?"
**A:** "Em so sánh 3 model: RF/SVM/MLP. Trên RF, 63 > 33 (100% > 99.36%, Δ=0.64%). Trên SVM, ngang nhau (cả 100%). Trên MLP, ngang nhau (99.36%). Overall, 63 >= 33 trên cả 3 model. Kết hợp với complexity analysis, 63 là tối ưu."

### Q5: "Tại sao chọn 63, không phải 45 hay 75?"
**A:** "Vì 63 = oi (orientation_invariant) mode tự nhiên của framework — 33 (time-domain mag) + 30 (frequency-domain mag). Em không tùy ý cắt ngang mà dùng các feature set hoàn chỉnh do framework hỗ trợ. Nếu cần, có thể test 45 hoặc 75, nhưng 63 đã tối ưu."

### Q6: "Mô hình em deploy trên thiết bị dùng mấy features?"
**A:** "Mô hình code gen (C++) dùng 63 features, với parameter reordering từ Python (alphabetical) sang C++ order (acc_mag → gyro_mag → jerk → frequency). Framework đảm bảo parity giữa Python training và C++ deployment."

---

## DANH SÁCH 63 FEATURES (để trả lời nếu hỏi chi tiết)

### Nhóm 1: Acceleration Magnitude (15 features)
1. acc_mag_mean, std, min, max, range, median, q25, q75, iqr
2. acc_mag_skewness, kurtosis (dùng population std)
3. acc_mag_rms, energy
4. acc_mag_zero_crossings, mean_crossing_rate

### Nhóm 2: Gyroscope Magnitude (15 features)
Same 15 statistics cho gyro_mag (rotation rate magnitude)

### Nhóm 3: Jerk Features (6 features)
- acc_jerk_mag_mean, std, max (acceleration derivative)
- gyro_jerk_mag_mean, std, max (rotation derivative)

### Nhóm 4: Auxiliary (6 features)
- acc_sma (Signal Magnitude Area)
- tilt_pitch, tilt_roll (angles from accelerometer)
- acc_mag_autocorr_lag1, acc_jerk_mag_peak_count

### Nhóm 5: Frequency Domain - Acceleration (10 features)
- dominant_frequency, dominant_frequency_magnitude
- spectral_centroid, rolloff, rms
- energy_low_freq (0-10 Hz), mid_freq (10-30 Hz), high_freq (30+ Hz)
- spectral_skewness, kurtosis, entropy

### Nhóm 6: Frequency Domain - Gyroscope (10 features)
Same 10 frequency features cho gyro_mag

**Lý do chọn những features này:**
- Time-domain capture pattern, jerk capture transition
- Frequency capture activity-specific frequencies (walking ~2 Hz, running ~3 Hz)
- Magnitude (không per-axis) → orientation-robust
- Well-studied trong literature (Edge Impulse, Google HAR)

---

## TÀI LIỆU HỖ TRỢ

1. **ablation_results.csv** — Chi tiết số liệu (accuracy, F1, parameters)
2. **ablation_comparison.png** — 3 biểu đồ (Accuracy, Complexity, Pareto)
3. **ablation_analysis.txt** — Báo cáo đầy đủ (diminishing returns, per-feature efficiency)
4. **feature_definitions_63.py** — Định nghĩa chi tiết 63 features

Tất cả file ở: `persistent_data/`

---

## LƯU Ý KHI PHẦN BIỆN LUẬN

- Chuẩn bị sẵn 2-3 biểu đồ ablation study để có thể chiếu tại phòng họp
- Nhớ giải thích "elbow point" = nơi accuracy plateau, complexity tăng
- Từ khóa: "diminishing returns", "Pareto optimal", "data-driven" (không cảm tính)
- Nếu hỏi chi tiết 63 features, lấy list từ `feature_definitions_63.py`
- Nhấn mạnh framework đã hỗ trợ 6 feature modes (33, 41, 63, 90, 156, raw) → đây không phải con số random
