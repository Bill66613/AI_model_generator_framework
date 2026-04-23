# ĐIỂM KHÁC BIỆT CỐT LÕI CỦA FRAMEWORK

## Mục đích tài liệu

Tài liệu này tổng hợp các điểm khác biệt cốt lõi của framework so với các nền tảng thương mại như Edge Impulse và SensiML. Trọng tâm không chỉ nằm ở việc "có thể huấn luyện mô hình" hay "có thể sinh mã", mà ở khả năng kiểm soát toàn bộ chuỗi xử lý từ dữ liệu thô đến mã triển khai trên thiết bị biên, đồng thời duy trì tính tương đồng giữa pha huấn luyện và pha suy luận trên thiết bị.

Các luận điểm dưới đây được rút ra từ mã nguồn hiện tại của hệ thống và từ các phát hiện kỹ thuật đã được ghi nhận trong `TECHNICAL_FINDINGS.md`, đặc biệt là các Finding 1, 2, 6, 7, 8, 10, 11, 12, 13 và 14.

---

## 1. Sinh mã triển khai độc lập và minh bạch hoàn toàn

### 1.1. Điểm khác biệt chính

Ưu điểm nổi bật nhất của framework là khả năng sinh ra mã triển khai độc lập (standalone deployment code), trong đó toàn bộ logic tiền xử lý, trích xuất đặc trưng, chuẩn hóa và suy luận được đóng gói trực tiếp vào mã C/C++ hoặc MicroPython. Điều này tạo ra khác biệt đáng kể so với các nền tảng thương mại vốn thường ẩn một phần pipeline dưới dạng SDK hoặc runtime đóng.

### 1.2. Giá trị thực tiễn

Framework không chỉ xuất ra một mô hình đã huấn luyện, mà xuất ra một gói triển khai hoàn chỉnh gồm:

- Mã suy luận cho mô hình.
- Mã trích xuất đặc trưng tương ứng với pipeline đã dùng khi huấn luyện.
- Các tham số chuẩn hóa đặc trưng.
- Tệp ví dụ tích hợp với nền tảng đích.

Đối với hướng triển khai trực tiếp, mã được sinh ra có thể hoạt động mà không cần thư viện học máy ngoài. Đây là lợi thế quan trọng trong bối cảnh thiết bị biên có bộ nhớ và năng lực tính toán hạn chế, đồng thời giúp người phát triển kiểm tra chính xác mô hình đang làm gì trên thiết bị.

### 1.3. Bằng chứng trong mã nguồn

- `deployment/base_generator.py`: sinh logic chuẩn hóa, trích xuất đặc trưng và khung suy luận C/C++.
- `deployment/code_generator_factory.py`: định tuyến generator theo loại mô hình và nền tảng.
- `deployment/neural_network_generator.py`, `deployment/random_forest_generator.py`, `deployment/svm_generator.py`, `deployment/cnn_generator.py`: hiện thực riêng theo từng họ mô hình.
- `deployment/micropython_generator.py`: sinh mã MicroPython độc lập, không phụ thuộc NumPy.

### 1.4. So sánh với nền tảng thương mại

Khác với Edge Impulse hoặc SensiML, nơi người dùng thường nhận được artifact triển khai đi kèm một SDK hoặc một pipeline nội bộ khó kiểm chứng, framework này cho phép kiểm tra trực tiếp từng mảng trọng số, từng công thức thống kê và từng bước suy luận. Hệ quả là khi xảy ra sai lệch giữa mô hình trong máy tính và mô hình trên thiết bị, người phát triển có thể truy vết tận gốc nguyên nhân thay vì chỉ quan sát hiện tượng đầu ra sai.

### 1.5. Luận điểm bảo vệ

Nếu xem triển khai edge AI là một bài toán kỹ thuật hệ thống thay vì chỉ là bài toán huấn luyện mô hình, thì khả năng sinh mã độc lập và minh bạch là một đóng góp có giá trị học thuật và thực tiễn, vì nó biến phần triển khai từ "hộp đen" thành đối tượng có thể phân tích, xác minh và tối ưu.

---

## 2. Trích xuất đặc trưng là thành phần hạng nhất của framework, không phải bước phụ trợ

### 2.1. Điểm khác biệt chính

Trong framework này, trích xuất đặc trưng (feature extraction) không bị xem là thao tác phụ sau tiền xử lý, mà là một thành phần trung tâm được thiết kế có hệ thống, có nhiều chế độ hoạt động, có ràng buộc tương đồng giữa Python và mã triển khai, và có thể kiểm chứng công thức đến từng đặc trưng.

### 2.2. Độ phong phú của không gian đặc trưng

Framework hiện hỗ trợ sáu chế độ đặc trưng:

- `orientation_invariant_time_only`: 33 đặc trưng.
- `orientation_invariant`: 53 đặc trưng.
- `time_domain`: 90 đặc trưng.
- `all`: 156 đặc trưng.
- `frequency_domain`: 66 đặc trưng.
- `raw`: 6 đặc trưng.

Điểm đáng chú ý là hệ thống không chỉ tính các thống kê cơ bản như trung bình hay độ lệch chuẩn, mà còn hỗ trợ đầy đủ bộ 15 thống kê trên mỗi tín hiệu, bao gồm `mean`, `std`, `min`, `max`, `range`, `median`, `q25`, `q75`, `iqr`, `skewness`, `kurtosis`, `rms`, `energy`, `zero_crossings`, `mean_crossing_rate`.

Điều này cho phép framework bao phủ cả biểu diễn miền thời gian lẫn miền tần số, đồng thời linh hoạt giữa hai hướng tiếp cận:

- Dùng đặc trưng theo từng trục cảm biến.
- Dùng đặc trưng bất biến theo hướng đặt cảm biến (orientation-invariant).

### 2.3. Đặc trưng bất biến theo hướng đặt cảm biến

Đây là một điểm mạnh mang tính thực dụng rất cao đối với bài toán nhận dạng hoạt động người dùng trong điều kiện triển khai thực. Thay vì phụ thuộc tuyệt đối vào việc cảm biến luôn được gắn đúng hướng, framework xây dựng các đại lượng như:

- Độ lớn gia tốc đã khử thành phần trung bình theo cửa sổ.
- Độ lớn vận tốc góc đã khử thành phần trung bình theo cửa sổ.
- Độ lớn jerk gia tốc.

Thiết kế này giúp giảm độ nhạy với việc thiết bị bị xoay, nghiêng hoặc đặt lệch giữa các lần thu thập. So với các pipeline thương mại vốn thường gom toàn bộ bước xử lý vào các block có sẵn, cách tiếp cận của framework cho phép người nghiên cứu hiểu chính xác vì sao một đặc trưng lại bền vững hơn trước thay đổi tư thế đeo thiết bị.

### 2.4. Tính kiểm chứng được của công thức

Một giá trị khác biệt lớn là mọi công thức đều có thể kiểm tra trực tiếp trong mã nguồn. Việc này dẫn tới các phát hiện kỹ thuật quan trọng:

- Finding 2: phát hiện sai lệch công thức `skewness` và `kurtosis` giữa Python và C++ do khác biệt giữa population standard deviation và sample standard deviation.
- Finding 7: phát hiện sai thứ tự đặc trưng giữa DataFrame Python và thứ tự tính toán trong C++.
- Finding 13: cải tiến độ bền của đặc trưng tần số bằng quy trình loại DC, dùng Hann window và bổ sung thống kê phổ.

Nếu pipeline là hộp đen, các sai lệch này gần như không thể phát hiện hoặc không thể chứng minh một cách chặt chẽ.

### 2.5. Bằng chứng trong mã nguồn

- `utils/feature_extraction.py`: hiện thực đầy đủ các đặc trưng thời gian, tần số và orientation-invariant.
- `deployment/base_generator.py`: hiện thực lại phần lõi của các công thức để bảo đảm tương đồng khi triển khai.
- `deployment/micropython_generator.py`: bản đồng bộ cho hướng triển khai MicroPython.

### 2.6. Luận điểm bảo vệ

Điểm bán hàng ở đây không chỉ là "có nhiều đặc trưng", mà là framework xem đặc trưng như một tài sản kỹ thuật có thể định nghĩa, kiểm tra, tái sử dụng và triển khai nhất quán. Đây là lợi thế rõ ràng so với các nền tảng thương mại vốn ưu tiên tốc độ thao tác hơn là minh bạch công thức.

---

## 3. Pipeline xử lý tín hiệu rõ ràng, có cơ sở vật lý và tránh tạo artifact

### 3.1. Điểm khác biệt chính

Framework xây dựng pipeline xử lý tín hiệu (signal processing) với giả định rằng dữ liệu cảm biến là dữ liệu vật lý, do đó mọi thao tác tiền xử lý đều phải tránh tạo ra giá trị phi thực tế. Đây là khác biệt quan trọng so với cách tiếp cận thiên về "thử và chạy" thường thấy ở các công cụ đóng gói sẵn.

### 3.2. Xử lý làm sạch và lọc tín hiệu

Hệ thống hỗ trợ:

- Làm sạch dữ liệu thiếu.
- Loại bỏ ngoại lệ theo ngưỡng thống kê.
- Lọc thông thấp Butterworth.

Việc lọc tín hiệu không chỉ phục vụ trực quan hóa hay làm đẹp dữ liệu, mà đóng vai trò ổn định hóa đầu vào trước khi phân đoạn cửa sổ và trích xuất đặc trưng. Khi triển khai, framework còn có thể sinh mã bộ lọc IIR tương ứng cho thiết bị, giúp rút ngắn khoảng cách giữa giai đoạn xử lý offline và online.

### 3.3. Chiến lược đệm cửa sổ có ý nghĩa vật lý

Đây là một trong những điểm khác biệt mạnh nhất của framework và là luận điểm trực tiếp để so sánh với nền tảng thương mại.

Theo Finding 1, framework đã phát hiện rằng đệm số không (zero-padding) trên dữ liệu IMU có thể tạo ra giá trị `acc_mag = 0`, trong khi về mặt vật lý điều này không hợp lý vì độ lớn gia tốc luôn chịu ảnh hưởng của trọng lực. Từ đó, hệ thống chuyển sang chiến lược lặp giá trị biên (edge-value replication).

Lợi ích của chiến lược này gồm:

- Tránh tạo ra phân phối giả trong dữ liệu huấn luyện.
- Giữ các thống kê như `min`, `range`, `energy` gần hơn với thực tế.
- Làm giảm nguy cơ mô hình học phải artifact thay vì học tín hiệu hoạt động thực.

Điểm mạnh ở đây không chỉ là chọn một cách padding khác, mà là phát hiện được vì sao zero-padding gây sai lệch và chứng minh được hậu quả của nó trên mô hình triển khai.

### 3.4. Quy trình phân đoạn cửa sổ và chồng lắp

Framework cung cấp lựa chọn phân đoạn cửa sổ bằng tương tác trực quan và hỗ trợ cửa sổ trượt với tỉ lệ chồng lắp. Điều này hữu ích trong thực nghiệm vì cho phép cân bằng giữa số lượng mẫu huấn luyện, độ mượt của suy luận theo thời gian và tải tính toán trên thiết bị đích.

### 3.5. Bằng chứng trong mã nguồn

- `utils/data_processing.py`: làm sạch dữ liệu, lọc thông thấp Butterworth.
- `callbacks/preprocessing_callbacks.py`: pipeline tiền xử lý và tạo cửa sổ.
- `callbacks/feature_engineering_callbacks.py`: logic đệm cửa sổ bằng edge-value replication và chuẩn bị dữ liệu đặc trưng.

### 3.6. Luận điểm bảo vệ

So với các nền tảng thương mại, thế mạnh của framework không nằm ở việc cung cấp nhiều block xử lý hơn, mà ở chỗ mỗi bước xử lý đều gắn với một lập luận tín hiệu học và vật lý rõ ràng. Điều này đặc biệt quan trọng trong bài toán cảm biến đeo người, nơi artifact nhỏ ở giai đoạn đầu có thể phá hỏng toàn bộ kết quả triển khai.

---

## 4. Bảo đảm tương đồng giữa huấn luyện và triển khai là đóng góp kỹ thuật cốt lõi

### 4.1. Điểm khác biệt chính

Một selling point rất mạnh của framework là không xem huấn luyện và triển khai là hai giai đoạn tách rời. Thay vào đó, framework giải quyết trực diện bài toán training-deployment parity, tức là làm sao để đặc trưng, scaler, mô hình và logic hậu xử lý trên thiết bị phản ánh đúng những gì đã được dùng khi huấn luyện trên máy tính.

### 4.2. Các lỗi parity đã được phát hiện và sửa

Framework đã phát hiện và xử lý nhiều lỗi mà trong hệ đóng rất khó quan sát:

- Finding 1: zero-padding gây sai lệch đặc trưng.
- Finding 2: sai lệch công thức thống kê bậc cao.
- Finding 6: chuẩn hóa kép giữa bước feature engineering và training.
- Finding 7: sai thứ tự đặc trưng giữa Python và C++.
- Finding 8: tái trích xuất tham số làm mất hiệu lực bước reorder.
- Finding 9: bộ kiểm định mã sinh ra chưa phân biệt CNN với pipeline dựa trên feature.

Chỉ riêng việc hệ thống có khả năng tự phát hiện những lỗi này đã là một điểm khác biệt đáng kể so với công cụ thương mại. Quan trọng hơn, các lỗi này không chỉ được mô tả hiện tượng, mà đã được truy ra nguyên nhân gốc trong mã nguồn và được khắc phục có hệ thống.

### 4.3. Ý nghĩa học thuật và thực tiễn

Trong nhiều hệ edge AI, phần khó nhất không phải là đạt độ chính xác cao trên tập kiểm thử, mà là làm cho mô hình hoạt động đúng khi rời khỏi môi trường Python và chạy trên vi điều khiển. Framework này biến bài toán parity thành một đối tượng thiết kế rõ ràng. Đây là đóng góp có giá trị vì nó trả lời câu hỏi thực tế nhất của edge AI: tại sao mô hình tốt trên máy tính nhưng sai trên thiết bị, và làm sao sửa được điều đó.

### 4.4. Bằng chứng trong mã nguồn

- `deployment/code_generator_factory.py`: reorder tham số mô hình theo thứ tự đặc trưng C++.
- `deployment/validation.py`: kiểm tra tính hợp lệ của mã triển khai theo từng kiến trúc.
- `utils/edge_ml_model.py`: mô phỏng hành vi triển khai với confidence threshold và smoothing.

### 4.5. Luận điểm bảo vệ

Nếu Edge Impulse và SensiML mạnh ở việc rút ngắn thời gian tạo nguyên mẫu, thì framework này mạnh ở chỗ kiểm soát được độ đúng đắn khi chuyển từ nguyên mẫu sang thiết bị thật. Với đề tài luận văn, đây là sự khác biệt có giá trị nghiên cứu hơn là chỉ khác biệt về giao diện công cụ.

---

## 5. Hỗ trợ đa mô hình và đa nền tảng triển khai trong cùng một kiến trúc

### 5.1. Điểm khác biệt chính

Framework không bị khóa vào một loại mô hình hay một hệ sinh thái triển khai duy nhất. Cùng một kiến trúc tổng thể có thể hỗ trợ nhiều loại mô hình và nhiều đích triển khai khác nhau.

### 5.2. Đa mô hình

Framework hiện hỗ trợ:

- Random Forest.
- SVM.
- Neural Network kiểu MLP của scikit-learn.
- PyTorch MLP.
- PyTorch CNN 1D.

Khả năng này có ý nghĩa thực tế vì người dùng có thể đánh đổi giữa:

- Mức độ chính xác.
- Kích thước mô hình.
- Độ phức tạp suy luận.
- Mức phù hợp với thiết bị đích.

### 5.3. Đa nền tảng

Framework hỗ trợ các hướng triển khai trên:

- Arduino-compatible.
- ARM Cortex-M.
- ESP32 và ESP-IDF.
- Generic C/C++.
- Zephyr RTOS.
- MicroPython.

Ngoài hướng triển khai trực tiếp, hệ thống còn hỗ trợ backend TFLite Micro và ONNX Runtime. Điều này cho thấy framework không chỉ là một prototype cho một bo mạch cụ thể, mà là một kiến trúc sinh mã đa đích.

### 5.4. Ý nghĩa so với nền tảng thương mại

Nhiều nền tảng thương mại có thể hỗ trợ nhiều bo mạch, nhưng người dùng thường ít kiểm soát được cách pipeline được ánh xạ sang từng đích triển khai. Trong framework này, người dùng có thể nhìn thấy rõ generator nào được dùng, tham số nào được nhúng, mức tối ưu hóa nào đang áp dụng và tại sao kích thước mã hay độ chính xác thay đổi theo nền tảng.

### 5.5. Bằng chứng trong mã nguồn

- `deployment/code_generator_factory.py`: ma trận model-platform-generator.
- `deployment/arm_cortex_generator.py`, `deployment/zephyr_generator.py`, `deployment/micropython_generator.py`: tùy biến theo nền tảng.
- `deployment/tflite_generator.py`, `deployment/onnx_generator.py`: hỗ trợ backend thay thế.

### 5.6. Luận điểm bảo vệ

Điểm khác biệt ở đây không phải chỉ là "hỗ trợ nhiều thiết bị", mà là hỗ trợ nhiều thiết bị trong khi vẫn tái sử dụng cùng model artifact và vẫn duy trì một triết lý parity nhất quán. Đây là hướng tiếp cận mang tính framework thực thụ, thay vì chỉ là demo triển khai cho một bo mạch duy nhất.

---

## 6. Hỗ trợ độ tin cậy khi triển khai thực tế, không chỉ tối ưu độ chính xác offline

### 6.1. Điểm khác biệt chính

Framework đã đi xa hơn mô hình đánh giá truyền thống bằng cách đưa các cơ chế an toàn triển khai vào ngay trong pipeline, thay vì chỉ báo cáo accuracy trên tập test.

### 6.2. Confidence threshold cho trường hợp "không chắc chắn"

Theo Finding 10, hệ thống hỗ trợ confidence threshold để trả về trạng thái "unknown" khi độ tin cậy của mô hình không đủ cao. Đây là một cải tiến quan trọng cho bối cảnh thực tế, nơi người dùng có thể thực hiện hoạt động ngoài tập huấn luyện hoặc có dữ liệu cảm biến nhiễu, lệch hoặc chưa từng xuất hiện trước đó.

So với việc luôn ép mô hình phải chọn một lớp trong tập nhãn huấn luyện, cơ chế này giúp hệ thống hành xử an toàn hơn và trung thực hơn với mức độ chắc chắn của nó.

### 6.3. Majority-vote smoothing

Theo Finding 14, framework còn mô phỏng và hỗ trợ hậu xử lý bằng majority-vote smoothing trên chuỗi dự đoán. Điều này cho phép đánh giá mô hình sát với bối cảnh triển khai thời gian thực hơn, thay vì chỉ dùng một chỉ số accuracy độc lập theo từng cửa sổ.

### 6.4. Data augmentation có nhận thức theo lớp hoạt động

Theo Finding 11, framework bổ sung cơ chế tăng cường dữ liệu với bảo vệ riêng cho các hoạt động tĩnh. Đây là một chi tiết rất đáng giá vì augmentation nếu áp dụng ngây thơ có thể làm hỏng cấu trúc của các lớp ít biến thiên như đứng yên, ngồi hoặc nằm. Cách làm hiện tại cho thấy framework không chỉ "có augmentation", mà còn hiểu khi nào augmentation là có lợi và khi nào có thể phản tác dụng.

### 6.5. Bằng chứng trong mã nguồn

- `utils/edge_ml_model.py`: đánh giá deployment accuracy, confidence threshold, smoothing.
- `utils/data_augmentation.py`: jitter, scaling, rotation, time warp, permutation với bảo vệ cho lớp tĩnh.
- `callbacks/training_callbacks.py`, `callbacks/feature_engineering_callbacks.py`: kết nối augmentation và đánh giá vào giao diện người dùng.

### 6.6. Luận điểm bảo vệ

So với các nền tảng thiên về tối ưu hóa accuracy theo pipeline chuẩn, framework này thể hiện định hướng "deployment-first": mô hình không chỉ cần đúng trong notebook mà còn phải đáng tin khi chạy liên tục trên thiết bị thật.

---

## 7. Tính mở, chi phí thấp và giá trị nghiên cứu cao

### 7.1. Tính mở

Framework cho phép người dùng quan sát toàn bộ đường đi của dữ liệu và mô hình:

- Dữ liệu thô.
- Làm sạch và lọc.
- Cửa sổ dữ liệu.
- Đặc trưng.
- Scaler.
- Trọng số mô hình.
- Mã triển khai.

Tính mở này tạo điều kiện cho tái lập thí nghiệm, kiểm định lỗi, cải tiến thuật toán và phục vụ mục đích học thuật tốt hơn so với mô hình dịch vụ đóng.

### 7.2. Chi phí

So với các nền tảng thuê bao thương mại, framework có lợi thế chi phí rõ rệt vì người dùng có thể sử dụng và mở rộng trực tiếp từ mã nguồn hiện có. Với môi trường nghiên cứu, giáo dục hoặc phòng thí nghiệm nhỏ, đây là yếu tố rất thực tế.

### 7.3. Giá trị nghiên cứu

Điểm mạnh quan trọng nhất là framework đã tạo ra các phát hiện kỹ thuật thực sự, chứ không chỉ tái hiện lại quy trình của công cụ có sẵn. Các phát hiện về parity, padding, công thức thống kê, confidence threshold và augmentation cho thấy framework này là một đối tượng nghiên cứu có khả năng sinh ra tri thức mới, phù hợp với mục tiêu của một luận văn thạc sĩ.

---

## 8. Bảng tóm tắt so sánh

| Tiêu chí | Edge Impulse / SensiML | Framework của luận văn |
|----------|------------------------|-------------------------|
| Mức độ minh bạch của pipeline | Hạn chế, nhiều thành phần dạng hộp đen | Minh bạch toàn bộ từ dữ liệu đến mã triển khai |
| Sinh mã triển khai độc lập | Thường phụ thuộc SDK hoặc runtime riêng | Có thể sinh mã trực tiếp, nhúng trọng số và logic suy luận |
| Kiểm chứng công thức đặc trưng | Khó hoặc không thể kiểm tra chi tiết | Có thể kiểm tra trực tiếp trong mã nguồn Python và C/C++ |
| Xử lý tín hiệu có cơ sở vật lý | Người dùng ít kiểm soát nội bộ | Có chiến lược xử lý, lọc và padding được giải thích rõ |
| Đảm bảo parity training-deployment | Không thể xác minh đầy đủ | Có phát hiện lỗi, sửa lỗi và cơ chế validation |
| Hỗ trợ nhiều loại mô hình | Có nhưng ít minh bạch khi triển khai | RF, SVM, MLP, PyTorch MLP, PyTorch CNN |
| Hỗ trợ nhiều nền tảng đích | Có nhưng pipeline triển khai đóng | Kiến trúc generator đa đích, dễ phân tích và mở rộng |
| Hỗ trợ cơ chế an toàn triển khai | Thường không phải trọng tâm | Có confidence threshold, smoothing, deployment simulation |
| Giá trị nghiên cứu học thuật | Chủ yếu là công cụ thương mại | Có khả năng tạo ra phát hiện kỹ thuật mới |

---

## 9. Kết luận cô đọng để dùng trong báo cáo hoặc bảo vệ

Có thể tóm tắt điểm khác biệt cốt lõi của framework bằng một luận đề ngắn như sau:

> Framework này khác với các nền tảng edge AI thương mại ở chỗ nó không chỉ hỗ trợ huấn luyện và triển khai mô hình, mà còn làm cho toàn bộ chuỗi từ xử lý tín hiệu, trích xuất đặc trưng, chuẩn hóa, sinh mã và suy luận trên thiết bị trở nên minh bạch, kiểm chứng được và duy trì tương đồng với pha huấn luyện. Nhờ đó, framework vừa có giá trị triển khai thực tế, vừa có giá trị nghiên cứu học thuật.

Nếu cần rút gọn hơn nữa để dùng trong phần trình bày bảo vệ, có thể nhấn mạnh ba ý chính sau:

1. Sinh mã triển khai độc lập, minh bạch và đa đích.
2. Pipeline trích xuất đặc trưng và xử lý tín hiệu có thể kiểm chứng, tránh artifact và phù hợp dữ liệu IMU thực.
3. Giải quyết trực diện bài toán tương đồng giữa huấn luyện và triển khai, vốn là điểm yếu khó nhìn thấy ở các nền tảng hộp đen.
