# Tomato Quality Classification (PyTorch & Scikit-Learn)

Hệ thống nhận diện và phân loại mức độ trưởng thành, chất lượng của cà chua chạy trên hệ thống nhúng/Edge AI hoặc máy tính cá nhân. Hệ thống đánh giá đồng thời hai phương pháp:
- **Học máy truyền thống (Machine Learning)**: Sử dụng các mô hình như SVM, Random Forest, K-NN, Gaussian Naive Bayes kết hợp với trích xuất đặc trưng thống kê 12D.
- **Học sâu (Deep Learning)**: Mạng CNN tự định nghĩa xây dựng bằng PyTorch.

Hệ thống tích hợp thuật toán phân tích màu đa không gian (RGB, HSV, CIE Lab, YCbCr), thuật toán tách nền Center Bias ưu việt và một máy chủ AI Server độc lập phục vụ cho việc tích hợp thực tế với băng chuyền thông qua ESP32-CAM.

Dự án sử dụng trình quản lý gói cực nhanh **uv** để cấu hình và chạy mọi luồng công việc.

---

## 📂 Cấu trúc Dự án

```text
.
├── train_cachua_module.py # Orchestrator chính: Huấn luyện đồng loạt ML & CNN trên 4 không gian màu
├── train_ml.py            # Huấn luyện mô hình Học máy truyền thống (SVM, RF, KNN, GNB) bằng GridSearchCV
├── train_cnn.py           # Huấn luyện mô hình Học sâu (CNN) tự định nghĩa
├── predict_module.py      # Phân đoán và suy luận nhãn trực tiếp từ một hình ảnh đơn lẻ (Tự động tải mô hình tốt nhất)
├── model.py               # Định nghĩa cấu trúc CNN và hàm tiền xử lý Tensor
├── preprocessing.py       # Cắt nền ảnh (Background Cancellation) nâng cao dùng OpenCV Center Bias
├── statistical_features.py# Rút trích 12 đặc trưng thống kê màu sắc cho học máy
├── reporting.py           # Sinh báo cáo phân loại, Confusion Matrix, AUC-ROC, bảng đánh giá
├── evaluation.py          # Hàm tính toán chỉ số đánh giá (Acc, F1, Precision, Specificity,...)
├── visualization.py       # Biểu đồ trực quan kết quả 
├── config.py              # Cấu hình tham số hyper-parameters mặc định
├── .env.example           # Tệp cấu hình biến môi trường mẫu
└── AI-SERVER/             # Thư mục máy chủ AI Flask API phục vụ suy luận thực tế (ESP32-CAM)
```

---

## ✨ Tính năng Nổi bật

1. **Thuật toán Tách Nền Center Bias (Background Cancellation)**:
   * Chuyển đổi sang hệ màu HSV để lọc dải màu Đỏ/Xanh lá và loại bỏ các nhiễu từ kim loại, ánh sáng phản xạ.
   * Sử dụng phép toán hình thái học (Morphological Operations) để lấp các vùng trống bên trong quả.
   * Áp dụng **Center Bias** để đo khoảng cách từ trọng tâm (Centroid) của các vật thể màu tới trung tâm bức ảnh, lọc bỏ hoàn toàn nhiễu từ biên ngoài băng chuyền và chỉ giữ lại quả cà chua ở tâm.

2. **So sánh Đa Không Gian Màu (Multi-Color Space Analysis)**:
   * Tự động huấn luyện, đánh giá và tìm ra mô hình hoạt động tốt nhất độc lập trên 4 không gian màu khác nhau: **RGB**, **HSV**, **CIE Lab**, và **YCbCr**.
   * Tính toán và xuất biểu đồ `Đóng góp Kênh Màu (Channel Contribution)` qua phương pháp Permutation Importance.

3. **Mạng Deep Learning & Machine Learning Tự Động Hóa**:
   * **Machine Learning**: Tìm kiếm siêu tham số tối ưu (Hyperparameter Tuning) tự động sử dụng `GridSearchCV` trên 12D đặc trưng.
   * **CNN**: Huấn luyện đầu cuối, tự động tính toán trọng số cân bằng lớp (Class Weights), tính toán ROC-AUC và Channel Importance.

4. **Zero-Config Prediction (Suy luận thông minh)**:
   * Module dự đoán (`predict_module.py` và `AI-SERVER`) tự động phân tích tệp cấu hình `best_model_info.json` sau khi train để tải lên "Mô Hình Tốt Nhất" (dù là CNN hay ML), áp dụng đúng không gian màu chuẩn, tính năng chuẩn bị ảnh trước mà không cần cấu hình tay.

5. **AI Server phục vụ Băng Chuyền Bất Đồng Bộ**:
   * Flask Server độc lập hỗ trợ nhận ảnh từ board ESP32-CAM qua form-data `/predict`.
   * Hỗ trợ tải tự động mô hình vô địch cuối cùng (Absolute Best Model) để dự đoán.
   * Xử lý luồng chạy ngầm bất đồng bộ (Background Thread) để upload ảnh lên MinIO và gọi Web Backend API, trả kết quả về ESP32 tức thì.

---

## 🛠 Hướng dẫn Sử dụng (Workflow)

Yêu cầu cài đặt công cụ **uv** để quản lý môi trường ảo.

### 0. Cài đặt Môi trường
Tự động đồng bộ hóa và cài đặt tất cả các thư viện (PyTorch, Torchvision, OpenCV, Scikit-Learn, Pandas...):
```bash
uv sync
```

Tạo cấu hình biến môi trường cục bộ:
```bash
copy .env.example .env
```
Mở tệp `.env` và tùy chỉnh đường dẫn thư mục dataset của bạn (`DATASET_CACHUA_DIR`).

---

### 1. Huấn luyện Mô hình Đánh Giá Toàn Diện (Orchestrator Pipeline)
Chạy kịch bản tự động tải dữ liệu, trích xuất đặc trưng, huấn luyện 4 mô hình ML và 1 mô hình CNN qua tất cả không gian màu, sau đó tổng hợp báo cáo ROC-AUC:
```bash
uv run train_cachua_module.py
```
* Báo cáo text chi tiết và biểu đồ lưu tại: `results/results_<color_space>/`
* Các bảng so sánh toàn diện lưu tại: `results/table_evaluation_results.txt` và `results/table_dominant_channel.txt`
* Thông tin Mô hình Tốt nhất (Vô địch) được lưu tự động tại: `results/best_model_info.json`

---

### 2. Phân đoán Trực tiếp (Inference)
Chạy dự đoán nhãn chất lượng của 1 hình ảnh đơn lẻ (Tự động tải mô hình Vô Địch từ bước 1):
```bash
uv run predict_module.py --image "test_sample.jpg"
```

---

## ⚡ AI Server Deployment
Để khởi chạy Flask server phục vụ ESP32-CAM gửi ảnh thực tế từ băng chuyền:
1. Chuyển vào thư mục server:
   ```bash
   cd AI-SERVER
   ```
2. Cấu hình các biến trong `AI-SERVER/.env` (Tạo từ mẫu `copy .env.example .env`).
3. Khởi động server:
   ```bash
   uv run app.py
   ```
Server sẽ chạy mặc định tại cổng `5000` sẵn sàng nhận dữ liệu tại endpoint `/predict`. Nó cũng sẽ tự động sử dụng chung pipeline và tải mô hình Vô Địch từ kết quả huấn luyện.
