# Tomato Quality Classification (PyTorch)

Hệ thống nhận diện và phân loại mức độ trưởng thành, chất lượng của cà chua bằng công nghệ **Deep Learning** chạy trên hệ thống nhúng/Edge AI (MobileNetV3Edge) hoặc Custom CNN tùy chỉnh phát triển từ đầu bằng PyTorch. Hệ thống tích hợp thuật toán xử lý ảnh số tiên tiến giúp tối ưu hóa khả năng loại bỏ nhiễu phông nền của băng chuyền và một máy chủ AI Server độc lập phục vụ cho việc tích hợp thực tế.

Dự án sử dụng trình quản lý gói cực nhanh **uv** để cấu hình và chạy mọi luồng công việc.

---

## 📂 Cấu trúc Dự án

```text
.
├── train_module.py      # Huấn luyện mô hình từ scratch trên Dataset ban đầu
├── transfer_module.py   # Huấn luyện Transfer Learning (Fine-tune) trên Dataset mới (Cà chua)
├── evaluate_model.py    # Đánh giá độ chính xác của mô hình tốt nhất trên tập dữ liệu Test mới
├── predict_module.py    # Phân đoán và suy luận nhãn chất lượng trực tiếp từ một hình ảnh đơn lẻ
├── model.py             # Định nghĩa cấu trúc CustomCNN, MobileNetV3Edge và Training Loop
├── preprocessing.py     # Cắt nền ảnh (Background Cancellation) nâng cao dùng OpenCV Center Bias
├── augmentation.py      # Tăng cường dữ liệu cân bằng động và lưu giữ bộ nhớ
├── evaluation.py        # Hàm tính toán chỉ số đánh giá (Acc, F1, Precision, Specificity,...)
├── visualization.py     # Biểu đồ trực quan kết quả (Confusion Matrix, biểu đồ so sánh)
├── config.py            # Cấu hình tham số hyper-parameters mặc định
├── .env.example         # Tệp cấu hình biến môi trường mẫu
└── AI-SERVER/           # Thư mục máy chủ AI Flask API phục vụ suy luận thực tế (ESP32-CAM)
```

---

## ✨ Tính năng Nổi bật

1. **Thuật toán Tách Nền Center Bias (Background Cancellation)**:
   * Chuyển đổi sang hệ màu HSV để lọc dải màu Đỏ/Xanh lá và loại bỏ các nhiễu từ kim loại, ánh sáng phản xạ.
   * Sử dụng phép toán hình thái học (Morphological Operations) để lấp các vùng trống bên trong quả.
   * Áp dụng **Center Bias** để đo khoảng cách từ trọng tâm (Centroid) của các vật thể màu tới trung tâm bức ảnh, lọc bỏ hoàn toàn nhiễu từ biên ngoài băng chuyền và chỉ giữ lại quả cà chua ở tâm.
   * Tự động cắt cúp (Crop) và đệm viền đen (Padding) để chuyển ảnh về dạng vuông giúp giữ nguyên tỷ lệ cấu trúc quả.

2. **Mạng Deep Learning Linh hoạt (MobileNetV3Edge & CustomCNN)**:
   * **MobileNetV3Edge**: Nhẹ, nhanh, tối ưu hóa cho Edge AI/thiết bị nhúng dựa trên MobileNetV3-Small.
   * **CustomCNN**: Mạng 5 khối tích chập tự định nghĩa cho độ chính xác cao.
   * Hỗ trợ tự động tính toán Class Weights để xử lý mất cân bằng dữ liệu (Class Imbalance).

3. **Resume Training tự động**:
   * Lưu trữ trạng thái Optimizer, Epoch và Learning Rate Scheduler sau mỗi epoch dưới tệp `_last.pth`.
   * Tự động khôi phục quá trình huấn luyện tiếp tục từ epoch bị ngắt nếu có sự cố xảy ra.

4. **Trình Đánh giá Mô hình Độc lập (`evaluate_model.py`)**:
   * Kiểm tra trực tiếp độ chính xác của các checkpoint trên một tập dữ liệu Test mới.
   * Tự động phát hiện kiến trúc mô hình (CustomCNN có/không có Dropout hoặc MobileNetV3Edge) của checkpoint để load trọng số tương thích.
   * Tính toán đầy đủ: Accuracy, Precision, Recall, F1-Score, và Specificity cho từng lớp (`Reject`, `Ripe`, `Unripe`).
   * Xuất báo cáo text chi tiết và vẽ Confusion Matrix trực quan.

5. **AI Server phục vụ Băng Chuyền Bất Đồng Bộ**:
   * Flask Server độc lập hỗ trợ nhận ảnh từ board ESP32-CAM qua form-data `/predict`.
   * Xử lý luồng chạy ngầm bất đồng bộ (Background Thread) để upload ảnh lên MinIO và gọi Web Backend API, giúp trả kết quả về ESP32 tức thì trong thời gian dưới 0.1 giây.

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
Mở tệp `.env` và tùy chỉnh đường dẫn thư mục dataset của bạn.

---

### 1. Huấn luyện Mô hình cơ bản (Base Training)
Dùng để huấn luyện mô hình từ đầu (Mặc định sử dụng `MobileNetV3Edge` trên nhánh này để đạt hiệu năng Edge AI tối ưu):
```bash
uv run train_module.py
```
* Trọng số tốt nhất được lưu tại: `results/train_save_model/mobilenet_best.pth`
* Biểu đồ lịch sử huấn luyện: `results/train_save_model/mobilenet_history.png`

---

### 2. Fine-tune Mô hình (Transfer Learning)
Thực hiện chuyển giao học máy trên tập dữ liệu cà chua mới (`DATASET_CACHUA_DIR`):
```bash
uv run transfer_module.py
```
* Hệ thống sẽ tự động tìm kiếm `results/train_save_model/mobilenet_best.pth` làm mô hình gốc để tinh chỉnh.
* Mô hình tốt nhất được lưu tại: `results/transfer_save_model/mobilenet_finetuned_best.pth`

---

### 3. Đánh giá Mô hình trên Tập dữ liệu Test (`evaluate_model.py`)
Kiểm tra độ chính xác trên tập Test bên ngoài. Bạn có thể định cấu hình đường dẫn qua các biến môi trường trong `.env` (`EVAL_MODEL_PATH`, `EVAL_TEST_DIR`) hoặc truyền trực tiếp qua CLI:

```bash
uv run evaluate_model.py --test_dir "C:\Path\To\Test_Set" --model_path "results/train_save_model/base_cnn_best.pth"
```

* **Kết quả đầu ra**:
  * Báo cáo đánh giá dạng text lưu tại: `results/test_evaluation/test_evaluation_report.txt`
  * Ma trận nhầm lẫn Confusion Matrix lưu tại: `results/test_evaluation/test_confusion_matrices.png`

---

### 4. Phân đoán Trực tiếp (Inference)
Chạy dự đoán nhãn chất lượng của 1 hình ảnh đơn lẻ:
```bash
uv run predict_module.py --image "test_sample.jpg" --model "results/train_save_model/base_cnn_best.pth"
```

---

## ⚡ AI Server Deployment
Để khởi chạy Flask server phục vụ ESP32-CAM gửi ảnh thực tế từ băng chuyền:
1. Chuyển vào thư mục server:
   ```bash
   cd AI-SERVER
   ```
2. Cấu hình các biến trong `AI-SERVER/.env` (MODEL_PATH, MINIO, BACKEND_API_URL).
3. Khởi động server:
   ```bash
   uv run app.py
   ```
Server sẽ chạy mặc định tại cổng `5000` sẵn sàng nhận dữ liệu tại endpoint `/predict`.
