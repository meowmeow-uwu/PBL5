# 🍅 Tomato Quality Classification & Real-time AI Server

Hệ thống nhận diện và phân loại mức độ chín, chất lượng quả cà chua bằng công nghệ **Mạng Neural Tích Chập (Custom CNN)** phát triển từ đầu bằng PyTorch, kết hợp với các bộ phân lớp Học máy (Machine Learning Classifiers) như SVM, Random Forest và KNN để đánh giá so sánh.

Dự án này tích hợp đầy đủ từ khâu huấn luyện mô hình (Training), chuyển giao tri thức (Transfer Learning), nhận diện hình ảnh đơn lẻ (Inference) cho tới việc cung cấp một **AI Server độc lập** kết nối trực tiếp với băng tải phân loại IoT (ESP32-CAM).

---

## 📂 Cấu Trúc Dự Án

```text
.
├── AI-SERVER/            # Module AI Server độc lập kết nối ESP32 & Web Backend
├── Dataset/              # Thư mục chứa tập dữ liệu gốc (3 phân lớp)
├── Dataset_Cachua/       # Thư mục chứa tập dữ liệu mới phục vụ Transfer Learning
├── results/              # Thư mục lưu kết quả huấn luyện (trọng số, biểu đồ)
├── config.py             # Cấu hình siêu tham số (Hyper-parameters)
├── model.py              # Định nghĩa lớp CustomCNN và vòng lặp huấn luyện PyTorch
├── preprocessing.py      # Tiền xử lý ảnh (Background Cancellation tách nền bằng OpenCV)
├── augmentation.py       # Tăng cường dữ liệu (Data Augmentation bằng Torchvision)
├── train_module.py       # Module huấn luyện mô hình CustomCNN cơ sở (Base model)
├── transfer_module.py    # Module Transfer Learning trên tập dữ liệu cà chua mới
├── predict_module.py     # Module nhận diện đơn lẻ cho một bức ảnh từ Terminal
├── classifiers.py        # Thử nghiệm trích xuất đặc trưng với các bộ phân lớp ML (SVM, RF, KNN)
├── visualization.py      # Xuất biểu đồ phân tích (Confusion Matrix, Loss/Acc curves)
├── .env.example          # Tệp cấu hình đường dẫn mẫu cho dự án gốc
└── README.md             # Hướng dẫn sử dụng dự án
```

---

## 🛠 Hướng Dẫn Cài Đặt (Installation)

### 1. Chuẩn bị môi trường
Dự án được quản lý gói bằng công cụ hiện đại `uv`. Bạn có thể cài đặt các dependencies tự động:

```bash
# Cài đặt toàn bộ môi trường ảo và thư viện thông qua uv
uv sync
```

Hoặc sử dụng cách truyền thống bằng `pip` (khuyên dùng tạo môi trường ảo trước):

```bash
# Tạo môi trường ảo
python -m venv .venv
# Kích hoạt môi trường ảo (Windows)
.venv\Scripts\activate
# Cài đặt các thư viện cần thiết
pip install torch torchvision opencv-python scikit-learn matplotlib requests flask minio python-dotenv
```

### 2. Cấu hình biến môi trường
Tạo tệp `.env` tại thư mục gốc bằng cách sao chép từ tệp mẫu:

```bash
copy .env.example .env
```

Mở tệp `.env` vừa tạo và chỉnh sửa các đường dẫn thư mục dữ liệu trên máy của bạn:
*   `DATASET_DIR`: Đường dẫn tới tập dữ liệu huấn luyện cơ sở (gồm 3 thư mục con `Reject`, `Ripe`, `Unripe`).
*   `DATASET_CACHUA_DIR`: Đường dẫn tới tập dữ liệu cà chua mới dùng để Transfer Learning.
*   `RESULTS_DIR`: Nơi lưu trữ trọng số mô hình và các biểu đồ phân tích (mặc định là `./results`).

---

## 💻 Hướng Dẫn Sử Dụng (Usage Workflow)

### **Bước 1: Huấn luyện mô hình cơ sở (Base Training)**
Chạy script huấn luyện để dạy mô hình `CustomCNN` phân loại trên tập dữ liệu cơ sở:

```bash
uv run train_module.py
# Hoặc: python train_module.py
```
*   **Đặc điểm**: Chương trình tự động lưu trọng số mô hình có độ chính xác cao nhất trên tập validation vào `results/train_save_model/base_cnn_best.pth`.
*   **Epoch Resuming (Tự khôi phục)**: Nếu tiến trình học bị ngắt đột ngột (mất điện, tắt terminal), bạn chỉ cần chạy lại lệnh trên. Chương trình sẽ tự động tải checkpoint `_last.pth` và tiếp tục huấn luyện từ epoch bị ngắt.
*   **Xuất đồ thị**: Biểu đồ Accuracy/Loss (`base_cnn_history.png`) được cập nhật trực tiếp sau mỗi epoch.

### **Bước 2: Huấn luyện chuyển giao (Transfer Learning)**
Huấn luyện mô hình trên tập dữ liệu cà chua mới (`DATASET_CACHUA_DIR`) bằng cách kế thừa các đặc trưng đã học từ mô hình cơ sở:

```bash
uv run transfer_module.py
# Hoặc: python transfer_module.py
```
*   Chương trình sẽ tự động lấy trọng số từ `base_cnn_best.pth`, đóng băng phần trích xuất đặc trưng và chỉ tối ưu hóa các lớp phân loại cuối cùng cho bài toán mới.
*   Kết quả lưu trữ tại `results/transfer_save_model/transfer_cnn_best.pth`.

### **Bước 3: Nhận diện ảnh đơn lẻ từ Terminal (Inference)**
Nếu bạn muốn kiểm tra nhanh kết quả nhận diện của mô hình đối với một bức ảnh cà chua đơn lẻ:

```bash
# Chạy dự đoán với mô hình mặc định (Transfer Learning model)
uv run predict_module.py --image "duong_dan_anh_ca_chua.jpg"

# Hoặc sử dụng lệnh python truyền thống với chỉ định tệp trọng số mô hình khác
python predict_module.py --image "duong_dan_anh_ca_chua.jpg" --model "results/train_save_model/base_cnn_best.pth"
```
*   Ảnh thô sẽ tự động được đưa qua hàm `background_cancellation` tách nền, chuẩn hóa kích thước thành 299x299 trước khi đưa vào PyTorch dự đoán.
*   Terminal sẽ hiển thị xác suất (probability) cụ thể của cả 3 nhãn.

---

## ⚡️ AI Server Kết Nối Hệ Thống Phân Loại IoT

Thư mục **`AI-SERVER`** chứa một máy chủ độc lập dùng để kết nối trực tiếp với board mạch **ESP32-CAM** trên băng tải phân loại thực tế.

*   **Tính năng chính**:
    *   Sử dụng **HTTP Keep-Alive** giúp duy trì kết nối mạng tốc độ cao với camera.
    *   Tích hợp đa luồng (**Multi-threading**): Phân loại ảnh bằng PyTorch và trả kết quả ngay lập tức cho camera để băng tải hoạt động ổn định. Tác vụ đẩy ảnh lên **MinIO** và gọi Web Backend API được chạy ngầm dưới nền.
*   **Cách sử dụng**:
    Xem chi tiết hướng dẫn chạy và cấu hình API tại [README.md của AI-SERVER](file:///e:/python_project/PBL5/AI-SERVER/README.md).
