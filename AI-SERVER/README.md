# Tomato Quality Classification - AI Server (Flask & PyTorch)

Đây là máy chủ AI (AI Server) độc lập, được thiết kế để phục vụ việc nhận diện phân loại chất lượng cà chua (Chín, Xanh, Hỏng) từ hình ảnh gửi lên bởi board **ESP32-CAM** trên băng tải.

---

## 🚀 Các Tính Năng Nổi Bật

1.  **Phân Loại Chất Lượng Cà Chua (PyTorch)**: Tích hợp mô hình CNN tùy chỉnh (`CustomCNN`) chạy trên nền tảng PyTorch để phân loại 3 lớp chất lượng cà chua: `RIPE` (Chín), `UNRIPE` (Xanh), và `REJECT` (Hỏng).
2.  **Tách Nền Tự Động (Background Cancellation)**: Áp dụng thuật toán phân ngưỡng Otsu và phép toán hình thái học (Morphology) trên hai kênh màu Red/Green để loại bỏ ảnh nền băng tải trước khi đưa vào mô hình, tăng độ chính xác phân loại.
3.  **Xử Lý Đa Luồng Bất Đồng Bộ (Async Core)**:
    *   **Phản hồi tức thì**: Nhận diện ảnh và trả kết quả JSON về cho ESP32-CAM ngay lập tức để băng tải không bị dừng giật cục.
    *   **Xử lý ngầm (Background Thread)**: Tác vụ tải ảnh lên kho lưu trữ **MinIO** và gọi Web Backend API được đẩy vào luồng chạy ngầm độc lập để tối ưu hóa độ trễ mạng.
4.  **Hỗ Trợ HTTP Keep-Alive**: Tự động tính toán và phản hồi kèm tiêu đề `Content-Length` giúp ESP32-CAM duy trì và tái sử dụng kết nối TCP liên tục, tăng tốc độ gửi ảnh.

---

## 🛠 Yêu Cầu Hệ Thống (Prerequisites)

*   **Python**: Phiên bản `3.8` trở lên.
*   **Thư viện**: Cần cài đặt các gói Python sau:
    ```bash
    pip install flask minio torch torchvision opencv-python requests numpy python-dotenv
    ```

---

## ⚙️ Hướng Dẫn Cấu Hình (Configuration)

1.  Tạo tệp `.env` bằng cách sao chép từ tệp mẫu:
    ```bash
    copy .env.example .env
    ```
2.  Mở tệp `.env` và điều chỉnh các cấu hình mạng/mô hình phù hợp với hệ thống của bạn:
    *   `AI_HOST` & `AI_PORT`: Địa chỉ IP và cổng của Flask Server (mặc định cổng `5000`).
    *   `BACKEND_API_URL`: Địa chỉ Backend Web API nhận cập nhật sự kiện phân loại (mặc định cổng `8080`).
    *   `MINIO_ENDPOINT` & `MINIO_BUCKET`: Địa chỉ máy chủ MinIO và Bucket lưu trữ ảnh.
    *   `MODEL_PATH`: Đường dẫn tới tệp tin trọng số PyTorch `.pth` (ví dụ: `./transfer_cnn_best.pth`).

---

## 💻 Hướng Dẫn Chạy Server

Khởi chạy máy chủ AI bằng lệnh:
```bash
uv run app.py
# Hoặc: python app.py
```

Khi chạy thành công, màn hình sẽ hiển thị:
```text
[INFO] Using device: cuda (hoặc cpu)
[INFO] Loading model weights from: ./transfer_cnn_best.pth
[INFO] Model loaded successfully.
[INFO] Starting Flask Server on 0.0.0.0:5000...
```

---

## 🔌 Tài Liệu Tích Hợp API (API Reference)

### **Nhận Diện Cà Chua**
*   **Endpoint**: `/predict`
*   **Method**: `POST`
*   **Content-Type**: `multipart/form-data`

#### **Tham Số Đầu Vào (Request Body)**
| Key | Type | Description |
| :--- | :--- | :--- |
| `id` | `text (string)` | ID mã định danh duy nhất của quả cà chua |
| `image` | `file (binary)` | File ảnh chụp cà chua dạng JPEG/PNG |

#### **Dữ Liệu Phản Hồi Mẫu (Response Body)**
*   **Mã HTTP**: `200 OK`
*   **Định dạng**: `application/json`
```json
{
  "id": 1,
  "result": "RIPE",
  "confidence": 0.98
}
```

---

## 📂 Danh Sách Các Tệp Tin

*   `app.py`: Tệp khởi chạy Flask Server chính, điều phối API và luồng chạy ngầm.
*   `model_utils.py`: Định nghĩa lớp mô hình `CustomCNN` và các hàm tiền xử lý đầu vào của mô hình.
*   `preprocessing_utils.py`: Chứa hàm `background_cancellation` xử lý ảnh thô trước khi đưa vào mô hình AI.
*   `.env.example`: Tệp tin cấu hình mẫu.
