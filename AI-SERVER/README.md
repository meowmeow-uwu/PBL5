# Tomato Quality Classification - AI Server (Flask)

Đây là máy chủ AI (AI Server) độc lập, được thiết kế để phục vụ việc nhận diện phân loại chất lượng cà chua (Chín, Xanh, Hỏng) từ hình ảnh gửi lên bởi board **ESP32-CAM** trên băng tải.

---

## 🚀 Các Tính Năng Nổi Bật

1.  **Hỗ Trợ Đa Mô Hình Động (Dynamic Model Loading)**: Tự động tải mô hình tốt nhất từ `best_model_info.json` (dù là kiến trúc Học Sâu CNN hay Học Máy truyền thống như SVM, Random Forest) để phân loại 3 lớp chất lượng cà chua: `RIPE` (Chín), `UNRIPE` (Xanh), và `REJECT` (Hỏng).
2.  **Dùng Chung Pipeline Của PBL5**: Server tự động liên kết và tái sử dụng nguyên bản các module cốt lõi từ thư mục dự án chính như `model.py`, `preprocessing.py`, `config.py` để đảm bảo độ đồng nhất 100% giữa lúc huấn luyện và khi chạy thực tế.
3.  **Tách Nền Tự Động (Background Cancellation)**: Áp dụng thuật toán lọc Center Bias để loại bỏ ảnh nền băng tải trước khi đưa vào mô hình, tăng độ chính xác phân loại.
4.  **Xử Lý Đa Luồng Bất Đồng Bộ (Async Core)**:
    *   **Phản hồi tức thì**: Nhận diện ảnh và trả kết quả JSON về cho ESP32-CAM ngay lập tức để băng tải không bị dừng giật cục.
    *   **Xử lý ngầm (Background Thread)**: Tác vụ tải ảnh lên kho lưu trữ **MinIO** và gọi Web Backend API được đẩy vào luồng chạy ngầm độc lập để tối ưu hóa độ trễ mạng.
5.  **Hỗ Trợ HTTP Keep-Alive**: Tự động tính toán và phản hồi kèm tiêu đề `Content-Length` giúp ESP32-CAM duy trì và tái sử dụng kết nối TCP liên tục, tăng tốc độ gửi ảnh.

---

## 🛠 Yêu Cầu Hệ Thống (Prerequisites)

*   **Python**: Phiên bản `3.12` trở lên.
*   **Quản Lý Gói**: Khuyến nghị dùng trình quản lý `uv` từ thư mục gốc để tự động đồng bộ hóa thư viện.

---

## ⚙️ Hướng Dẫn Cấu Hình (Configuration)

1.  Tạo tệp `.env` bằng cách sao chép từ tệp mẫu:
    ```bash
    copy .env.example .env
    ```
2.  Mở tệp `.env` và điều chỉnh các cấu hình mạng phù hợp với hệ thống của bạn:
    *   `AI_HOST` & `AI_PORT`: Địa chỉ IP và cổng của Flask Server (mặc định cổng `5000`).
    *   `BACKEND_API_URL`: Địa chỉ Backend Web API nhận cập nhật sự kiện phân loại (mặc định cổng `8080`).
    *   `MINIO_ENDPOINT` & `MINIO_BUCKET`: Địa chỉ máy chủ MinIO và Bucket lưu trữ ảnh.

---

## 💻 Hướng Dẫn Chạy Server

Khởi chạy máy chủ AI bằng lệnh `uv`:
```bash
uv run app.py
```

Khi chạy thành công, màn hình sẽ hiển thị:
```text
[INFO] Using device: cuda (hoặc cpu)
[INFO] Loading CNN/ML model from: ...
[INFO] Loaded model successfully.
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

*   `app.py`: Tệp khởi chạy Flask Server chính, điều phối API và luồng chạy ngầm. Tự động liên kết mã nguồn từ thư mục gốc.
*   `.env.example`: Tệp tin cấu hình môi trường mẫu.
