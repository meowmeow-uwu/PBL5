import os
import io
import time
import cv2
import torch
import numpy as np
import requests
import threading
from flask import Flask, request, jsonify
from minio import Minio
from dotenv import load_dotenv

# Tải các cấu hình từ tệp .env
load_dotenv()

# Import cục bộ từ các tệp tiện ích trong thư mục AI-SERVER
from model_utils import CustomCNN, preprocess_input
from preprocessing_utils import background_cancellation

app = Flask(__name__)

# --- CẤU HÌNH ---
# Đường dẫn trọng số model (mặc định trỏ ra kết quả train ở thư mục ngoài nếu chạy trong gốc)
MODEL_PATH = os.getenv("MODEL_PATH", "../results/transfer_save_model/transfer_cnn_best.pth")

# Cấu hình kích thước ảnh đầu vào và danh sách các lớp nhãn nhận diện
IMG_SIZE = int(os.getenv("IMG_SIZE", "299"))
CLASS_NAMES = os.getenv("CLASS_NAMES", "Reject,Ripe,Unripe").split(",")

# Cấu hình Web Backend API
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8080/api/fruit")

# Cấu hình lưu trữ ảnh MinIO
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "12345678")
BUCKET_NAME = os.getenv("MINIO_BUCKET", "tomato-images")

# Khởi tạo MinIO Client
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# Tạo bucket lưu ảnh nếu chưa tồn tại
try:
    if not minio_client.bucket_exists(BUCKET_NAME):
        minio_client.make_bucket(BUCKET_NAME)
        print(f"[INFO] Created MinIO Bucket: {BUCKET_NAME}")
except Exception as e:
    print(f"[WARNING] Could not check/create MinIO Bucket: {e}")

# Thiết lập thiết bị chạy mô hình (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

# Tải mô hình PyTorch CustomCNN
num_classes = len(CLASS_NAMES)
model = CustomCNN(num_classes).to(device)

if os.path.exists(MODEL_PATH):
    print(f"[INFO] Loading model weights from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("[INFO] Model loaded successfully.")
else:
    print(f"[ERROR] Model path not found: {MODEL_PATH}")

# --- XỬ LÝ NỀN (BACKGROUND THREAD) ---
def process_background_tasks(image_bytes, request_id, result_label, confidence):
    """
    Task phụ chạy ngầm: Tải ảnh lên MinIO và gọi API thông báo cho Web Backend.
    """
    try:
        t_start = time.time()
        filename = f"{request_id}.jpg"
        image_stream = io.BytesIO(image_bytes)
        
        # 1. Upload ảnh lên MinIO
        minio_client.put_object(
            BUCKET_NAME, filename, image_stream, len(image_bytes), content_type="image/jpeg"
        )
        image_url = f"http://{MINIO_ENDPOINT}/{BUCKET_NAME}/{filename}"
        
        # 2. Tạo dữ liệu payload gửi lên Web Backend API
        backend_payload = {
            "id": request_id,
            "result": result_label,
            "imageUrl": image_url,
            "confidence": round(confidence, 2)
        }
        
        # 3. HTTP POST sang Web Backend
        requests.post(BACKEND_API_URL, json=backend_payload, timeout=5)
        
        t_end = time.time()
        print(f"[BG LOG] [{request_id}] Thành công! Upload MinIO + Gọi Backend xong (Tổng thời gian chạy ngầm: {t_end - t_start:.2f}s)")
    except Exception as e:
        print(f"[BG ERROR] [{request_id}] Lỗi xử lý ngầm: {e}")

# --- API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        t_received = time.time()
        
        # 1. Nhận ID và File ảnh từ request (form-data)
        request_id = request.form.get('id')
        file = request.files.get('image')
        
        if not file or not request_id:
            return jsonify({"error": "Missing id or image"}), 400

        print(f"[LOG] [{request_id}] 1. Nhận được ảnh từ ESP32.")
        
        # Đọc dữ liệu nhị phân của ảnh
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image file"}), 400

        # 2. Tiền xử lý ảnh (Tách nền và resize chuẩn hóa)
        roi = background_cancellation(img)
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Đóng gói dữ liệu đưa vào PyTorch Tensor
        batch_img = np.expand_dims(roi_rgb, axis=0) 
        batch_img_p = preprocess_input(batch_img)
        tensor_img = torch.tensor(batch_img_p).to(device)

        # 3. Dự đoán nhãn
        with torch.no_grad():
            outputs = model(tensor_img)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
            
        predicted_idx = np.argmax(probs)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(probs[predicted_idx])
        
        # Chuyển đổi nhãn thành IN HOA để khớp giao thức của Arduino/ESP32
        result_label = predicted_class.upper()

        t_processed = time.time()
        print(f"[LOG] [{request_id}] 2. Phân loại xong: {result_label} ({confidence*100:.1f}%) - Thời gian xử lý: {t_processed - t_received:.2f}s")

        # 4. Kích hoạt luồng phụ để thực hiện upload MinIO & Gọi Backend API (Không chặn ESP32)
        bg_thread = threading.Thread(
            target=process_background_tasks,
            args=(file_bytes, request_id, result_label, confidence)
        )
        bg_thread.start()

        # 5. Trả kết quả lập tức cho ESP32-CAM
        t_response = time.time()
        print(f"[LOG] [{request_id}] 3. Trả kết quả về ESP32. (Thời gian phản hồi HTTP: {t_response - t_received:.2f}s)")
        return jsonify({
            "id": int(request_id),
            "result": result_label,
            "confidence": round(confidence, 2)
        }), 200

    except Exception as e:
        print(f"[ERROR] Exception during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Đọc cấu hình Port và Host từ file .env
    host = os.getenv("AI_HOST", "0.0.0.0")
    port = int(os.getenv("AI_PORT", 5000))
    print(f"[INFO] Starting Flask Server on {host}:{port}...")
    app.run(host=host, port=port, debug=False)
