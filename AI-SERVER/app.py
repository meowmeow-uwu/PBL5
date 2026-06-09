import os
import io
import time
import cv2
import json
import torch
import numpy as np
import requests
import threading
from flask import Flask, request, jsonify
from minio import Minio
from dotenv import load_dotenv

import sys
# Thêm thư mục gốc PBL5 vào sys.path để dùng chung code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import RESULTS_DIR, CLASS_NAMES, IMG_SIZE
from model import preprocess_input, CNN
from preprocessing import background_cancellation
import preprocessing
from statistical_features import extract_statistical_features
import joblib

# Tải các cấu hình từ tệp .env
load_dotenv()

app = Flask(__name__)

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

# Tải mô hình tốt nhất từ best_model_info.json
info_path = os.path.join(RESULTS_DIR, "best_model_info.json")
if not os.path.exists(info_path):
    print(f"[ERROR] Could not find {info_path}. Please run train_cachua_module.py first.")
    exit(1)

with open(info_path, 'r') as f:
    best_info = json.load(f)

MODEL_PATH = best_info['model_path']
MODEL_TYPE = best_info['model_type']
MODEL_COLOR_SPACE = best_info['color_space']

model = None
ml_pipeline = None

if MODEL_TYPE == "ml":
    print(f"[INFO] Loading ML model from: {MODEL_PATH}")
    ml_pipeline = joblib.load(MODEL_PATH)
    model = ml_pipeline['model']
    print(f"[INFO] Loaded ML model ({type(model).__name__}) successfully.")
else:
    print(f"[INFO] Loading CNN model from: {MODEL_PATH}")
    num_classes = len(CLASS_NAMES)
    model = CNN(num_classes).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("[INFO] Loaded CNN model successfully.")


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
        print(f"[BG LOG] [{request_id}] Success! Uploaded to MinIO + Called Backend (Total bg time: {t_end - t_start:.2f}s)")
    except Exception as e:
        print(f"[BG ERROR] [{request_id}] Background processing error: {e}")

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

        print(f"[LOG] [{request_id}] 1. Received image from ESP32.")
        
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
        
        # 3. Dự đoán nhãn
        if MODEL_TYPE == "ml":
            roi_cs = preprocessing.convert_color_spaces(roi_rgb)[MODEL_COLOR_SPACE]
            features = extract_statistical_features(roi_cs)
            features_sc = ml_pipeline['scaler'].transform([features])
            
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features_sc)[0]
                confidence = float(np.max(probs))
            else:
                confidence = 1.0 # Support Vector Machines without probability can't give proper confidence
                
            pred_idx = model.predict(features_sc)[0]
            predicted_class = ml_pipeline['le'].inverse_transform([pred_idx])[0]
            
        else: # CNN
            if MODEL_COLOR_SPACE != 'RGB':
                roi_rgb = preprocessing.convert_color_spaces(roi_rgb)[MODEL_COLOR_SPACE]
                
            batch_img = np.expand_dims(roi_rgb, axis=0) 
            batch_img_p = preprocess_input(batch_img)
            tensor_img = torch.tensor(batch_img_p).to(device)

            with torch.no_grad():
                outputs = model(tensor_img)
                probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
                
            predicted_idx = np.argmax(probs)
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence = float(probs[predicted_idx])
        
        # Chuyển đổi nhãn thành IN HOA để khớp giao thức của Arduino/ESP32
        result_label = predicted_class.upper()

        t_processed = time.time()
        print(f"[LOG] [{request_id}] 2. Classification done: {result_label} ({confidence*100:.1f}%) - Processing time: {t_processed - t_received:.2f}s")

        # 4. Kích hoạt luồng phụ để thực hiện upload MinIO & Gọi Backend API (Không chặn ESP32)
        bg_thread = threading.Thread(
            target=process_background_tasks,
            args=(file_bytes, request_id, result_label, confidence)
        )
        bg_thread.start()

        # 5. Trả kết quả lập tức cho ESP32-CAM
        t_response = time.time()
        print(f"[LOG] [{request_id}] 3. Returned result to ESP32. (HTTP response time: {t_response - t_received:.2f}s)")
        return jsonify({
            "id": str(request_id),
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
