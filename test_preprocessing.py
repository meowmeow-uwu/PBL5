import cv2
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from preprocessing import background_cancellation
import numpy as np

def background_cancellation_inverse(image, img_size=299):
    """
    Tìm trực tiếp dải màu cà chua (chín/xanh), bỏ qua thanh kim loại trắng.
    Sử dụng kỹ thuật lấp đầy viền (Filled Contours) để vá đốm sáng phản quang.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # ---------------------------------------------------------
    # 1. LỌC TRỰC TIẾP MÀU CÀ CHUA (Bỏ qua màu trắng, đen, xanh lục của băng tải)
    # Yêu cầu Saturation > 60 để CHẮC CHẮN loại bỏ thanh kim loại trắng/xám.
    # ---------------------------------------------------------
    
    # Dải 1: Đỏ, Cam, Vàng (Cà chua chín / ương)
    lower_red1 = np.array([0, 60, 50])
    upper_red1 = np.array([30, 255, 255])
    
    lower_red2 = np.array([160, 60, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Dải 2: Xanh ngả vàng (Cà chua xanh). 
    # Băng tải là xanh lục lạnh (Hue thường > 50), ta chỉ lấy Hue 30-45.
    lower_green = np.array([30, 60, 50])
    upper_green = np.array([45, 255, 255])
    
    # Gộp các mask lại
    mask = cv2.inRange(hsv, lower_red1, upper_red1)
    mask |= cv2.inRange(hsv, lower_red2, upper_red2)
    mask |= cv2.inRange(hsv, lower_green, upper_green)
    
    # Lọc nhiễu nhỏ li ti (bụi trên băng tải)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # ---------------------------------------------------------
    # 2. TÌM VẬT THỂ & LẤP LỖ HỔNG (XỬ LÝ PHẢN QUANG)
    # ---------------------------------------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return cv2.resize(image, (img_size, img_size))
        
    c = max(contours, key=cv2.contourArea)
    
    # TẠO MẶT NẠ ĐẶC: Vẽ lại quả cà chua và lấp kín mọi lỗ thủng phản quang bên trong
    solid_mask = np.zeros_like(mask)
    cv2.drawContours(solid_mask, [c], -1, 255, thickness=cv2.FILLED)
    
    # ---------------------------------------------------------
    # 3. CẮT CÚP & PADDING THÀNH HÌNH VUÔNG
    # ---------------------------------------------------------
    x, y, w, h = cv2.boundingRect(c)
    
    H, W = image.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(W - x, w), min(H - y, h)
    
    cropped_roi = image[y:y+h, x:x+w]
    cropped_mask = solid_mask[y:y+h, x:x+w]  # Dùng solid_mask đã lấp lỗ
    
    # Đục nền
    roi_bg_removed = cv2.bitwise_and(cropped_roi, cropped_roi, mask=cropped_mask)
    
    # Padding viền đen để chống méo ảnh khi resize
    h_crop, w_crop = roi_bg_removed.shape[:2]
    max_side = max(h_crop, w_crop)
    
    top = (max_side - h_crop) // 2
    bottom = max_side - h_crop - top
    left = (max_side - w_crop) // 2
    right = max_side - w_crop - left
    
    squared_img = cv2.copyMakeBorder(
        roi_bg_removed, 
        top, bottom, left, right, 
        cv2.BORDER_CONSTANT, 
        value=[0, 0, 0]
    )
    
    final_img = cv2.resize(squared_img, (img_size, img_size))
    
    return final_img

def background_cancellation_robust(image, img_size=299):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 1. HẠ TIÊU CHUẨN SATURATION (Xuống 30) ĐỂ CỨU CÁC VÙNG CHÓI SÁNG
    # Dải Đỏ/Cam/Vàng/Nâu nhạt (Bao gồm cả nẹp gỗ/nhựa)
    lower_red = np.array([0, 30, 50])
    upper_red = np.array([30, 255, 255])
    
    lower_red_wrap = np.array([160, 30, 50])
    upper_red_wrap = np.array([180, 255, 255])
    
    # Dải Xanh nhạt/Vàng chanh (Mở rộng Hue để bắt cà chua xanh)
    lower_green = np.array([30, 30, 50])
    upper_green = np.array([60, 255, 255])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask |= cv2.inRange(hsv, lower_red_wrap, upper_red_wrap)
    mask |= cv2.inRange(hsv, lower_green, upper_green)
    
    # Dọn nhiễu bụi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 2. LỌC HÌNH HỌC VÀ TẠO BAO LỒI (CONVEX HULL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(image, (img_size, img_size))
        
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800:  # Bỏ qua mảng vỡ nhỏ
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        
        # LỌC ASPECT RATIO: Loại bỏ thanh nẹp (thường dài, aspect ratio < 0.4 hoặc > 2.5)
        # Quả cà chua hình tròn/oval sẽ nằm trong khoảng 0.5 đến 2.0
        if 0.5 <= aspect_ratio <= 2.0:
            valid_contours.append(cnt)
            
    if not valid_contours:
        return cv2.resize(image, (img_size, img_size))
        
    # Chọn vật thể to nhất trong số các vật thể ĐÃ ĐẠT CHUẨN hình dáng
    c = max(valid_contours, key=cv2.contourArea)
    
    # VŨ KHÍ BÍ MẬT: Tạo bao lồi căng ngang qua các vết mẻ do chói sáng
    hull = cv2.convexHull(c)
    
    # Vẽ mặt nạ đặc từ bao lồi
    solid_mask = np.zeros_like(mask)
    cv2.drawContours(solid_mask, [hull], -1, 255, thickness=cv2.FILLED)
    
    # 3. CẮT VÀ PADDING VUÔNG BẰNG BAO LỒI
    x, y, w, h = cv2.boundingRect(hull)
    H, W = image.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(W - x, w), min(H - y, h)
    
    cropped_roi = image[y:y+h, x:x+w]
    cropped_mask = solid_mask[y:y+h, x:x+w]
    
    roi_bg_removed = cv2.bitwise_and(cropped_roi, cropped_roi, mask=cropped_mask)
    
    h_crop, w_crop = roi_bg_removed.shape[:2]
    max_side = max(h_crop, w_crop)
    top = (max_side - h_crop) // 2
    bottom = max_side - h_crop - top
    left = (max_side - w_crop) // 2
    right = max_side - w_crop - left
    
    squared_img = cv2.copyMakeBorder(
        roi_bg_removed, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    return cv2.resize(squared_img, (img_size, img_size))

def background_cancellation_center_bias(image, img_size=299):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 1. TẠO MẶT NẠ MÀU VÀ LẤP LỖ HỔNG (Giữ nguyên như bản trước)
    lower_red = np.array([0, 30, 50])
    upper_red = np.array([30, 255, 255])
    lower_red_wrap = np.array([160, 30, 50])
    upper_red_wrap = np.array([180, 255, 255])
    lower_green = np.array([30, 30, 50])
    upper_green = np.array([60, 255, 255])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask |= cv2.inRange(hsv, lower_red_wrap, upper_red_wrap)
    mask |= cv2.inRange(hsv, lower_green, upper_green)
    
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)
    
    contours_temp, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_mask = np.zeros_like(mask)
    if contours_temp:
        for cnt in contours_temp:
            if cv2.contourArea(cnt) > 500:
                cv2.drawContours(solid_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    mask = solid_mask

    # ---------------------------------------------------------
    # 2. VŨ KHÍ TỐI THƯỢNG: CENTER BIAS (Khoảng cách tới tâm)
    # ---------------------------------------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(image, (img_size, img_size))
        
    H, W = image.shape[:2]
    center_x, center_y = W // 2, H // 2  # Tọa độ tâm của bức ảnh
    
    best_cnt = None
    min_dist = float('inf')
    
    for cnt in contours:
        # Bỏ qua các hạt bụi hoặc nhiễu quá nhỏ
        if cv2.contourArea(cnt) < 1000:
            continue
            
        # Tính Trọng tâm (Centroid) của khối màu bằng cv2.moments
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
            
        # Tính khoảng cách từ Trọng tâm vật thể tới Tâm bức ảnh
        dist = (cx - center_x)**2 + (cy - center_y)**2
        
        # Chọn vật thể CÓ KHOẢNG CÁCH GẦN TÂM NHẤT
        if dist < min_dist:
            min_dist = dist
            best_cnt = cnt
            
    # Nếu lọc xong không còn gì, trả về ảnh gốc
    if best_cnt is None:
        return cv2.resize(image, (img_size, img_size))
    
    # ---------------------------------------------------------
    # 3. CẮT CÚP & PADDING (Giữ nguyên)
    # ---------------------------------------------------------
    x, y, w, h = cv2.boundingRect(best_cnt)
    x, y = max(0, x), max(0, y)
    w, h = min(W - x, w), min(H - y, h)
    
    cropped_roi = image[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    
    roi_bg_removed = cv2.bitwise_and(cropped_roi, cropped_roi, mask=cropped_mask)
    
    h_crop, w_crop = roi_bg_removed.shape[:2]
    max_side = max(h_crop, w_crop)
    
    top = (max_side - h_crop) // 2
    bottom = max_side - h_crop - top
    left = (max_side - w_crop) // 2
    right = max_side - w_crop - left
    
    squared_img = cv2.copyMakeBorder(
        roi_bg_removed, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    return cv2.resize(squared_img, (img_size, img_size))


def test_single_image(img_path):
    print(f"Testing image: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not load the image. Please check the path.")
        return
        
    # Gọi hàm tiền xử lý (xóa phông)
    processed_img = background_cancellation_center_bias(img)
    
    # Convert BGR to RGB để hiển thị đúng màu bằng Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    
    # Tạo biểu đồ hiển thị Before - After
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Ảnh gốc (Original)")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(processed_rgb)
    plt.title("Sau khi tách nền & Cắt viền")
    plt.axis("off")
    
    out_path = "test_preprocessing_result.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Thành công! Ảnh so sánh đã được lưu tại: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Sử dụng: uv run python test_preprocessing.py <đường_dẫn_tới_ảnh>")
        print("Ví dụ: uv run python test_preprocessing.py '../dataset/Dataset_Cachua/Reject/20260417183959.jpg'")
    else:
        test_single_image(sys.argv[1])
