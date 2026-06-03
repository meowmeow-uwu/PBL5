import numpy as np
import cv2

def background_cancellation(image, img_size=299):
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
