import numpy as np
import cv2

def background_cancellation(image):
    """
    Loại bỏ nền bằng phương pháp phân ngưỡng Otsu trên kênh màu Đỏ và Xanh lá,
    kết hợp các phép toán hình thái học (morphological operations) để lấy ROI quả cà chua.
    """
    _, green, red = cv2.split(image)  # OpenCV sử dụng hệ màu BGR

    # Phân ngưỡng Otsu
    _, mask_red = cv2.threshold(red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_green = cv2.threshold(green, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Kết hợp hai mặt nạ bằng phép toán OR
    combined = cv2.bitwise_or(mask_red, mask_green)

    # Đóng hình thái học (Morphological closing) để lấp các lỗ nhỏ bên trong
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_large, iterations=3)

    # Flood-fill để lấp đầy toàn bộ vùng rỗng bên trong quả
    flood = combined.copy()
    h, w = flood.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    combined = cv2.bitwise_or(combined, cv2.bitwise_not(flood))

    # Mở hình thái học (Morphological opening) để lọc bỏ nhiễu hạt bên ngoài
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small, iterations=2)

    # Trộn ảnh gốc với mặt nạ để trích xuất ảnh ROI
    mask_3ch = cv2.merge([combined, combined, combined])
    return cv2.bitwise_and(image, mask_3ch)
