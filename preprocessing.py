"""
Image preprocessing: background cancellation using Otsu thresholding
on Red/Green channels + morphological operations to extract ROI.
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from config import DATASET_DIR, RESULTS_DIR, IMG_SIZE


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

def convert_color_spaces(roi_rgb):
    """ Nhận ảnh RGB đã cắt nền, trả về 4 định dạng """
    roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
    return {
        'RGB': roi_rgb,
        'HSV': cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV),
        'LAB': cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB),
        'YCrCb': cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    }

def _process_single_image(args):
    """Helper for parallel processing."""
    path, fname, cls, img_size = args
    img = cv2.imread(os.path.join(path, fname))
    if img is None:
        return None, None, None, None
    
    roi = background_cancellation(img)
    roi = cv2.resize(roi, (img_size, img_size))
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # Also return a resized original for sample visualization
    orig_resized = cv2.cvtColor(cv2.resize(img, (img_size, img_size)), cv2.COLOR_BGR2RGB)
    
    return roi_rgb, cls, orig_resized, fname

def load_and_preprocess_images(dataset_dir=DATASET_DIR, img_size=IMG_SIZE,
                                save_samples=True):
    """
    Load images from each class folder, apply background cancellation,
    resize to (img_size x img_size), and convert BGR -> RGB.

    Returns:
        images  – np.ndarray of shape (N, img_size, img_size, 3), dtype uint8
        labels  – list of string labels
        fnames  – list of string filenames
    """
    class_dirs = {
        'Reject': os.path.join(dataset_dir, 'Reject'),
        'Ripe':   os.path.join(dataset_dir, 'Ripe'),
        'Unripe': os.path.join(dataset_dir, 'Unripe'),
    }

    print("=" * 60)
    print("STEP 1: Loading & Preprocessing (Background Cancellation)")
    print("=" * 60)

    tasks = []
    for cls, path in class_dirs.items():
        if not os.path.exists(path):
            print(f"  [WARNING] Not found: {path}")
            continue
        
        files = [f for f in os.listdir(path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  {cls}: {len(files)} images ...")
        for fname in files:
            tasks.append((path, fname, cls, img_size))

    print(f"  Processing {len(tasks)} images in parallel ... ", end="", flush=True)
    
    images, labels, fnames = [], [], []
    samples = {}
    
    with ProcessPoolExecutor() as executor:
        for roi_rgb, cls, orig_resnet, fname in executor.map(_process_single_image, tasks):
            if roi_rgb is not None:
                images.append(roi_rgb)
                labels.append(cls)
                fnames.append(fname)
                if cls not in samples:
                    samples[cls] = {'original': orig_resnet, 'preprocessed': roi_rgb}

    print("[OK]")

    if save_samples and samples:
        _save_preprocessing_samples(samples)

    images = np.array(images, dtype=np.uint8)
    print(f"\n  Total: {len(images)} images, shape={images[0].shape}")
    return images, labels, fnames


def _save_preprocessing_samples(samples):
    """Save a side-by-side comparison of original vs preprocessed images."""
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    fig.suptitle("Background Cancellation Results", fontsize=14, fontweight='bold')

    for idx, (cls, imgs) in enumerate(samples.items()):
        axes[idx, 0].imshow(imgs['original'])
        axes[idx, 0].set_title(f"{cls} – Original")
        axes[idx, 0].axis('off')
        axes[idx, 1].imshow(imgs['preprocessed'])
        axes[idx, 1].set_title(f"{cls} – After Background Cancellation")
        axes[idx, 1].axis('off')

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "preprocessing_samples.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Sample visualization saved to {out}")
