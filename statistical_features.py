import numpy as np
from scipy.stats import skew, kurtosis

def extract_statistical_features(image):
    """ image có shape (128, 128, 3) """
    features = []
    # Loại bỏ các pixel đen (nền đã bị xóa bằng hàm background_cancellation)
    # Chỉ tính thống kê trên các pixel thuộc quả cà chua
    mask = np.any(image != [0, 0, 0], axis=-1)
    
    for channel in range(3):
        pixels = image[:, :, channel][mask]
        
        features.extend([
            np.mean(pixels),           # Mean
            np.std(pixels),            # Standard Deviation
            skew(pixels),              # Skewness
            kurtosis(pixels)           # Kurtosis
        ])
    return np.array(features) # Trả về vector 12 chiều