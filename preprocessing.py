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

from config import DATASET_DIR, RESULTS_DIR, IMG_SIZE

def background_cancellation(image):
    """
    Remove background using Otsu thresholding on Red and Green channels,
    combined with morphological operations.

    Steps:
      1. Extract Red and Green channels
      2. Otsu threshold each channel -> binary mask
      3. Combine masks (OR)
      4. Morphological closing + flood-fill to fill holes
      5. Morphological opening to remove small noise
      6. Multiply mask with original image -> ROI
    """
    _, green, red = cv2.split(image)  # OpenCV = BGR

    # Otsu thresholding
    _, mask_red = cv2.threshold(red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_green = cv2.threshold(green, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Combine with OR
    combined = cv2.bitwise_or(mask_red, mask_green)

    # Morphological closing (fill small holes)
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_large, iterations=3)

    # Flood-fill remaining holes
    flood = combined.copy()
    h, w = flood.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    combined = cv2.bitwise_or(combined, cv2.bitwise_not(flood))

    # Morphological opening (remove small noise)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small, iterations=2)

    # Apply mask
    mask_3ch = cv2.merge([combined, combined, combined])
    return cv2.bitwise_and(image, mask_3ch)

def load_and_preprocess_images(dataset_dir=DATASET_DIR, img_size=IMG_SIZE,
                                save_samples=True):
    """
    Load images from each class folder, apply background cancellation,
    resize to (img_size x img_size), and convert BGR -> RGB.

    Returns:
        images  – np.ndarray of shape (N, img_size, img_size, 3), dtype uint8
        labels  – list of string labels
    """
    class_dirs = {
        'Reject': os.path.join(dataset_dir, 'Reject'),
        'Ripe':   os.path.join(dataset_dir, 'Ripe'),
        'Unripe': os.path.join(dataset_dir, 'Unripe'),
    }

    print("=" * 60)
    print("STEP 1: Loading & Preprocessing (Background Cancellation)")
    print("=" * 60)

    images, labels = [], []
    samples = {}

    for cls, path in class_dirs.items():
        if not os.path.exists(path):
            print(f"  [WARNING] Not found: {path}")
            continue

        files = [f for f in os.listdir(path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  {cls}: {len(files)} images ... ", end="")

        for fname in files:
            img = cv2.imread(os.path.join(path, fname))
            if img is None:
                continue

            roi = background_cancellation(img)
            roi = cv2.resize(roi, (img_size, img_size))
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            images.append(roi_rgb)
            labels.append(cls)

            if cls not in samples:
                orig = cv2.cvtColor(cv2.resize(img, (img_size, img_size)),
                                    cv2.COLOR_BGR2RGB)
                samples[cls] = {'original': orig, 'preprocessed': roi_rgb}

        print("[OK]")

    if save_samples and samples:
        _save_preprocessing_samples(samples)

    images = np.array(images, dtype=np.uint8)
    print(f"\n  Total: {len(images)} images, shape={images[0].shape}")
    return images, labels


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
