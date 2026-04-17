"""
Data augmentation for the training set:
rotation, flip, shift, zoom.
"""

import numpy as np
from PIL import Image
from torchvision.transforms import v2

from config import AUGMENTATION_FACTOR


def create_augmented_data(X_train, y_train, factor=AUGMENTATION_FACTOR):
    """
    Augment training images using rotation, flipping, shifting, and zoom.

    Args:
        X_train : np.ndarray (N, H, W, 3) uint8
        y_train : np.ndarray (N,) int
        factor  : int – number of augmented copies per image

    Returns:
        X_aug, y_aug  (includes the originals)
    """
    print("\n" + "=" * 60)
    print("STEP 3: Data Augmentation (Training Set)")
    print("=" * 60)

    transform = v2.Compose([
        v2.ToImage(),
        v2.RandomRotation(30),
        v2.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ToPILImage()
    ])

    aug_images = list(X_train)
    aug_labels = list(y_train)
    added = 0

    for i in range(len(X_train)):
        img = X_train[i]
        for _ in range(factor):
            aug_img = np.array(transform(img), dtype=np.uint8)
            aug_images.append(aug_img)
            aug_labels.append(y_train[i])
            added += 1

    X_aug = np.array(aug_images, dtype=np.uint8)
    y_aug = np.array(aug_labels)

    print(f"  Original:  {len(X_train)}")
    print(f"  Added:     {added}")
    print(f"  Total:     {len(X_aug)}")

    return X_aug, y_aug
