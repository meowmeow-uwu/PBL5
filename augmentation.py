"""
Data augmentation for the training set:
rotation, flip, shift, zoom.
"""

import numpy as np
from PIL import Image
from torchvision.transforms import v2

from config import AUGMENTATION_FACTOR


def get_dynamic_transform():
    """
    Returns a torchvision transform pipeline designed for PyTorch tensors
    representing images in [0, 1] format.
    """
    return v2.Compose([
        v2.RandomRotation(30),
        v2.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])

def get_balanced_indices(y_train):
    """
    Creates an array of indices that balances all classes to match the 
    size of the majority class (Oversampling).
    
    Args:
        y_train : np.ndarray (N,) int
        
    Returns:
        np.ndarray of indices
    """
    print("\n" + "=" * 60)
    print("STEP 3: Class Balancing (Index Oversampling)")
    print("=" * 60)
    
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    max_count = np.max(class_counts)
    
    print("  Original Class Distribution:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"    Class {cls}: {count} images")
        
    print(f"\n  Target balance: {max_count} images per class")
    
    indices = []
    
    for cls, count in zip(unique_classes, class_counts):
        deficit = max_count - count
        cls_indices = np.where(y_train == cls)[0]
        
        # Add original indices
        indices.extend(cls_indices)
        
        if deficit > 0:
            print(f"  -> Oversampling Class {cls}: adding {deficit} indices...")
            sampled_indices = np.random.choice(cls_indices, size=deficit, replace=True)
            indices.extend(sampled_indices)
        else:
            print(f"  -> Class {cls} is the majority class.")
            
    indices = np.array(indices)
    np.random.shuffle(indices)
    
    print(f"\n  Original Dataset Size:  {len(y_train)}")
    print(f"  Total Balanced Size:    {len(indices)}")
    
    return indices