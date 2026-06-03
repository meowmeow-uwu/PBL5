import torch
import torch.nn as nn
import numpy as np

# Các hằng số mạng huấn luyện của PBL5
DROPOUT_1 = 0.5
DROPOUT_2 = 0.3
DENSE_UNITS = 512

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(DROPOUT_1),
            nn.Linear(512, DENSE_UNITS),
            nn.ReLU(),
            nn.BatchNorm1d(DENSE_UNITS),
            nn.Dropout(DROPOUT_2),
            nn.Linear(DENSE_UNITS, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
        
    def extract_features(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

def preprocess_input(X):
    """
    Chuẩn hóa ảnh NumPy cho PyTorch CustomCNN.
    Hỗ trợ cả ảnh đơn lẻ (H, W, 3) hoặc một batch ảnh (N, H, W, 3).
    """
    if X.ndim == 3:
        # Ảnh đơn
        X = X.astype(np.float32) / 255.0
        X = np.transpose(X, (2, 0, 1))
        X = (X - 0.5) / 0.5
    else:
        # Batch ảnh
        X = X.astype(np.float32) / 255.0
        X = np.transpose(X, (0, 3, 1, 2))
        X = (X - 0.5) / 0.5
    return X
