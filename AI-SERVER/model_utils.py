import torch
import torch.nn as nn
import numpy as np
from torchvision import models

# Các hằng số mạng huấn luyện của PBL5
DROPOUT_1 = 0.5
DROPOUT_2 = 0.3
DENSE_UNITS = 512

class MobileNetV3Edge(nn.Module):
    def __init__(self, num_classes, fine_tune=False):
        super(MobileNetV3Edge, self).__init__()
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self.backbone = models.mobilenet_v3_small(weights=weights)
        
        # Freeze early layers for Transfer Learning, unless fine-tuning
        if not fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
            
        # Replace the final classification layer
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        
        # Make sure the new classifier requires gradients
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        for i in range(len(self.backbone.classifier) - 1):
            x = self.backbone.classifier[i](x)
        return x

class CustomCNN(nn.Module):
    def __init__(self, num_classes, has_dropout=True):
        super(CustomCNN, self).__init__()
        layers = []
        
        # Conv 1
        layers.extend([
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        if has_dropout:
            layers.append(nn.Dropout2d(0.1))
            
        # Conv 2
        layers.extend([
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        if has_dropout:
            layers.append(nn.Dropout2d(0.1))
            
        # Conv 3
        layers.extend([
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        if has_dropout:
            layers.append(nn.Dropout2d(0.2))
            
        # Conv 4
        layers.extend([
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        if has_dropout:
            layers.append(nn.Dropout2d(0.2))
            
        # Conv 5
        layers.extend([
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        ])
        
        self.features = nn.Sequential(*layers)
        
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
    Chuẩn hóa ảnh NumPy cho PyTorch.
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
