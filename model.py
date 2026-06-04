"""
Custom CNN Models and Training/Evaluation Loops.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models

from config import (
    LEARNING_RATE, DROPOUT_1, DROPOUT_2, DENSE_UNITS
)

# --- Dataset ---
class FruitDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, transform=None, indices=None, color_space='RGB'):
        self.images = images  # uint8 images (N, H, W, 3)
        self.labels = labels
        self.transform = transform
        self.indices = indices if indices is not None else np.arange(len(images))
        self.color_space = color_space

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = self.images[real_idx]
        
        if self.color_space != 'RGB':
            import preprocessing
            img = preprocessing.convert_color_spaces(img)[self.color_space]
            
        # Preprocess on the fly (uint8 -> float32)
        img = img.astype(np.float32) / 255.0
        # Transpose to (C, H, W) for PyTorch
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img)
        
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)
            
        # Normalize to [-1, 1] after augmentations that might expect [0,1]
        img_tensor = (img_tensor - 0.5) / 0.5
        
        if self.labels is not None:
            label = torch.tensor(self.labels[real_idx], dtype=torch.long)
            return img_tensor, label
        return img_tensor


# --- Loss Function ---
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss_unweighted = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss_unweighted)
        focal_term = (1 - pt) ** self.gamma
        
        ce_loss_weighted = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        focal_loss = focal_term * ce_loss_weighted
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

import torch
import torch.nn as nn

# 1. Khối Cơ chế Chú ý (Squeeze-and-Excitation)
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # Squeeze
        y = self.fc(y).view(b, c, 1, 1) # Excitation
        return x * y.expand_as(x)       # Re-weighting

# 2. Khối Tích chập siêu nhẹ (Depthwise Separable + SE)
class LightweightBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(LightweightBlock, self).__init__()
        # Depthwise Conv (Học Không gian)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise Conv (Học Màu sắc/Đặc trưng kênh)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Gắn Attention
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        x = self.se(x) # Nhấn mạnh các đặc trưng quan trọng
        return x

class CNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.15):
        super(CNN, self).__init__()
        
        # Lớp tiếp nhận ảnh ban đầu
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Các khối rút đặc trưng chính (Nhẹ và thông minh)
        self.features = nn.Sequential(
            LightweightBlock(32, 64, stride=2),
            LightweightBlock(64, 128, stride=2),
            LightweightBlock(128, 256, stride=2)
        )
        
        # Lớp đầu ra sử dụng GAP (Không dùng Flatten)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Ép kích thước về 1x1
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes) # Siêu nhẹ, chỉ có 256 x 3 tham số
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

def preprocess_input(X):
    """
    Standardize NumPy images for PyTorch CustomCNN.
    Supports both single images (H, W, 3) and batches (N, H, W, 3).
    """
    if X.ndim == 3:
        # Single image
        X = X.astype(np.float32) / 255.0
        X = np.transpose(X, (2, 0, 1))
        X = (X - 0.5) / 0.5
    else:
        # Batch
        X = X.astype(np.float32) / 255.0
        X = np.transpose(X, (0, 3, 1, 2))
        X = (X - 0.5) / 0.5
    return X


def _plot_history(history, plot_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['train_acc'],     label='Train', linewidth=2)
    ax1.plot(history['val_acc'],  label='Val',   linewidth=2)
    ax1.set_title('Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(history['train_loss'],     label='Train', linewidth=2)
    ax2.plot(history['val_loss'], label='Val',   linewidth=2)
    ax2.set_title('Loss', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def train_cnn(
    model, train_loader, val_loader, 
    epochs, device, 
    checkpoint_dir, prefix="model",
    class_weights=None,
    learning_rate=None
):
    """
    Train CNN with checkpointing and plotting per epoch.
    If a checkpoint exists, training resumes from it.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_ckpt_path = os.path.join(checkpoint_dir, f"{prefix}_last.pth")
    best_ckpt_path = os.path.join(checkpoint_dir, f"{prefix}_best.pth")
    plot_path = os.path.join(checkpoint_dir, f"{prefix}_history.png")
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    lr = learning_rate if learning_rate is not None else LEARNING_RATE
    criterion = FocalLoss(weight=class_weights, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )

    start_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 3
    
    # Resume training if the last checkpoint exists
    if os.path.exists(last_ckpt_path):
        print(f"  => Resuming from checkpoint: {last_ckpt_path}")
        checkpoint = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            history = checkpoint['history']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"  => Resumed seamlessly at epoch {start_epoch}")
        except RuntimeError:
            print("  => [WARNING] Architecture mismatch! Cannot resume from old checkpoint. Starting from scratch.")

    if start_epoch >= epochs:
        print(f"  => Training already completed max epochs ({epochs}).")
        
        # Load best model for returning
        if os.path.exists(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt['model_state_dict'])
        return model, history

    print(f"\n  Training (up to {epochs} epochs) ...")
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        corrects = 0
        total_samples = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)
            
        epoch_loss = running_loss / total_samples
        epoch_acc = corrects / total_samples
        
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data).item()
                val_samples += inputs.size(0)
                
        val_epoch_loss = val_loss / val_samples
        val_epoch_acc = val_corrects / val_samples
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        
        print(f"  Epoch {epoch+1:02d}/{epochs} - "
              f"loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - "
              f"val_loss: {val_epoch_loss:.4f} - val_acc: {val_epoch_acc:.4f}")
              
        scheduler.step(val_epoch_loss)
        
        # Plot every epoch and save explicitly to avoid missing out a plot
        _plot_history(history, plot_path)
        
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'best_val_loss': best_val_loss,
                'epochs_no_improve': epochs_no_improve
            }, best_ckpt_path)
        else:
            epochs_no_improve += 1
            
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_val_loss': best_val_loss,
            'epochs_no_improve': epochs_no_improve
        }, last_ckpt_path)

        if epochs_no_improve >= early_stopping_patience:
            print(f"  Early stopping triggered at epoch {epoch+1}")
            break
            
    # Always load best weights before returning
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
    
    print(f"  Finished training workflow. Models are in: {checkpoint_dir}")
    return model, history

