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

# --- Define Model ---
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

# --- Define CustomCNN (from scratch) ---
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
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

class PaperCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(PaperCNN, self).__init__()
        # Kiến trúc 3 khối Convolution chuẩn bài báo
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128), # Ảnh 128x128 qua 3 lần MaxPool còn 16x16
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.fc(self.conv(x))


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
    early_stopping_patience = 7
    
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


