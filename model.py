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

from config import (
    LEARNING_RATE, DROPOUT_1, DROPOUT_2, DENSE_UNITS
)

# --- Define Model ---
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
        # Result dimension: 512
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


def preprocess_input(X):
    """
    Standardize NumPy images for PyTorch CustomCNN.
    """
    # Convert to float32 and scale to [0, 1]
    X = X.astype(np.float32) / 255.0
    # Transpose to (N, C, H, W)
    X = np.transpose(X, (0, 3, 1, 2))
    # Normalize arbitrarily to [-1, 1]
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
    checkpoint_dir, prefix="model"
):
    """
    Train CNN with checkpointing and plotting per epoch.
    If a checkpoint exists, training resumes from it.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_ckpt_path = os.path.join(checkpoint_dir, f"{prefix}_last.pth")
    best_ckpt_path = os.path.join(checkpoint_dir, f"{prefix}_best.pth")
    plot_path = os.path.join(checkpoint_dir, f"{prefix}_history.png")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7, verbose=True
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
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        history = checkpoint['history']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"  => Resumed seamlessly at epoch {start_epoch}")

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


def extract_features_loop(model, dataloader, device):
    """
    Extract features directly from CustomCNN GAP layer.
    """
    model.eval()
    feat_list = []
    with torch.no_grad():
        for inputs in dataloader:
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                inputs = inputs[0]
            outputs = model.extract_features(inputs.to(device))
            feat_list.append(outputs.cpu().numpy())
    return np.vstack(feat_list)
