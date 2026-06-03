"""
CNN Training: Train CNN model with channel importance analysis.
"""

import gc
import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

from config import BATCH_SIZE, FINE_TUNE_EPOCHS
from model import CNN, train_cnn, FruitDataset
from evaluation import compute_metrics
import preprocessing


def train_paper_cnn(X_tr_rgb, X_v_rgb, X_te_rgb, y_tr, y_v, y_te, color_space, num_classes, CS_RESULTS_DIR, device):
    """Train CNN model, evaluate on test set, and compute channel importances."""
    print("\n  Training CNN...")
    class_sample_count = np.bincount(y_tr)
    weights = 1. / class_sample_count
    weights = weights / weights.sum() * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    
    train_loader = DataLoader(
        FruitDataset(X_tr_rgb, y_tr.astype(np.int64), color_space=color_space),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        FruitDataset(X_v_rgb, y_v.astype(np.int64), color_space=color_space),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        FruitDataset(X_te_rgb, y_te.astype(np.int64), color_space=color_space),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    
    cnn_model = CNN(num_classes).to(device)
    cnn_save_dir = os.path.join(CS_RESULTS_DIR, "CNN_model")
    
    t0_train = time.time()
    cnn_model, _ = train_cnn(
        cnn_model, train_loader, val_loader,
        epochs=FINE_TUNE_EPOCHS, device=device,
        checkpoint_dir=cnn_save_dir, prefix="CNN",
        class_weights=class_weights
    )
    t1_train = time.time()
    
    # Evaluate on test set
    cnn_model.eval()
    cnn_preds = []
    cnn_probs = []
    
    t0_inf = time.time()
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = cnn_model(inputs.to(device))
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            cnn_preds.extend(preds.cpu().numpy())
            cnn_probs.extend(probs.cpu().numpy())
    t1_inf = time.time()
            
    cnn_preds = np.array(cnn_preds)
    cnn_probs = np.array(cnn_probs)
    
    cnn_metrics = compute_metrics(y_te, cnn_preds, num_classes)
    cnn_metrics['y_pred'] = cnn_preds
    cnn_metrics['train_time'] = t1_train - t0_train
    cnn_metrics['inference_time'] = t1_inf - t0_inf
    
    # Channel importance via permutation
    print("    Computing channel importance for CNN...")
    cnn_model.eval()
    base_acc = np.mean(cnn_preds == y_te)
    importances = []
    X_te_cs_local = np.array([preprocessing.convert_color_spaces(img)[color_space] for img in X_te_rgb])
    for c in range(3):
        X_shuf = X_te_cs_local.copy()
        flat = X_shuf[:, :, :, c].flatten()
        np.random.shuffle(flat)
        X_shuf[:, :, :, c] = flat.reshape(X_shuf[:, :, :, c].shape)
        
        shuf_loader = DataLoader(FruitDataset(X_shuf, y_te.astype(np.int64), color_space='RGB'), batch_size=BATCH_SIZE, shuffle=False)
        shuf_preds = []
        with torch.no_grad():
            for inputs, _ in shuf_loader:
                outputs = cnn_model(inputs.to(device))
                _, p = torch.max(outputs, 1)
                shuf_preds.extend(p.cpu().numpy())
        shuf_acc = np.mean(np.array(shuf_preds) == y_te)
        importances.append(max(0, base_acc - shuf_acc))
        
    del X_te_cs_local
    gc.collect()
        
    total = sum(importances)
    if total > 0:
        cnn_metrics['channel_importances'] = [i/total*100 for i in importances]
    else:
        cnn_metrics['channel_importances'] = [33.33, 33.33, 33.33]

    return cnn_metrics, cnn_probs
