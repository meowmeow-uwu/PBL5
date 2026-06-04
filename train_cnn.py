"""
CNN Training: Train CNN model with channel importance analysis.
"""

import gc
import os
import time
import itertools
import torch
import numpy as np
from torch.utils.data import DataLoader

from config import FINE_TUNE_EPOCHS
from model import CNN, train_cnn, FruitDataset
from evaluation import compute_metrics
import preprocessing

param_grid_cnn = {
    'batch_size': [16, 32],
    'learning_rate': [1e-3, 5e-4],
    'dropout_rate': [0.15, 0.3]
}


def train_paper_cnn(X_tr_rgb, X_v_rgb, X_te_rgb, y_tr, y_v, y_te, color_space, num_classes, CS_RESULTS_DIR, device):
    """Train CNN model, evaluate on test set, and compute channel importances."""
    print("\n  Training CNN with Grid Search...")
    class_sample_count = np.bincount(y_tr)
    weights = 1. / class_sample_count
    weights = weights / weights.sum() * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    
    keys = list(param_grid_cnn.keys())
    values = list(param_grid_cnn.values())
    combinations = list(itertools.product(*values))
    
    best_val_loss = float('inf')
    best_params = {k: v[0] for k, v in param_grid_cnn.items()}
    best_model_state = None
    
    t0_train = time.time()
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        bs = params['batch_size']
        lr = params['learning_rate']
        dr = params['dropout_rate']
        
        print(f"\n    [Grid Search] Trial: batch_size={bs}, lr={lr}, dropout_rate={dr}")
        
        train_loader = DataLoader(
            FruitDataset(X_tr_rgb, y_tr.astype(np.int64), color_space=color_space),
            batch_size=int(bs), shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            FruitDataset(X_v_rgb, y_v.astype(np.int64), color_space=color_space),
            batch_size=int(bs), shuffle=False, num_workers=4, pin_memory=True
        )
        
        cnn_model = CNN(num_classes, dropout_rate=dr).to(device)
        cnn_save_dir = os.path.join(CS_RESULTS_DIR, f"CNN_model_bs{bs}_lr{lr}_dr{dr}")
        
        trained_model, history = train_cnn(
            cnn_model, train_loader, val_loader,
            epochs=FINE_TUNE_EPOCHS, device=device,
            checkpoint_dir=cnn_save_dir, prefix="CNN",
            class_weights=class_weights,
            learning_rate=lr
        )
        
        val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
        if val_loss < best_val_loss or best_model_state is None:
            best_val_loss = val_loss
            best_params = params
            best_model_state = {k: v.cpu() for k, v in trained_model.state_dict().items()}
            
    t1_train = time.time()
    
    print(f"\n  Best CNN Params: {best_params}")
    
    # Reload best model and evaluate on test set
    cnn_model = CNN(num_classes, dropout_rate=best_params['dropout_rate']).to(device)
    if best_model_state is not None:
        cnn_model.load_state_dict(best_model_state)
        
        # Save absolute best model to standard location for prediction module
        final_save_dir = os.path.join(CS_RESULTS_DIR, "CNN_model")
        os.makedirs(final_save_dir, exist_ok=True)
        torch.save({
            'model_state_dict': best_model_state,
            'best_params': best_params
        }, os.path.join(final_save_dir, "CNN_best.pth"))
    
    test_loader = DataLoader(
        FruitDataset(X_te_rgb, y_te.astype(np.int64), color_space=color_space),
        batch_size=int(best_params['batch_size']), shuffle=False, num_workers=4, pin_memory=True
    )
    
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
    cnn_metrics['best_params'] = best_params
    
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
        
        shuf_loader = DataLoader(FruitDataset(X_shuf, y_te.astype(np.int64), color_space='RGB'), batch_size=int(best_params['batch_size']), shuffle=False)
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
