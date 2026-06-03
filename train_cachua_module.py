"""
Main Training Orchestrator for Dataset_Cachua.
Coordinates data loading, ML training, CNN training, and reporting.
"""

import gc
import os
import re
import torch
import numpy as np
import warnings

from config import (
    DATASET_CACHUA_DIR, RESULTS_DIR, RANDOM_STATE,
    COLOR_SPACES, VAL_SIZE_FROM_TRAINVAL
)
import preprocessing
import visualization
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from train_ml import train_traditional_ml
from train_cnn import train_paper_cnn
from reporting import save_and_visualize_reports, generate_final_tables, save_absolute_best_model

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


def load_and_split_data():
    """Load images and split into train/val/test using group-aware stratification."""
    print("Loading Dataset_Cachua Train...")
    X_tr_full_rgb, labels_tr_full, fnames_tr_full = preprocessing.load_and_preprocess_images(
        dataset_dir=os.path.join(DATASET_CACHUA_DIR, "train")
    )
    
    print("Loading Dataset_Cachua Test...")
    X_te_rgb, labels_te, _ = preprocessing.load_and_preprocess_images(
        dataset_dir=os.path.join(DATASET_CACHUA_DIR, "test"), save_samples=False
    )
    
    le = LabelEncoder()
    y_tr_full = le.fit_transform(labels_tr_full)
    y_te = le.transform(labels_te)
    num_classes = len(le.classes_)
    
    # Group-aware split to prevent data leakage
    groups = []
    for fname in fnames_tr_full:
        match = re.match(r"^(\d+)", fname)
        groups.append(match.group(1) if match else fname)
    groups = np.array(groups)
    
    unique_groups = np.unique(groups)
    group_labels = np.array([y_tr_full[np.where(groups == g)[0][0]] for g in unique_groups])
    
    g_tr, g_v, _, _ = train_test_split(
        unique_groups, group_labels, test_size=VAL_SIZE_FROM_TRAINVAL, 
        stratify=group_labels, random_state=RANDOM_STATE
    )
    
    train_mask = np.isin(groups, g_tr)
    val_mask = np.isin(groups, g_v)
    n = len(X_tr_full_rgb) + len(X_te_rgb)

    X_tr_rgb, y_tr = X_tr_full_rgb[train_mask], y_tr_full[train_mask]
    X_v_rgb, y_v = X_tr_full_rgb[val_mask], y_tr_full[val_mask]

    del X_tr_full_rgb
    gc.collect()

    print("\n" + "=" * 60)
    print("Dataset Split Summary")
    print("=" * 60)
    print(f"  Train: {len(X_tr_rgb)} ({len(X_tr_rgb)/n*100:.1f}%)")
    print(f"  Val:   {len(X_v_rgb)}  ({len(X_v_rgb)/n*100:.1f}%)")
    print(f"  Test:  {len(X_te_rgb)} ({len(X_te_rgb)/n*100:.1f}%)")

    return X_tr_rgb, X_v_rgb, X_te_rgb, y_tr, y_v, y_te, le, num_classes


def main():
    print("\n" + "=" * 60)
    print("  TRAINING MODULE (Dataset_Cachua)")
    print("=" * 60 + "\n")

    X_tr_rgb, X_v_rgb, X_te_rgb, y_tr, y_v, y_te, le, num_classes = load_and_split_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    global_results = {
        'SVM': {}, 'Random Forest': {}, 'K-NN': {}, 'Gaussian NB': {}, 'CNN': {}
    }

    for color_space in COLOR_SPACES:
        print("\n" + "=" * 80)
        print(f"  RUNNING PIPELINE FOR COLOR SPACE: {color_space}")
        print("=" * 80)
        
        CS_RESULTS_DIR = os.path.join(RESULTS_DIR, f"results_{color_space}")
        os.makedirs(CS_RESULTS_DIR, exist_ok=True)
        visualization.RESULTS_DIR = CS_RESULTS_DIR
        
        # Train ML models
        ml_results, best_models, X_te_feat_sc = train_traditional_ml(
            X_tr_rgb, X_te_rgb, y_tr, y_te, color_space, num_classes, le, CS_RESULTS_DIR
        )
        
        # Train CNN
        cnn_metrics, cnn_probs = train_paper_cnn(
            X_tr_rgb, X_v_rgb, X_te_rgb, y_tr, y_v, y_te, color_space, num_classes, CS_RESULTS_DIR, device
        )
        
        # Combine and report
        all_results = ml_results.copy()
        all_results['CNN'] = cnn_metrics
        
        save_and_visualize_reports(
            all_results, best_models, X_te_feat_sc, cnn_probs, 
            y_te, num_classes, le, color_space, CS_RESULTS_DIR
        )
        
        for name, metrics in all_results.items():
            global_results[name][color_space] = {
                'accuracy': metrics['accuracy'] * 100,
                'precision': metrics['precision'] * 100,
                'recall': metrics['recall'] * 100,
                'f1_score': metrics['f1_score'] * 100,
                'channel_importances': metrics.get('channel_importances', [33.33, 33.33, 33.33]),
                'train_time': metrics.get('train_time', 0.0),
                'inference_time': metrics.get('inference_time', 0.0)
            }
        
        print(f"\n  Finished Pipeline for {color_space}. Results saved to {CS_RESULTS_DIR}")

    generate_final_tables(global_results)
    save_absolute_best_model(global_results)


if __name__ == "__main__":
    main()
