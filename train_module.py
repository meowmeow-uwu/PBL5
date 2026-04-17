"""
1. Train Module
Train a Custom CNN on the original Dataset (from scratch).
"""

import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import warnings

from config import (
    DATASET_DIR, RESULTS_DIR, BATCH_SIZE, RANDOM_STATE, FINE_TUNE_EPOCHS
)
from preprocessing import load_and_preprocess_images
from main import split_dataset   # Re-using the same split function
from augmentation import create_augmented_data
from classifiers import train_and_evaluate
from model import CustomCNN, preprocess_input, train_cnn, extract_features_loop
from visualization import plot_confusion_matrices, plot_comparison_chart, print_summary_table

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

def main():
    print("\n" + "=" * 60)
    print("  TRAINING MODULE (Original Dataset)")
    print("=" * 60 + "\n")

    # 1. Load Data
    images, labels = load_and_preprocess_images(dataset_dir=DATASET_DIR)
    
    # 2. Split
    X_tr, X_v, X_te, y_tr, y_v, y_te, le = split_dataset(images, labels)
    num_classes = len(le.classes_)
    
    # 3. Augment
    X_tr_aug, y_tr_aug = create_augmented_data(X_tr, y_tr)
    
    # Preprocess NumPy to PyTorch DataLoaders
    X_tr_p = preprocess_input(X_tr_aug)
    X_v_p  = preprocess_input(X_v)
    X_tr_orig_p = preprocess_input(X_tr)
    X_te_p = preprocess_input(X_te)
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr_p), torch.tensor(y_tr_aug.astype(np.int64))),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_v_p), torch.tensor(y_v.astype(np.int64))),
        batch_size=BATCH_SIZE, shuffle=False
    )
    
    train_orig_loader = DataLoader(TensorDataset(torch.tensor(X_tr_orig_p)), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_te_p)), batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Train Custom CNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomCNN(num_classes).to(device)
    
    save_dir = os.path.join(RESULTS_DIR, "train_save_model")
    model, history = train_cnn(
        model, train_loader, val_loader,
        epochs=FINE_TUNE_EPOCHS, device=device,
        checkpoint_dir=save_dir, prefix="base_cnn"
    )
    
    # 5. Extract Features
    print("\n" + "=" * 60)
    print("STEP 5: Feature Extraction (GAP)")
    print("=" * 60)
    train_feat = extract_features_loop(model, train_orig_loader, device)
    test_feat = extract_features_loop(model, test_loader, device)
    
    # 6. Classifiers
    results = train_and_evaluate(train_feat, test_feat, y_tr, y_te, le)
    
    # 7. Visualize
    print("\n" + "=" * 60)
    print("STEP 7: Generating Visualizations")
    print("=" * 60)
    plot_confusion_matrices(results, y_te, le)
    plot_comparison_chart(results)
    print_summary_table(results)

if __name__ == "__main__":
    main()
