"""
2. Transfer Module
Load the pre-trained base CNN and fine-tune it on Dataset_Cachua.
"""

import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import warnings

from config import (
    DATASET_CACHUA_DIR, RESULTS_DIR, BATCH_SIZE, RANDOM_STATE, FINE_TUNE_EPOCHS
)
from preprocessing import load_and_preprocess_images, split_dataset
from augmentation import create_augmented_data
from classifiers import train_and_evaluate
from model import CustomCNN, preprocess_input, train_cnn, extract_features_loop
from visualization import plot_confusion_matrices, plot_comparison_chart, print_summary_table

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

def main():
    print("\n" + "=" * 60)
    print("  TRANSFER MODULE (Dataset_Cachua)")
    print("=" * 60 + "\n")

    # 1. Load Data
    images, labels = load_and_preprocess_images(dataset_dir=DATASET_CACHUA_DIR)
    
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
    
    # 4. Initialize and Load Checkpoint from Train Module
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomCNN(num_classes).to(device)
    
    base_model_path = os.path.join(RESULTS_DIR, "train_save_model", "base_cnn_best.pth")
    if os.path.exists(base_model_path):
        print(f"  => Loading pre-trained base model from {base_model_path}")
        checkpoint = torch.load(base_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("  => [WARNING] Base model not found! Training from scratch!")
        
    # Freezing feature layers for Transfer Learning (optional)
    # for param in model.features.parameters():
    #     param.requires_grad = False
    
    save_dir = os.path.join(RESULTS_DIR, "transfer_save_model")
    model, history = train_cnn(
        model, train_loader, val_loader,
        epochs=FINE_TUNE_EPOCHS, device=device,
        checkpoint_dir=save_dir, prefix="transfer_cnn"
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
    print("STEP 7: Generating Visualizations (Transfer)")
    print("=" * 60)
    plot_confusion_matrices(results, y_te, le)
    plot_comparison_chart(results)
    print_summary_table(results)

if __name__ == "__main__":
    main()
