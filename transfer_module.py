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
from model import CustomCNN, preprocess_input, train_cnn, FruitDataset
from evaluation import compute_metrics
from sklearn.metrics import classification_report
from visualization import plot_confusion_matrices, plot_comparison_chart, print_summary_table

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

def main():
    print("\n" + "=" * 60)
    print("  TRANSFER MODULE (Dataset_Cachua)")
    print("=" * 60 + "\n")

    # 1. Load Data
    images, labels, fnames = load_and_preprocess_images(dataset_dir=DATASET_CACHUA_DIR)
    
    # 2. Split
    X_tr, X_v, X_te, y_tr, y_v, y_te, le = split_dataset(images, labels, fnames)
    num_classes = len(le.classes_)
    
    # 3. Augment (Temporarily Disabled)
    X_tr_aug, y_tr_aug = X_tr, y_tr
    # X_tr_aug, y_tr_aug = create_augmented_data(X_tr, y_tr)
    
    # Use FruitDataset to handle preprocessing on-the-fly (saves memory)
    train_loader = DataLoader(
        FruitDataset(X_tr_aug, y_tr_aug.astype(np.int64)),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        FruitDataset(X_v, y_v.astype(np.int64)),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    train_orig_loader = DataLoader(FruitDataset(X_tr), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(FruitDataset(X_te), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
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
        
    # Class weights for Reject: 291, Ripe: 716, Unripe: 616
    # le.classes_ are ['Reject', 'Ripe', 'Unripe'] usually, but let's be safe and check dist
    counts = np.array([291, 716, 616])
    total = counts.sum()
    weights = total / (len(counts) * counts)
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"  => Calculated Class Weights: {weights}")

    save_dir = os.path.join(RESULTS_DIR, "transfer_save_model")
    model, history = train_cnn(
        model, train_loader, val_loader,
        epochs=FINE_TUNE_EPOCHS, device=device,
        checkpoint_dir=save_dir, prefix="transfer_cnn",
        class_weights=class_weights
    )
    
    # Evaluate MobileNetV3 directly
    model.eval()
    cnn_preds = []
    with torch.no_grad():
        for inputs in test_loader:
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                inputs = inputs[0]
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)
            cnn_preds.extend(preds.cpu().numpy())
    cnn_preds = np.array(cnn_preds)
    
    cnn_metrics = compute_metrics(y_te, cnn_preds, num_classes)
    cnn_metrics['y_pred'] = cnn_preds
    
    print("\n" + "=" * 60)
    print("Evaluating MobileNetV3 (Edge AI)")
    print("=" * 60)
    print(f"  Accuracy:    {cnn_metrics['accuracy']*100:.2f}%")
    print(f"  Precision:   {cnn_metrics['precision']*100:.2f}%")
    print(f"  Recall:      {cnn_metrics['recall']*100:.2f}%")
    print(f"  F1-Score:    {cnn_metrics['f1_score']*100:.2f}%")
    
    report_str = classification_report(y_te, cnn_preds, target_names=le.classes_)
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    
    with open(report_path, "w") as f:
        f.write(f"{'='*60}\n")
        f.write(f"Classification Report (MobileNetV3)\n")
        f.write(f"{'='*60}\n")
        f.write(report_str + "\n")
        
    results = {'MobileNetV3': cnn_metrics}
    
    # 7. Visualize
    print("\n" + "=" * 60)
    print("STEP 7: Generating Visualizations (Transfer)")
    print("=" * 60)
    plot_confusion_matrices(results, y_te, le)
    plot_comparison_chart(results)
    print_summary_table(results)

if __name__ == "__main__":
    main()
