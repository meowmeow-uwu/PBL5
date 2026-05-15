"""
Train Cachua Module
Train a Custom CNN on the Dataset_Cachua from scratch,
then extract features and train ML classifiers.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import warnings

from config import (
    DATASET_CACHUA_DIR, RESULTS_DIR, BATCH_SIZE, RANDOM_STATE, FINE_TUNE_EPOCHS
)
from model import CustomCNN, preprocess_input, train_cnn, extract_features_loop, FruitDataset
from evaluation import compute_metrics
from sklearn.metrics import classification_report

import preprocessing
import augmentation
import classifiers
import visualization

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

def main():
    print("\n" + "=" * 60)
    print("  TRAINING MODULE (Dataset_Cachua - From Scratch)")
    print("=" * 60 + "\n")

    # Save to a different directory
    TRAIN_CACHUA_RESULTS_DIR = os.path.join(RESULTS_DIR, "train_cachua_results")
    os.makedirs(TRAIN_CACHUA_RESULTS_DIR, exist_ok=True)
    
    # Patch RESULTS_DIR in other modules so they save to the right place
    preprocessing.RESULTS_DIR = TRAIN_CACHUA_RESULTS_DIR
    classifiers.RESULTS_DIR = TRAIN_CACHUA_RESULTS_DIR
    visualization.RESULTS_DIR = TRAIN_CACHUA_RESULTS_DIR

    # 1. Load Data
    images, labels = preprocessing.load_and_preprocess_images(dataset_dir=DATASET_CACHUA_DIR)
    
    # 2. Split
    X_tr, X_v, X_te, y_tr, y_v, y_te, le = preprocessing.split_dataset(images, labels)
    num_classes = len(le.classes_)
    
    # 3. Augment
    X_tr_aug, y_tr_aug = augmentation.create_augmented_data(X_tr, y_tr)
    
    # 4. DataLoaders (using FruitDataset for memory efficiency)
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
    
    # 4. Train Custom CNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = CustomCNN(num_classes).to(device)
    
    # Calculate class weights
    targets = y_tr_aug.astype(np.int64)
    class_sample_count = np.bincount(targets)
    weights = 1. / class_sample_count
    weights = weights / weights.sum() * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"  => Calculated Class Weights: {weights}")

    save_dir = os.path.join(TRAIN_CACHUA_RESULTS_DIR, "train_save_model")
    model, history = train_cnn(
        model, train_loader, val_loader,
        epochs=FINE_TUNE_EPOCHS, device=device,
        checkpoint_dir=save_dir, prefix="cachua_cnn",
        class_weights=class_weights
    )
    
    # 5. Extract Features
    print("\n" + "=" * 60)
    print("STEP 5: Feature Extraction (GAP)")
    print("=" * 60)
    train_feat = extract_features_loop(model, train_orig_loader, device)
    test_feat = extract_features_loop(model, test_loader, device)
    
    # Evaluate CustomCNN directly
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
    print("Evaluating CustomCNN")
    print("=" * 60)
    print(f"  Accuracy:    {cnn_metrics['accuracy']*100:.2f}%")
    print(f"  Precision:   {cnn_metrics['precision']*100:.2f}%")
    print(f"  Recall:      {cnn_metrics['recall']*100:.2f}%")
    print(f"  F1-Score:    {cnn_metrics['f1_score']*100:.2f}%")
    
    report_str = classification_report(y_te, cnn_preds, target_names=le.classes_)
    report_path = os.path.join(TRAIN_CACHUA_RESULTS_DIR, "classification_report.txt")
    
    # 6. Classifiers
    results = classifiers.train_and_evaluate(train_feat, test_feat, y_tr, y_te, le)
    
    # append CustomCNN report
    with open(report_path, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Classification Report (CustomCNN)\n")
        f.write(f"{'='*60}\n")
        f.write(report_str + "\n")
        
    results['CustomCNN'] = cnn_metrics
    
    # 7. Visualize
    print("\n" + "=" * 60)
    print("STEP 7: Generating Visualizations")
    print("=" * 60)
    visualization.plot_confusion_matrices(results, y_te, le)
    visualization.plot_comparison_chart(results)
    visualization.print_summary_table(results)

if __name__ == "__main__":
    main()
