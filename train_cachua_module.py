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
    DATASET_CACHUA_DIR, DATASET_DIR, RESULTS_DIR, BATCH_SIZE, RANDOM_STATE, FINE_TUNE_EPOCHS
)
from model import CustomCNN, MobileNetV3Edge, preprocess_input, train_cnn, FruitDataset
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
    print("Loading Dataset_Cachua Train...")
    X_tr_full, labels_tr_full, fnames_tr_full = preprocessing.load_and_preprocess_images(dataset_dir=os.path.join(DATASET_CACHUA_DIR, "train"))
    
    print("Loading Dataset_Cachua Test...")
    X_te, labels_te, fnames_te = preprocessing.load_and_preprocess_images(dataset_dir=os.path.join(DATASET_CACHUA_DIR, "test"), save_samples=False)
    
    # 2. Process Labels & Split Val from Train
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    import re

    le = LabelEncoder()
    y_tr_full = le.fit_transform(labels_tr_full)
    y_te = le.transform(labels_te)
    num_classes = len(le.classes_)
    
    # Extract Group IDs to prevent data leakage in train/val split
    groups = []
    for fname in fnames_tr_full:
        match = re.match(r"^(\d+)", fname)
        if match:
            groups.append(match.group(1))
        else:
            groups.append(fname)
            
    groups = np.array(groups)
    
    unique_groups = np.unique(groups)
    group_labels = []
    for g in unique_groups:
        idx = np.where(groups == g)[0][0]
        group_labels.append(y_tr_full[idx])
        
    unique_groups = np.array(unique_groups)
    group_labels = np.array(group_labels)
    
    from config import VAL_SIZE_FROM_TRAINVAL
    # Split Val from Train
    g_tr, g_v, _, _ = train_test_split(
        unique_groups, group_labels, test_size=VAL_SIZE_FROM_TRAINVAL, 
        stratify=group_labels, random_state=RANDOM_STATE
    )
    
    train_mask = np.isin(groups, g_tr)
    val_mask = np.isin(groups, g_v)
    
    X_tr, y_tr = X_tr_full[train_mask], y_tr_full[train_mask]
    X_v, y_v = X_tr_full[val_mask], y_tr_full[val_mask]

    n = len(X_tr_full) + len(X_te)
    print("\n" + "=" * 60)
    print("Dataset Split Summary")
    print("=" * 60)
    print(f"  Train: {len(X_tr)} ({len(X_tr)/n*100:.1f}%)")
    print(f"  Val:   {len(X_v)}  ({len(X_v)/n*100:.1f}%)")
    print(f"  Test:  {len(X_te)} ({len(X_te)/n*100:.1f}%)")

    for tag, ys in [("Train", y_tr), ("Val", y_v), ("Test", y_te)]:
        counts = np.bincount(ys)
        dist = ", ".join(f"{le.classes_[i]}: {counts[i]}" for i in range(len(counts)))
        print(f"    {tag}: {dist}")
    
    # 3. Augment & Balance (Dynamic and memory efficient)
    train_indices = augmentation.get_balanced_indices(y_tr)
    train_transform = augmentation.get_dynamic_transform()
    
    # 4. DataLoaders (using FruitDataset for memory efficiency)
    train_loader = DataLoader(
        FruitDataset(X_tr, y_tr.astype(np.int64), transform=train_transform, indices=train_indices),
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
    # Calculate class weights
    targets = y_tr[train_indices].astype(np.int64)
    class_sample_count = np.bincount(targets)
    weights = 1. / class_sample_count
    weights = weights / weights.sum() * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"  => Calculated Class Weights: {weights}")

    print("\n" + "=" * 60)
    print("Training MobileNetV3 (Fine Tuning)")
    print("=" * 60)
    model_mn = MobileNetV3Edge(num_classes, fine_tune=True).to(device)
    save_dir_mn = os.path.join(TRAIN_CACHUA_RESULTS_DIR, "train_save_model", "mobilenet")
    model_mn, _ = train_cnn(
        model_mn, train_loader, val_loader,
        epochs=FINE_TUNE_EPOCHS, device=device,
        checkpoint_dir=save_dir_mn, prefix="mobilenet",
        class_weights=class_weights,
        learning_rate=1e-5
    )

    print("\n" + "=" * 60)
    print("Training CustomCNN (From Scratch)")
    print("=" * 60)
    model_custom = CustomCNN(num_classes).to(device)
    save_dir_custom = os.path.join(TRAIN_CACHUA_RESULTS_DIR, "train_save_model", "customcnn")
    model_custom, _ = train_cnn(
        model_custom, train_loader, val_loader,
        epochs=FINE_TUNE_EPOCHS, device=device,
        checkpoint_dir=save_dir_custom, prefix="customcnn",
        class_weights=class_weights
    )

    def evaluate_model(eval_model, eval_loader, name):
        eval_model.eval()
        preds_list = []
        true_list = []
        with torch.no_grad():
            for inputs in eval_loader:
                if isinstance(inputs, list) or isinstance(inputs, tuple):
                    true_list.extend(inputs[1].numpy())
                    inputs = inputs[0]
                outputs = eval_model(inputs.to(device))
                _, preds = torch.max(outputs, 1)
                preds_list.extend(preds.cpu().numpy())
        preds_arr = np.array(preds_list)
        true_arr = np.array(true_list) if len(true_list) > 0 else y_te
        
        metrics = compute_metrics(true_arr, preds_arr, num_classes)
        metrics['y_pred'] = preds_arr
        
        print("\n" + "=" * 60)
        print(f"Evaluating {name}")
        print("=" * 60)
        print(f"  Accuracy:    {metrics['accuracy']*100:.2f}%")
        print(f"  Precision:   {metrics['precision']*100:.2f}%")
        print(f"  Recall:      {metrics['recall']*100:.2f}%")
        print(f"  F1-Score:    {metrics['f1_score']*100:.2f}%")
        
        report_str = classification_report(true_arr, preds_arr, target_names=le.classes_)
        return metrics, report_str

    metrics_mn, report_mn = evaluate_model(model_mn, test_loader, "MobileNetV3")
    metrics_custom, report_custom = evaluate_model(model_custom, test_loader, "CustomCNN")
    
    report_path = os.path.join(TRAIN_CACHUA_RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"{'='*60}\nClassification Report (MobileNetV3)\n{'='*60}\n{report_mn}\n")
        f.write(f"\n{'='*60}\nClassification Report (CustomCNN)\n{'='*60}\n{report_custom}\n")
        
    results = {
        'MobileNetV3': metrics_mn,
        'CustomCNN': metrics_custom
    }
    
    # 7. Visualize
    print("\n" + "=" * 60)
    print("STEP 7: Generating Visualizations")
    print("=" * 60)
    visualization.plot_confusion_matrices(results, y_te, le)
    visualization.plot_comparison_chart(results)
    visualization.print_summary_table(results)

    # 8. Cross-Dataset Evaluation
    # (Removed because datasets are now combined and evaluated together in Step 1 & 7)

if __name__ == "__main__":
    main()
