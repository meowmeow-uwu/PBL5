"""
Evaluation script to test the best model (CustomCNN and hybrid classifiers) on a test dataset.
Supports evaluating on:
1. A new test set directory (with subfolders Reject, Ripe, Unripe)
2. The 20% test split from the existing Dataset_Cachua.
"""

import os
import argparse
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import torch
import cv2
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from config import (
    DATASET_CACHUA_DIR, RESULTS_DIR, BATCH_SIZE, RANDOM_STATE, IMG_SIZE, CLASS_NAMES,
    PCA_VARIANCE_RATIO, KNN_NEIGHBORS, SVM_KERNEL, SVM_C, RF_N_ESTIMATORS
)
from preprocessing import background_cancellation, load_and_preprocess_images, split_dataset
from model import CustomCNN, preprocess_input, extract_features_loop, FruitDataset

def compute_metrics_robust(y_true, y_pred, class_names):
    """
    Compute classification metrics robustly, even if some classes are missing in the test set.
    """
    num_classes = len(class_names)
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Per-class specificity: TN / (TN + FP)
    specs = []
    for i in range(num_classes):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        
    return {
        'accuracy':              accuracy_score(y_true, y_pred),
        'precision':             precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
        'recall':                recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
        'f1_score':              f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
        'specificity_per_class': specs,
        'avg_specificity':       float(np.mean(specs)),
        'confusion_matrix':      cm
    }

def load_custom_test_set(test_dir, img_size):
    """
    Load images from a new directory containing subfolders: Reject, Ripe, Unripe.
    """
    class_dirs = {
        'Reject': os.path.join(test_dir, 'Reject'),
        'Ripe':   os.path.join(test_dir, 'Ripe'),
        'Unripe': os.path.join(test_dir, 'Unripe'),
    }
    
    print("\n" + "=" * 60)
    print(f"Loading Custom Test Set from: {test_dir}")
    print("=" * 60)
    
    images, labels = [], []
    for cls, path in class_dirs.items():
        if not os.path.exists(path):
            print(f"  [WARNING] Class folder not found: {path}")
            continue
        
        files = [f for f in os.listdir(path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  {cls}: Found {len(files)} images.")
        
        for fname in files:
            img_path = os.path.join(path, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            # Apply preprocessing
            roi = background_cancellation(img)
            roi = cv2.resize(roi, (img_size, img_size))
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            images.append(roi_rgb)
            labels.append(cls)
            
    if len(images) == 0:
        raise ValueError(f"No valid images found in {test_dir}! Check folder names (Reject, Ripe, Unripe) and image files.")
        
    images = np.array(images, dtype=np.uint8)
    label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    y_test = np.array([label_map[l] for l in labels], dtype=np.int64)
    
    print(f"  Successfully loaded {len(images)} test images.")
    return images, y_test

def plot_test_confusion_matrices(results, y_test, class_names, save_path):
    """
    Plot confusion matrices side-by-side or in a grid.
    """
    n = len(results)
    if n == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]
            
    fig.suptitle("Confusion Matrices on Test Set", fontsize=15, fontweight='bold')
    
    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx]
        cm = res['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f"{name}\nAcc: {res['accuracy']*100:.2f}%", fontweight='bold')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Confusion matrices saved to: {save_path}")

def print_summary_table(results, class_names):
    """
    Print a neat summary table of results.
    """
    print("\n" + "=" * 85)
    print("TEST EVALUATION SUMMARY")
    print("=" * 85)
    hdr = (f"{'Model/Classifier':<25} {'Accuracy':>10} {'Precision':>10} "
           f"{'Recall':>10} {'F1-Score':>10} {'Avg Spec':>12}")
    print(hdr)
    print("-" * 85)
    
    best_acc, best_name = 0, ""
    for name, r in results.items():
        print(f"{name:<25} {r['accuracy']*100:>9.2f}% {r['precision']*100:>9.2f}% "
              f"{r['recall']*100:>9.2f}% {r['f1_score']*100:>9.2f}% "
              f"{r['avg_specificity']*100:>11.2f}%")
        if r['accuracy'] > best_acc:
            best_acc, best_name = r['accuracy'], name
            
    print("-" * 85)
    print(f"  >>> Best Performing Model: {best_name} (Accuracy: {best_acc*100:.2f}%)")
    print("=" * 85 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Tomato Quality Model on a Test Set")
    parser.add_argument("--test_dir", type=str, default=None, 
                        help="Path to a new test dataset containing Reject, Ripe, Unripe folders. If omitted, uses test split of the existing dataset.")
    parser.add_argument("--dataset_dir", type=str, default=DATASET_CACHUA_DIR, 
                        help="Path to dataset to extract test split from (used when --test_dir is not specified).")
    parser.add_argument("--model_path", type=str, default=None, 
                        help="Path to the trained model checkpoint (.pth).")
    parser.add_argument("--only_cnn", action="store_true", 
                        help="Only evaluate the pure CustomCNN model, skip ML hybrid classifiers.")
    parser.add_argument("--results_dir", type=str, default=None, 
                        help="Directory to save evaluation reports and plots.")
    args = parser.parse_args()

    # 1. Setup paths with environment variables fallback
    if args.test_dir is None:
        args.test_dir = os.getenv("EVAL_TEST_DIR", None)

    if args.model_path is None:
        args.model_path = os.getenv("EVAL_MODEL_PATH", os.getenv("MODEL_PATH", None))
        if args.model_path is None:
            # Default fallback to base_cnn_best.pth in train_save_model as requested
            args.model_path = os.path.join(RESULTS_DIR, "train_save_model", "base_cnn_best.pth")
        
    if args.results_dir is None:
        args.results_dir = os.path.join(RESULTS_DIR, "test_evaluation")
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 2. Check model
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Trained model checkpoint not found at: {args.model_path}")
        print("Please check the model path or train the model first.")
        return

    # 3. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    num_classes = len(CLASS_NAMES)
    
    print(f"Loading weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    try:
        model = CustomCNN(num_classes, has_dropout=True).to(device)
        model.load_state_dict(state_dict)
        print("  => Loaded CustomCNN (with dropout) successfully.")
    except RuntimeError as e:
        print("  => Retrying CustomCNN without dropout...")
        try:
            model = CustomCNN(num_classes, has_dropout=False).to(device)
            model.load_state_dict(state_dict)
            print("  => Loaded CustomCNN (without dropout) successfully.")
        except RuntimeError as e2:
            print("  => Retrying MobileNetV3Edge...")
            from model import MobileNetV3Edge
            model = MobileNetV3Edge(num_classes, fine_tune=False).to(device)
            model.load_state_dict(state_dict)
            print("  => Loaded MobileNetV3Edge successfully.")
    model.eval()
    print("Model loaded successfully.")

    # 4. Load Test Data
    if args.test_dir:
        # Load from custom test folder
        X_te, y_te = load_custom_test_set(args.test_dir, IMG_SIZE)
        X_tr, y_tr = None, None
    else:
        # Load from training dataset and split to get the test split
        if not os.path.exists(args.dataset_dir):
            print(f"[ERROR] Dataset directory not found at: {args.dataset_dir}")
            return
        images, labels = load_and_preprocess_images(dataset_dir=args.dataset_dir)
        X_tr, _, X_te, y_tr, _, y_te, _ = split_dataset(images, labels)
        
    # Preprocess test images for PyTorch
    X_te_p = preprocess_input(X_te)
    test_loader = DataLoader(
        FruitDataset(X_te),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    results = {}

    # 5. Evaluate Pure CNN
    print("\n" + "=" * 60)
    print("Evaluating Pure CustomCNN Model...")
    print("=" * 60)
    test_loader_eval = DataLoader(
        FruitDataset(X_te, y_te),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    cnn_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader_eval:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            cnn_preds.extend(preds.cpu().numpy())
            
    cnn_preds = np.array(cnn_preds)
    results['Pure CustomCNN'] = compute_metrics_robust(y_te, cnn_preds, CLASS_NAMES)
    results['Pure CustomCNN']['y_pred'] = cnn_preds

    # 6. Evaluate Hybrid Classifiers (CNN + ML)
    run_ml = not args.only_cnn
    if run_ml:
        # We need training data to fit the classifiers on top of the extracted features
        if X_tr is None:
            if os.path.exists(args.dataset_dir):
                print(f"\nLoading training dataset from {args.dataset_dir} to train ML classifiers...")
                images_tr, labels_tr = load_and_preprocess_images(dataset_dir=args.dataset_dir, save_samples=False)
                X_tr, _, _, y_tr, _, _, _ = split_dataset(images_tr, labels_tr)
            else:
                print(f"\n[WARNING] Training dataset not found at {args.dataset_dir}. Skipping hybrid classifiers.")
                run_ml = False
                
    if run_ml:
        print("\n" + "=" * 60)
        print("Extracting Features & Training Hybrid Classifiers...")
        print("=" * 60)
        
        # Extract features for training set
        train_orig_loader = DataLoader(
            FruitDataset(X_tr),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        print("  Extracting training features...")
        train_feat = extract_features_loop(model, train_orig_loader, device)
        
        # Extract features for test set
        print("  Extracting test features...")
        test_feat = extract_features_loop(model, test_loader, device)
        
        # Standardize features
        scaler = StandardScaler()
        train_sc = scaler.fit_transform(train_feat)
        test_sc  = scaler.transform(test_feat)
        
        # PCA for KNN
        print(f"  Applying PCA for KNN (retaining {PCA_VARIANCE_RATIO*100:.0f}% variance)...")
        pca = PCA(n_components=PCA_VARIANCE_RATIO, random_state=RANDOM_STATE)
        train_pca = pca.fit_transform(train_sc)
        test_pca  = pca.transform(test_sc)
        print(f"    Dimensions: {train_feat.shape[1]} -> {train_pca.shape[1]}")
        
        # Train classifiers
        classifiers = {
            'CNN-SVM': {
                'clf': SVC(kernel=SVM_KERNEL, C=SVM_C, gamma='scale', random_state=RANDOM_STATE),
                'tr': train_sc, 'te': test_sc,
            },
            'CNN-RF': {
                'clf': RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1),
                'tr': train_sc, 'te': test_sc,
            },
            'CNN-KNN (PCA)': {
                'clf': KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric='minkowski', p=2),
                'tr': train_pca, 'te': test_pca,
            },
        }
        
        for name, cfg in classifiers.items():
            print(f"  Training {name}...")
            clf = cfg['clf']
            clf.fit(cfg['tr'], y_tr)
            y_pred = clf.predict(cfg['te'])
            
            metrics = compute_metrics_robust(y_te, y_pred, CLASS_NAMES)
            metrics['y_pred'] = y_pred
            results[name] = metrics

    # 7. Print and Save Reports
    print_summary_table(results, CLASS_NAMES)
    
    # Save text report
    report_path = os.path.join(args.results_dir, "test_evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL EVALUATION REPORT ON TEST SET\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model path: {args.model_path}\n")
        if args.test_dir:
            f.write(f"Test data source: Custom Directory ({args.test_dir})\n")
        else:
            f.write(f"Test data source: 20% split of {args.dataset_dir}\n")
        f.write(f"Total test samples: {len(X_te)}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("1. METRICS SUMMARY TABLE\n")
        f.write("=" * 80 + "\n")
        hdr = (f"{'Model/Classifier':<25} {'Accuracy':>10} {'Precision':>10} "
               f"{'Recall':>10} {'F1-Score':>10} {'Avg Spec':>12}\n")
        f.write(hdr)
        f.write("-" * 80 + "\n")
        for name, r in results.items():
            f.write(f"{name:<25} {r['accuracy']*100:>9.2f}% {r['precision']*100:>9.2f}% "
                    f"{r['recall']*100:>9.2f}% {r['f1_score']*100:>9.2f}% "
                    f"{r['avg_specificity']*100:>11.2f}%\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("2. DETAILED CLASSIFICATION REPORTS\n")
        f.write("=" * 80 + "\n\n")
        for name, r in results.items():
            f.write(f"--- Model: {name} ---\n")
            report_str = classification_report(y_te, r['y_pred'], target_names=CLASS_NAMES, labels=[0, 1, 2], zero_division=0)
            f.write(report_str + "\n")
            
            f.write("Specificity per class:\n")
            for i, cls in enumerate(CLASS_NAMES):
                f.write(f"  - {cls}: {r['specificity_per_class'][i]*100:.2f}%\n")
            f.write("\n" + "-"*40 + "\n\n")
            
    print(f"  Classification report saved to: {report_path}")
    
    # Save Confusion Matrices plot
    cm_plot_path = os.path.join(args.results_dir, "test_confusion_matrices.png")
    plot_test_confusion_matrices(results, y_te, CLASS_NAMES, cm_plot_path)

if __name__ == "__main__":
    main()
