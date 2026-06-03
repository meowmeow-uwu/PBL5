import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
import warnings
import re
import joblib

from config import (
    DATASET_CACHUA_DIR, RESULTS_DIR, BATCH_SIZE, RANDOM_STATE, FINE_TUNE_EPOCHS,
    COLOR_SPACES, VAL_SIZE_FROM_TRAINVAL
)
from model import PaperCNN, train_cnn, FruitDataset
from evaluation import compute_metrics
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

import preprocessing
import classifiers
import visualization
from statistical_features import extract_statistical_features

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


def compute_and_plot_roc(models_dict, X_test_dict, y_test, num_classes, le, save_path, title_prefix):
    plt.figure(figsize=(10, 8))
    y_test_bin = label_binarize(y_test, classes=range(num_classes))
    if num_classes == 2:
        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
    for i, (name, clf) in enumerate(models_dict.items()):
        X_te = X_test_dict[name]
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_te)
        elif hasattr(clf, "decision_function"):
            y_score = clf.decision_function(X_te)
            if y_score.ndim == 1:
                y_score = np.vstack([-y_score, y_score]).T
        else:
            continue
            
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i_c in range(num_classes):
            fpr[i_c], tpr[i_c], _ = roc_curve(y_test_bin[:, i_c], y_score[:, i_c])
            roc_auc[i_c] = auc(fpr[i_c], tpr[i_c])

        all_fpr = np.unique(np.concatenate([fpr[i_c] for i_c in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i_c in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i_c], tpr[i_c])

        mean_tpr /= num_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        plt.plot(fpr["macro"], tpr["macro"], color=colors[i % len(colors)], lw=2,
                 label=f'{name} (macro-average AUC = {roc_auc["macro"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title_prefix}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=150)
    plt.close()


def load_and_split_data():
    print("Loading Dataset_Cachua Train...")
    X_tr_full_rgb, labels_tr_full, fnames_tr_full = preprocessing.load_and_preprocess_images(dataset_dir=os.path.join(DATASET_CACHUA_DIR, "train"))
    
    print("Loading Dataset_Cachua Test...")
    X_te_rgb, labels_te, fnames_te = preprocessing.load_and_preprocess_images(dataset_dir=os.path.join(DATASET_CACHUA_DIR, "test"), save_samples=False)
    
    le = LabelEncoder()
    y_tr_full = le.fit_transform(labels_tr_full)
    y_te = le.transform(labels_te)
    num_classes = len(le.classes_)
    
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
    import gc
    gc.collect()

    print("\n" + "=" * 60)
    print("Dataset Split Summary")
    print("=" * 60)
    print(f"  Train: {len(X_tr_rgb)} ({len(X_tr_rgb)/n*100:.1f}%)")
    print(f"  Val:   {len(X_v_rgb)}  ({len(X_v_rgb)/n*100:.1f}%)")
    print(f"  Test:  {len(X_te_rgb)} ({len(X_te_rgb)/n*100:.1f}%)")

    return X_tr_rgb, X_v_rgb, X_te_rgb, y_tr, y_v, y_te, le, num_classes


def get_features(X_rgb, cs):
    feats = []
    for img in X_rgb:
        feats.append(extract_statistical_features(preprocessing.convert_color_spaces(img)[cs]))
    return np.array(feats)


def train_traditional_ml(X_tr_rgb, X_te_rgb, y_tr, y_te, color_space, num_classes, le, CS_RESULTS_DIR):
    print(f"Extracting 12D Statistical Features for {color_space}...")
    X_tr_feat = get_features(X_tr_rgb, color_space)
    X_te_feat = get_features(X_te_rgb, color_space)
    
    scaler = StandardScaler()
    X_tr_feat_sc = scaler.fit_transform(X_tr_feat)
    X_te_feat_sc = scaler.transform(X_te_feat)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    param_grids = {
        'SVM': {
            'model': SVC(probability=True, random_state=RANDOM_STATE),
            'params': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'K-NN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        },
        'Gaussian NB': {
            'model': GaussianNB(),
            'params': {
                'var_smoothing': [1e-10, 5e-10, 1e-9, 5e-9, 1e-8]
            }
        }
    }
    
    best_models = {}
    ml_results = {}
    
    for name, config in param_grids.items():
        print(f"  GridSearchCV for {name}...")
        grid = GridSearchCV(config['model'], config['params'], cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
        grid.fit(X_tr_feat_sc, y_tr)
        best_model = grid.best_estimator_
        best_models[name] = best_model
        
        print(f"    Best params: {grid.best_params_}")
        y_pred = best_model.predict(X_te_feat_sc)
        
        metrics = compute_metrics(y_te, y_pred, num_classes)
        metrics['y_pred'] = y_pred
        metrics['best_params'] = grid.best_params_
        
        model_save_path = os.path.join(CS_RESULTS_DIR, f"{name.replace(' ', '_').lower()}_best.pkl")
        joblib.dump({
            'model': best_model,
            'scaler': scaler,
            'le': le,
            'color_space': color_space
        }, model_save_path)
        
        print(f"    Computing channel importance for {name}...")
        pi = permutation_importance(best_model, X_te_feat_sc, y_te, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
        imps = pi.importances_mean
        c1 = max(0, sum(imps[0:4]))
        c2 = max(0, sum(imps[4:8]))
        c3 = max(0, sum(imps[8:12]))
        total = c1 + c2 + c3
        if total > 0:
            c_pct = [c1/total*100, c2/total*100, c3/total*100]
        else:
            c_pct = [33.33, 33.33, 33.33]
        metrics['channel_importances'] = c_pct
        
        ml_results[name] = metrics
        
    return ml_results, best_models, X_te_feat_sc


def train_paper_cnn(X_tr_rgb, X_v_rgb, X_te_rgb, y_tr, y_v, y_te, color_space, num_classes, CS_RESULTS_DIR, device):
    print("\n  Training PaperCNN...")
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
    
    cnn_model = PaperCNN(num_classes).to(device)
    cnn_save_dir = os.path.join(CS_RESULTS_DIR, "papercnn_model")
    
    cnn_model, _ = train_cnn(
        cnn_model, train_loader, val_loader,
        epochs=FINE_TUNE_EPOCHS, device=device,
        checkpoint_dir=cnn_save_dir, prefix="papercnn",
        class_weights=class_weights
    )
    
    cnn_model.eval()
    cnn_preds = []
    cnn_probs = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = cnn_model(inputs.to(device))
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            cnn_preds.extend(preds.cpu().numpy())
            cnn_probs.extend(probs.cpu().numpy())
            
    cnn_preds = np.array(cnn_preds)
    cnn_probs = np.array(cnn_probs)
    
    cnn_metrics = compute_metrics(y_te, cnn_preds, num_classes)
    cnn_metrics['y_pred'] = cnn_preds
    
    print(f"    Computing channel importance for PaperCNN...")
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
    import gc
    gc.collect()
        
    total = sum(importances)
    if total > 0:
        cnn_metrics['channel_importances'] = [i/total*100 for i in importances]
    else:
        cnn_metrics['channel_importances'] = [33.33, 33.33, 33.33]

    return cnn_metrics, cnn_probs


def save_and_visualize_reports(all_results, best_models, X_te_feat_sc, cnn_probs, y_te, num_classes, le, color_space, CS_RESULTS_DIR):
    report_path = os.path.join(CS_RESULTS_DIR, f"classification_report_{color_space}.txt")
    with open(report_path, "w") as f:
        for name, metrics in all_results.items():
            f.write(f"{'='*60}\nClassification Report ({name})\n")
            if 'best_params' in metrics:
                f.write(f"Optimal Parameters: {metrics['best_params']}\n")
            f.write(f"{'='*60}\n")
            f.write(classification_report(y_te, metrics['y_pred'], target_names=le.classes_))
            f.write("\n\n")
            
    print("\n  Generating Visualizations...")
    visualization.plot_confusion_matrices(all_results, y_te, le)
    visualization.print_summary_table(all_results)
    
    roc_models = best_models.copy()
    class CNNWrapper:
        def predict_proba(self, X):
            return cnn_probs
    roc_models['PaperCNN'] = CNNWrapper()
    
    X_test_dict = {name: X_te_feat_sc for name in best_models.keys()}
    X_test_dict['PaperCNN'] = None
    
    roc_path = os.path.join(CS_RESULTS_DIR, f"roc_auc_{color_space}.png")
    compute_and_plot_roc(roc_models, X_test_dict, y_te, num_classes, le, roc_path, color_space)


def generate_final_tables(global_results):
    print("\n" + "=" * 80)
    print("  GENERATING FINAL EVALUATION TABLE")
    print("=" * 80)
    
    def save_evaluation_table_to_txt(results, output_filename):
        data = []
        metrics_list = [
            ("Accuracy (%)", "accuracy"),
            ("Precision (%)", "precision"),
            ("Recall (%)", "recall"),
            ("F1-score (%)", "f1_score")
        ]
        
        for method_name in ["SVM", "Random Forest", "K-NN", "Gaussian NB", "PaperCNN"]:
            disp_name = method_name
            if method_name == "Random Forest": 
                disp_name = "RF"
            elif method_name == "Gaussian NB": 
                disp_name = "GNB"
            elif method_name == "PaperCNN": 
                disp_name = "CNN"
            
            for i, (metric_label, metric_key) in enumerate(metrics_list):
                row = [disp_name if i == 0 else "", metric_label]
                for cs in ["RGB", "HSV", "LAB", "YCrCb"]:
                    val = results[method_name].get(cs, {}).get(metric_key, 0)
                    row.append(f"{val:.2f}")
                data.append(row)
            data.append(["-"*10, "-"*15, "-"*10, "-"*10, "-"*10, "-"*10])
            
        data = data[:-1]
        columns = ["Method", "Metric", "RGB", "HSV", "CIE Lab", "YCbCr"]
        col_widths = [10, 17, 10, 10, 10, 10]
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("Table 4. Evaluation Results\n")
            f.write("=" * sum(col_widths) + "\n")
            header_str = "".join([f"{col:<{width}}" for col, width in zip(columns, col_widths)])
            f.write(header_str + "\n")
            f.write("=" * sum(col_widths) + "\n")
            for row in data:
                row_str = "".join([f"{str(item):<{width}}" for item, width in zip(row, col_widths)])
                f.write(row_str + "\n")
            f.write("=" * sum(col_widths) + "\n")
        print(f"  Đã lưu bảng kết quả đánh giá thực tế tại: {output_filename}")

    table_path = os.path.join(RESULTS_DIR, "table_evaluation_results.txt")
    save_evaluation_table_to_txt(global_results, table_path)

    def save_dominant_channel_table_to_txt(results, output_filename):
        data = []
        channel_names = {
            'RGB': ['R', 'G', 'B'],
            'HSV': ['H', 'S', 'V'],
            'LAB': ['L', 'a', 'b'],
            'YCrCb': ['Y', 'Cr', 'Cb']
        }
        display_cs = {'RGB': 'RGB', 'HSV': 'HSV', 'LAB': 'CIE Lab', 'YCrCb': 'YCbCr'}
        
        for method_name in ["SVM", "Random Forest", "K-NN", "Gaussian NB", "PaperCNN"]:
            disp_name = method_name
            if method_name == "Random Forest": 
                disp_name = "RF"
            elif method_name == "Gaussian NB": 
                disp_name = "GNB"
            elif method_name == "PaperCNN": 
                disp_name = "CNN"
            
            best_cs = 'RGB'
            best_acc = -1
            for cs in ["RGB", "HSV", "LAB", "YCrCb"]:
                acc = results[method_name].get(cs, {}).get('accuracy', 0)
                if acc > best_acc:
                    best_acc = acc
                    best_cs = cs
            
            imp = results[method_name].get(best_cs, {}).get('channel_importances', [33.33, 33.33, 33.33])
            ch_names = channel_names[best_cs]
            
            for i in range(3):
                row = [
                    disp_name if i == 0 else "",
                    display_cs[best_cs] if i == 0 else "",
                    ch_names[i],
                    f"{imp[i]:.2f}%"
                ]
                data.append(row)
            data.append(["-"*10, "-"*18, "-"*13, "-"*18])
            
        data = data[:-1]
        columns = ["Method", "Best color space", "Channel", "Contribution (%)"]
        col_widths = [12, 20, 15, 20]
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("Table 5. Dominant Color Channel Analysis\n")
            f.write("=" * sum(col_widths) + "\n")
            header_str = "".join([f"{col:<{width}}" for col, width in zip(columns, col_widths)])
            f.write(header_str + "\n")
            f.write("=" * sum(col_widths) + "\n")
            for row in data:
                row_str = "".join([f"{str(item):<{width}}" for item, width in zip(row, col_widths)])
                f.write(row_str + "\n")
            f.write("=" * sum(col_widths) + "\n")
        print(f"  Đã lưu bảng phân tích kênh màu thực tế tại: {output_filename}")
        
    dom_table_path = os.path.join(RESULTS_DIR, "table_dominant_channel.txt")
    save_dominant_channel_table_to_txt(global_results, dom_table_path)


def save_absolute_best_model(global_results):
    best_overall_acc = -1
    best_overall_info = {}
    for name, cs_data in global_results.items():
        for cs, metrics in cs_data.items():
            if metrics['accuracy'] > best_overall_acc:
                best_overall_acc = metrics['accuracy']
                best_overall_info = {
                    'model_name': name,
                    'color_space': cs,
                    'accuracy': metrics['accuracy']
                }
    
    if best_overall_info:
        info_path = os.path.join(RESULTS_DIR, "best_model_info.json")
        
        if best_overall_info['model_name'] == 'PaperCNN':
            best_overall_info['model_path'] = os.path.join(RESULTS_DIR, f"results_{best_overall_info['color_space']}", "papercnn_model", "papercnn_best.pth")
            best_overall_info['model_type'] = 'papercnn'
        else:
            best_overall_info['model_path'] = os.path.join(RESULTS_DIR, f"results_{best_overall_info['color_space']}", f"{best_overall_info['model_name'].replace(' ', '_').lower()}_best.pkl")
            best_overall_info['model_type'] = 'ml'
            
        with open(info_path, 'w') as f:
            json.dump(best_overall_info, f, indent=4)
            
        print("\n" + "=" * 80)
        print("  ABSOLUTE BEST MODEL SAVED FOR PREDICTION:")
        print(f"  Method: {best_overall_info['model_name']}")
        print(f"  Color Space: {best_overall_info['color_space']}")
        print(f"  Accuracy: {best_overall_info['accuracy']:.2f}%")
        print("=" * 80)


def main():
    print("\n" + "=" * 60)
    print("  TRAINING MODULE (Dataset_Cachua - Phase 2)")
    print("=" * 60 + "\n")

    X_tr_rgb, X_v_rgb, X_te_rgb, y_tr, y_v, y_te, le, num_classes = load_and_split_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    global_results = {
        'SVM': {},
        'Random Forest': {},
        'K-NN': {},
        'Gaussian NB': {},
        'PaperCNN': {}
    }

    for color_space in COLOR_SPACES:
        print("\n" + "=" * 80)
        print(f"  RUNNING PIPELINE FOR COLOR SPACE: {color_space}")
        print("=" * 80)
        
        CS_RESULTS_DIR = os.path.join(RESULTS_DIR, f"results_{color_space}")
        os.makedirs(CS_RESULTS_DIR, exist_ok=True)
        visualization.RESULTS_DIR = CS_RESULTS_DIR
        
        ml_results, best_models, X_te_feat_sc = train_traditional_ml(
            X_tr_rgb, X_te_rgb, y_tr, y_te, color_space, num_classes, le, CS_RESULTS_DIR
        )
        
        cnn_metrics, cnn_probs = train_paper_cnn(
            X_tr_rgb, X_v_rgb, X_te_rgb, y_tr, y_v, y_te, color_space, num_classes, CS_RESULTS_DIR, device
        )
        
        all_results = ml_results.copy()
        all_results['PaperCNN'] = cnn_metrics
        
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
                'channel_importances': metrics.get('channel_importances', [33.33, 33.33, 33.33])
            }
        
        print(f"\n  Finished Pipeline for {color_space}. Results saved to {CS_RESULTS_DIR}")

    generate_final_tables(global_results)
    save_absolute_best_model(global_results)


if __name__ == "__main__":
    main()
