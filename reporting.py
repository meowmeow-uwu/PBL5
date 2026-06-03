"""
Reporting: Classification reports, ROC curves, evaluation tables, and best model selection.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import RESULTS_DIR
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import visualization


def compute_and_plot_roc(models_dict, X_test_dict, y_test, num_classes, le, save_path, title_prefix):
    """Plot macro-average ROC curves for all models."""
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


def save_and_visualize_reports(all_results, best_models, X_te_feat_sc, cnn_probs, y_te, num_classes, le, color_space, CS_RESULTS_DIR):
    """Save classification reports, confusion matrices, and ROC curves."""
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
    roc_models['CNN'] = CNNWrapper()
    
    X_test_dict = {name: X_te_feat_sc for name in best_models.keys()}
    X_test_dict['CNN'] = None
    
    roc_path = os.path.join(CS_RESULTS_DIR, f"roc_auc_{color_space}.png")
    compute_and_plot_roc(roc_models, X_test_dict, y_te, num_classes, le, roc_path, color_space)


def generate_final_tables(global_results):
    """Generate evaluation results table and dominant channel analysis table."""
    print("\n" + "=" * 80)
    print("  GENERATING FINAL EVALUATION TABLE")
    print("=" * 80)
    
    _save_evaluation_table(global_results)
    _save_dominant_channel_table(global_results)


def _save_evaluation_table(results):
    data = []
    metrics_list = [
        ("Accuracy (%)", "accuracy"),
        ("Precision (%)", "precision"),
        ("Recall (%)", "recall"),
        ("F1-score (%)", "f1_score"),
        ("Train Time (s)", "train_time"),
        ("Infer Time (s)", "inference_time")
    ]
    
    for method_name in ["SVM", "Random Forest", "K-NN", "Gaussian NB", "CNN"]:
        disp_name = method_name
        if method_name == "Random Forest": 
            disp_name = "RF"
        elif method_name == "Gaussian NB": 
            disp_name = "GNB"
        elif method_name == "CNN": 
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
    col_widths = [12, 20, 10, 10, 10, 10]
    
    output_filename = os.path.join(RESULTS_DIR, "table_evaluation_results.txt")
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
    print(f"  Đã lưu bảng kết quả đánh giá tại: {output_filename}")


def _save_dominant_channel_table(results):
    data = []
    channel_names = {
        'RGB': ['R', 'G', 'B'],
        'HSV': ['H', 'S', 'V'],
        'LAB': ['L', 'a', 'b'],
        'YCrCb': ['Y', 'Cr', 'Cb']
    }
    display_cs = {'RGB': 'RGB', 'HSV': 'HSV', 'LAB': 'CIE Lab', 'YCrCb': 'YCbCr'}
    
    for method_name in ["SVM", "Random Forest", "K-NN", "Gaussian NB", "CNN"]:
        disp_name = method_name
        if method_name == "Random Forest": 
            disp_name = "RF"
        elif method_name == "Gaussian NB": 
            disp_name = "GNB"
        elif method_name == "CNN": 
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
    
    output_filename = os.path.join(RESULTS_DIR, "table_dominant_channel.txt")
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
    print(f"  Đã lưu bảng phân tích kênh màu tại: {output_filename}")


def save_absolute_best_model(global_results):
    """Find the best model across all methods/color spaces and save its info for prediction."""
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
        
        if best_overall_info['model_name'] == 'CNN':
            best_overall_info['model_path'] = os.path.join(RESULTS_DIR, f"results_{best_overall_info['color_space']}", "CNN_model", "CNN_best.pth")
            best_overall_info['model_type'] = 'CNN'
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
