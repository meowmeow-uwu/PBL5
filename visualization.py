"""
Visualization helpers: confusion matrices, comparison bar chart, summary table.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import RESULTS_DIR

def plot_confusion_matrices(results, y_test, le):
    """Heatmap confusion matrix for each classifier, side-by-side."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    fig.suptitle("Confusion Matrices — Three-Class Classification",
                 fontsize=14, fontweight='bold')

    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_title(f"{name}\nAcc: {res['accuracy']*100:.2f}%", fontweight='bold')
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Confusion matrices saved to {path}")

def plot_comparison_chart(results):
    """Grouped bar chart comparing all five metrics across classifiers."""
    names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'avg_specificity']
    labels  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    colors  = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

    x = np.arange(len(names))
    w = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (m, lab, c) in enumerate(zip(metrics, labels, colors)):
        vals = [results[n][m] * 100 for n in names]
        bars = ax.bar(x + i * w, vals, w, label=lab, color=c, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Classifier', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Classifier Performance Comparison — Three-Class',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + w * 2)
    ax.set_xticklabels(names)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "classifier_comparison.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Comparison chart saved to {path}")

def print_summary_table(results):
    """Print a formatted comparison table and highlight the best classifier."""
    print("\n" + "=" * 80)
    print("SUMMARY: Three-Class Tomato Quality Classification")
    print("=" * 80)

    hdr = (f"{'Classifier':<25} {'Accuracy':>10} {'Precision':>10} "
           f"{'Recall':>10} {'F1-Score':>10} {'Specificity':>12}")
    print(hdr)
    print("-" * 80)

    best_acc, best_name = 0, ""
    for name, r in results.items():
        print(f"{name:<25} {r['accuracy']*100:>9.2f}% {r['precision']*100:>9.2f}% "
              f"{r['recall']*100:>9.2f}% {r['f1_score']*100:>9.2f}% "
              f"{r['avg_specificity']*100:>11.2f}%")
        if r['accuracy'] > best_acc:
            best_acc, best_name = r['accuracy'], name

    print("-" * 80)
    print(f"\n  >>> Best: {best_name}  —  Accuracy = {best_acc*100:.2f}%")
    print(f"  >>> Paper reports CNN-SVM ≈ 96.67% for three-class")
    print("=" * 80)
