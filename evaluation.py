"""
Evaluation metrics: accuracy, precision, recall, F1-score, and specificity.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)


def calculate_specificity(y_true, y_pred, num_classes):
    """
    Per-class specificity = TN / (TN + FP).

    Returns:
        list of float, one per class.
    """
    cm = confusion_matrix(y_true, y_pred)
    specs = []
    for i in range(num_classes):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return specs


def compute_metrics(y_true, y_pred, num_classes):
    """
    Compute all required metrics at once.

    Returns:
        dict with keys: accuracy, precision, recall, f1_score,
                        specificity_per_class, avg_specificity.
    """
    specs = calculate_specificity(y_true, y_pred, num_classes)
    return {
        'accuracy':              accuracy_score(y_true, y_pred),
        'precision':             precision_score(y_true, y_pred, average='weighted'),
        'recall':                recall_score(y_true, y_pred, average='weighted'),
        'f1_score':              f1_score(y_true, y_pred, average='weighted'),
        'specificity_per_class': specs,
        'avg_specificity':       float(np.mean(specs)),
    }
