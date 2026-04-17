"""
Train SVM, Random Forest, and KNN classifiers on extracted features.
KNN uses PCA for dimensionality reduction.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

import os
from config import (
    RANDOM_STATE, PCA_VARIANCE_RATIO, KNN_NEIGHBORS,
    SVM_KERNEL, SVM_C, RF_N_ESTIMATORS, RESULTS_DIR,
)
from evaluation import compute_metrics


def train_and_evaluate(train_features, test_features, y_train, y_test, le):
    """
    Standardize features, apply PCA (for KNN), train three classifiers,
    and evaluate each.

    Returns:
        results – dict keyed by classifier name with metrics & predictions.
    """
    print("\n" + "=" * 60)
    print("STEP 6: Training Classifiers")
    print("=" * 60)

    num_classes = len(le.classes_)

    scaler = StandardScaler()
    train_sc = scaler.fit_transform(train_features)
    test_sc  = scaler.transform(test_features)

    print(f"\n  PCA for KNN (retain {PCA_VARIANCE_RATIO*100:.0f}% variance) ...")
    pca = PCA(n_components=PCA_VARIANCE_RATIO, random_state=RANDOM_STATE)
    train_pca = pca.fit_transform(train_sc)
    test_pca  = pca.transform(test_sc)
    print(f"  {train_features.shape[1]} dims → {train_pca.shape[1]} dims")

    classifiers = {
        'CNN-SVM': {
            'clf': SVC(kernel=SVM_KERNEL, C=SVM_C, gamma='scale',
                       random_state=RANDOM_STATE),
            'tr': train_sc, 'te': test_sc,
        },
        'CNN-RF': {
            'clf': RandomForestClassifier(n_estimators=RF_N_ESTIMATORS,
                                          random_state=RANDOM_STATE, n_jobs=-1),
            'tr': train_sc, 'te': test_sc,
        },
        'CNN-KNN (PCA)': {
            'clf': KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS,
                                        metric='minkowski', p=2),
            'tr': train_pca, 'te': test_pca,
        },
    }

    results = {}

    for name, cfg in classifiers.items():
        print(f"\n  {'─' * 50}")
        print(f"  Training {name} ...")
        clf = cfg['clf']
        clf.fit(cfg['tr'], y_train)
        y_pred = clf.predict(cfg['te'])

        metrics = compute_metrics(y_test, y_pred, num_classes)
        metrics['y_pred'] = y_pred
        results[name] = metrics

        print(f"  Accuracy:    {metrics['accuracy']*100:.2f}%")
        print(f"  Precision:   {metrics['precision']*100:.2f}%")
        print(f"  Recall:      {metrics['recall']*100:.2f}%")
        print(f"  F1-Score:    {metrics['f1_score']*100:.2f}%")
        print(f"  Specificity: {metrics['avg_specificity']*100:.2f}% (avg)")
        for i, cls in enumerate(le.classes_):
            print(f"    {cls}: {metrics['specificity_per_class'][i]*100:.2f}%")

        report_str = classification_report(y_test, y_pred, target_names=le.classes_)
        print(f"\n  Classification Report ({name}):")
        print(report_str)
        
        report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
        mode = 'a' if name != list(classifiers.keys())[0] else 'w'
        with open(report_path, mode) as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Classification Report ({name})\n")
            f.write(f"{'='*60}\n")
            f.write(report_str + "\n")
            
    print(f"\n  Classification reports saved to: {os.path.join(RESULTS_DIR, 'classification_report.txt')}")

    return results
