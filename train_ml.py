"""
Traditional ML Training: Feature extraction + GridSearchCV for SVM, RF, KNN, GNB.
"""

import os
import numpy as np
import joblib
import time

from config import RANDOM_STATE
from evaluation import compute_metrics
import preprocessing
from statistical_features import extract_statistical_features

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def get_features(X_rgb, cs):
    """Extract 12D statistical features for a given color space."""
    feats = []
    for img in X_rgb:
        feats.append(extract_statistical_features(preprocessing.convert_color_spaces(img)[cs]))
    return np.array(feats)


def train_traditional_ml(X_tr_rgb, X_te_rgb, y_tr, y_te, color_space, num_classes, le, CS_RESULTS_DIR):
    """Train SVM, RF, KNN, GNB with GridSearchCV and evaluate on test set."""
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
                'kernel': ['rbf', 'poly', 'linear']
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
        t0_train = time.time()
        grid = GridSearchCV(config['model'], config['params'], cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
        grid.fit(X_tr_feat_sc, y_tr)
        t1_train = time.time()
        best_model = grid.best_estimator_
        best_models[name] = best_model
        
        print(f"    Best params: {grid.best_params_}")
        
        t0_inf = time.time()
        y_pred = best_model.predict(X_te_feat_sc)
        t1_inf = time.time()
        
        metrics = compute_metrics(y_te, y_pred, num_classes)
        metrics['y_pred'] = y_pred
        metrics['best_params'] = grid.best_params_
        metrics['train_time'] = t1_train - t0_train
        metrics['inference_time'] = t1_inf - t0_inf
        
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
