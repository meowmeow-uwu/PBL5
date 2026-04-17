"""
Orchestrates the full pipeline:
  1. Load & preprocess (background cancellation)
  2. Split 70 / 10 / 20
  3. Data augmentation (training set)
  4. Fine-tune InceptionV3 + extract 2048-dim features
  5. Train SVM, RF, KNN (PCA) classifiers
  6. Evaluate & visualize
"""

import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import RANDOM_STATE, TEST_SIZE, VAL_SIZE_FROM_TRAINVAL, RESULTS_DIR
from preprocessing import load_and_preprocess_images
from augmentation import create_augmented_data
from model import fine_tune_and_extract_features
from classifiers import train_and_evaluate
from visualization import (
    plot_confusion_matrices, plot_comparison_chart, print_summary_table,
)

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_STATE)

def split_dataset(images, labels):
    """Stratified three-way split."""
    print("\n" + "=" * 60)
    print("STEP 2: Splitting Dataset (70% Train / 10% Val / 20% Test)")
    print("=" * 60)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_tv, X_te, y_tv, y_te = train_test_split(
        images, y, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_STATE,
    )

    X_tr, X_v, y_tr, y_v = train_test_split(
        X_tv, y_tv, test_size=VAL_SIZE_FROM_TRAINVAL,
        stratify=y_tv, random_state=RANDOM_STATE,
    )

    n = len(images)
    print(f"  Train: {len(X_tr)} ({len(X_tr)/n*100:.1f}%)")
    print(f"  Val:   {len(X_v)}  ({len(X_v)/n*100:.1f}%)")
    print(f"  Test:  {len(X_te)} ({len(X_te)/n*100:.1f}%)")

    for tag, ys in [("Train", y_tr), ("Val", y_v), ("Test", y_te)]:
        counts = np.bincount(ys)
        dist = ", ".join(f"{le.classes_[i]}: {counts[i]}" for i in range(len(counts)))
        print(f"    {tag}: {dist}")

    return X_tr, X_v, X_te, y_tr, y_v, y_te, le

def main():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 4: Three-Class Tomato Quality Classification")
    print("  InceptionV3 (Fine-tuned) + SVM / RF / KNN (PCA)")
    print("=" * 60 + "\n")

    images, labels = load_and_preprocess_images()

    X_tr, X_v, X_te, y_tr, y_v, y_te, le = split_dataset(images, labels)

    X_tr_aug, y_tr_aug = create_augmented_data(X_tr, y_tr)

    tr_feat, te_feat, history = fine_tune_and_extract_features(
        X_tr_aug, y_tr_aug, X_v, y_v,
        X_tr, X_te, y_tr, y_te,
        num_classes=len(le.classes_), le=le,
    )

    results = train_and_evaluate(tr_feat, te_feat, y_tr, y_te, le)

    print("\n" + "=" * 60)
    print("STEP 7: Generating Visualizations")
    print("=" * 60)
    plot_confusion_matrices(results, y_te, le)
    plot_comparison_chart(results)
    print_summary_table(results)

    print(f"\n  All outputs saved to: {RESULTS_DIR}")
    print("  Done!\n")


if __name__ == "__main__":
    main()
