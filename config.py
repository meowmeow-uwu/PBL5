"""
Configuration constants for Experiment 4: Three-Class Tomato Quality Classification.
"""

import os

DATASET_DIR = r"e:\PBL5\Dataset\Dataset\Three Classes"
DATASET_CACHUA_DIR = r"e:\PBL5\Dataset\Dataset_Cachua"
RESULTS_DIR = r"e:\PBL5\results"

IMG_SIZE = 299              # Input size
CLASS_NAMES = ['Reject', 'Ripe', 'Unripe']

RANDOM_STATE = 42
BATCH_SIZE = 32
FINE_TUNE_EPOCHS = 30
AUGMENTATION_FACTOR = 3     # Number of augmented copies per training image

TEST_SIZE = 0.20
VAL_SIZE_FROM_TRAINVAL = 0.125  # 10% of total = 12.5% of the 80% trainval

LEARNING_RATE = 1e-4
DROPOUT_1 = 0.5
DROPOUT_2 = 0.3
DENSE_UNITS = 512

PCA_VARIANCE_RATIO = 0.95   # Retain 95% of variance
KNN_NEIGHBORS = 5

SVM_KERNEL = 'rbf'
SVM_C = 10

RF_N_ESTIMATORS = 200

os.makedirs(RESULTS_DIR, exist_ok=True)

