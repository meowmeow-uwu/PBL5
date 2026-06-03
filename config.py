"""
Configuration constants for Tomato Quality Classification.
"""

import os
from dotenv import load_dotenv

load_dotenv()

DATASET_DIR = os.getenv("DATASET_DIR", "../dataset/Dataset/Three Classes")
DATASET_CACHUA_DIR = os.getenv("DATASET_CACHUA_DIR", "../dataset/Dataset_Cachua")
RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")

IMG_SIZE = 128              # Input size
CLASS_NAMES = ['Reject', 'Ripe', 'Unripe']

RANDOM_STATE = 42
BATCH_SIZE = 32
FINE_TUNE_EPOCHS = 20

TEST_SIZE = 0.2
VAL_SIZE_FROM_TRAINVAL = 0.125  # 10% of total = 12.5% of the 80% trainval

COLOR_SPACES = ['RGB', 'HSV', 'LAB', 'YCrCb']

LEARNING_RATE = 1e-4
DROPOUT_1 = 0.5
DROPOUT_2 = 0.3
DENSE_UNITS = 512

os.makedirs(RESULTS_DIR, exist_ok=True)
