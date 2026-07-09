"""
=========================================================
Visual Product Recommendation System
Configuration File
=========================================================

This file contains all configurable parameters used
throughout the project.

Author: Abhikriti Saxena
"""

from pathlib import Path

# =========================================================
# Project Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

RAW_IMAGE_DIR = RAW_DATA_DIR / "images"

RAW_CSV_PATH = RAW_DATA_DIR / "styles.csv"

PROCESSED_DIR = DATA_DIR / "processed"

SUBSET_DIR = DATA_DIR / "subset"

EMBEDDINGS_DIR = DATA_DIR / "embeddings"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"

# =========================================================
# Image Settings
# =========================================================

IMAGE_SIZE = (224, 224)

IMAGE_CHANNELS = 3

# =========================================================
# Dataset Settings
# =========================================================

RANDOM_STATE = 42

SAMPLES_PER_CATEGORY = 300

SELECTED_CATEGORIES = [
    "Tshirts",
    "Shirts",
    "Casual Shoes",
    "Sports Shoes",
    "Heels",
    "Watches",
    "Handbags",
    "Kurtas",
    "Tops",
    "Sandals",
]

# =========================================================
# Train / Validation / Test Split
# =========================================================

TRAIN_SPLIT = 0.70

VALID_SPLIT = 0.15

TEST_SPLIT = 0.15

# =========================================================
# Model Settings
# =========================================================

BATCH_SIZE = 32

EPOCHS = 15

LEARNING_RATE = 1e-4

EMBEDDING_DIM = 2048

TOP_K = 5

# =========================================================
# Random Seed
# =========================================================

SEED = 42