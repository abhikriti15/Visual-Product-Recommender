"""
=========================================================
Generate Image Embeddings
=========================================================

Reads subset.csv,
generates ResNet50 embeddings,
and saves them for similarity search.

Author : Abhikriti Saxena
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from configs.config import (
    SUBSET_DIR,
    EMBEDDINGS_DIR
)

from models.resnet50_feature_extractor import (
    ResNet50FeatureExtractor
)


# =====================================================
# Create Output Folder
# =====================================================

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Load Metadata
# =====================================================

csv_path = SUBSET_DIR / "subset.csv"

df = pd.read_csv(csv_path)

print("=" * 70)
print("GENERATING IMAGE EMBEDDINGS")
print("=" * 70)

print(f"Total Images : {len(df)}")

# =====================================================
# Load Feature Extractor
# =====================================================

extractor = ResNet50FeatureExtractor()

embeddings = []
image_paths = []

# =====================================================
# Generate Embeddings
# =====================================================

for _, row in tqdm(df.iterrows(), total=len(df)):

    image_path = Path(row["image_path"])

    embedding = extractor.extract(image_path)

    embeddings.append(embedding)

    image_paths.append(str(image_path))

# =====================================================
# Convert
# =====================================================

embeddings = np.array(embeddings)

print("\nEmbedding Matrix Shape:")

print(embeddings.shape)

# =====================================================
# Save
# =====================================================

np.save(
    EMBEDDINGS_DIR / "embeddings.npy",
    embeddings
)

with open(
    EMBEDDINGS_DIR / "image_paths.pkl",
    "wb"
) as f:

    pickle.dump(image_paths, f)

df.to_csv(
    EMBEDDINGS_DIR / "metadata.csv",
    index=False
)

print("\nEmbeddings Saved Successfully!")

print("\nLocation:")

print(EMBEDDINGS_DIR)