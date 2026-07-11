"""
=========================================================
Generate Siamese Embeddings
=========================================================

Loads the trained Siamese model and generates embeddings
for all subset images.

"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from tensorflow.keras.models import load_model, Model

from configs.config import (
    PROJECT_ROOT,
    IMAGE_SIZE,
    EMBEDDINGS_DIR
)

# =========================================================
# Paths
# =========================================================

model_path = PROJECT_ROOT / "models" / "saved_models" / "siamese_model.keras"

subset_csv = PROJECT_ROOT / "data" / "subset" / "subset.csv"

output_embeddings = EMBEDDINGS_DIR / "siamese_embeddings.npy"

output_paths = EMBEDDINGS_DIR / "siamese_image_paths.npy"

# =========================================================
# Load Siamese Model
# =========================================================

print("=" * 60)
print("LOADING SIAMESE MODEL")
print("=" * 60)

siamese_model = load_model(
    model_path,
    compile=False,
    safe_mode=False
)

print("Model Loaded Successfully!")

# =========================================================
# Extract Encoder
# =========================================================

encoder = siamese_model.get_layer("functional")

print("\nEncoder Extracted!")
encoder.summary()

# =========================================================
# Load Dataset
# =========================================================

df = pd.read_csv(subset_csv)

image_paths = df["image_path"].tolist()

print("\nTotal Images :", len(image_paths))

# =========================================================
# Generate Embeddings
# =========================================================

embeddings = []

print("\nGenerating Embeddings...\n")

for path in tqdm(image_paths):

    full_path = PROJECT_ROOT / path

    image = cv2.imread(str(full_path))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, IMAGE_SIZE)

    image = image.astype(np.float32)

    image = np.expand_dims(image, axis=0)

    embedding = encoder.predict(image, verbose=0)

    embeddings.append(embedding.squeeze())

embeddings = np.array(embeddings)

# =========================================================
# Save
# =========================================================

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

np.save(output_embeddings, embeddings)

np.save(output_paths, np.array(image_paths))

print("\nEmbedding Shape :", embeddings.shape)

print("\nEmbeddings Saved Successfully!")

print("\nLocation:")

print(output_embeddings)