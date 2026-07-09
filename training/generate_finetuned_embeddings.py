"""
=========================================================
Generate Fine-Tuned Embeddings
=========================================================

Uses the trained ResNet50 model to generate feature
embeddings for all products.

"""

import pickle
import numpy as np
import pandas as pd

from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

from configs.config import (
    SUBSET_DIR,
    MODELS_DIR,
    EMBEDDINGS_DIR,
    IMAGE_SIZE
)

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

from tqdm import tqdm

# =====================================================
# Create Output Folder
# =====================================================

EMBEDDINGS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# Load Trained Model
# =====================================================

print("=" * 60)
print("LOADING FINE-TUNED MODEL")
print("=" * 60)

model_path = MODELS_DIR / "fashion_resnet50.keras"

model = load_model(model_path)

print("\nModel Loaded Successfully!")

print(model.summary())


# =====================================================
# Create Feature Extractor
# =====================================================

print("\nCreating Feature Extractor...")

feature_extractor = Model(
    inputs=model.input,
    outputs=model.get_layer("global_average_pooling2d").output
)

print("Feature Extractor Created!")

print("\nEmbedding Shape:")

print(feature_extractor.output_shape)

# =====================================================
# Load Metadata
# =====================================================

csv_path = SUBSET_DIR / "subset.csv"

df = pd.read_csv(csv_path)

print("=" * 60)
print("GENERATING FINE-TUNED EMBEDDINGS")
print("=" * 60)

print(f"Total Images : {len(df)}")

print("\nFirst 5 Images:")

print(df[["id", "subset_category"]].head())

# =====================================================
# Storage
# =====================================================

embeddings = []

image_paths = []

# =====================================================
# Feature Extraction Function
# =====================================================

def extract_embedding(image_path):

    img = image.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    img = image.img_to_array(img)

    img = np.expand_dims(
        img,
        axis=0
    )

    img = preprocess_input(img)

    embedding = feature_extractor.predict(
        img,
        verbose=0
    )

    embedding = embedding.flatten()

    embedding = embedding / np.linalg.norm(
        embedding
    )

    return embedding

# =====================================================
# Generate Embeddings
# =====================================================

from pathlib import Path
from tqdm import tqdm

print("\nGenerating Embeddings...\n")

for _, row in tqdm(df.iterrows(), total=len(df)):

    image_path = Path(row["image_path"])

    embedding = extract_embedding(image_path)

    embeddings.append(embedding)

    image_paths.append(str(image_path))

# =====================================================
# Convert to NumPy
# =====================================================

embeddings = np.array(embeddings)

print("\nEmbedding Matrix Shape:")

print(embeddings.shape)

# =====================================================
# Save Embeddings
# =====================================================

np.save(
    EMBEDDINGS_DIR / "finetuned_embeddings.npy",
    embeddings
)

with open(
    EMBEDDINGS_DIR / "finetuned_image_paths.pkl",
    "wb"
) as f:

    pickle.dump(image_paths, f)

df.to_csv(
    EMBEDDINGS_DIR / "finetuned_metadata.csv",
    index=False
)

print("\nFine-Tuned Embeddings Saved Successfully!")

print("\nSaved Location:")

print(EMBEDDINGS_DIR)