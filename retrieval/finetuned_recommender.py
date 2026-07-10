"""
=========================================================
Fine-Tuned Product Recommendation
=========================================================

Uses fine-tuned ResNet50 embeddings
to recommend visually similar products.

Author : Abhikriti Saxena
"""

import pickle
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

from configs.config import EMBEDDINGS_DIR

# =====================================================
# Load Data
# =====================================================

print("="*60)
print("LOADING FINE-TUNED EMBEDDINGS")
print("="*60)

embeddings = np.load(
    EMBEDDINGS_DIR /
    "finetuned_embeddings.npy"
)

with open(
    EMBEDDINGS_DIR /
    "finetuned_image_paths.pkl",
    "rb"
) as f:

    image_paths = pickle.load(f)

metadata = pd.read_csv(
    EMBEDDINGS_DIR /
    "finetuned_metadata.csv"
)

print("\nEmbeddings Shape :")

print(embeddings.shape)

print("\nTotal Images :")

print(len(metadata))

# =====================================================
# Select Query Image
# =====================================================

query_index = 100

query_embedding = embeddings[
    query_index
].reshape(1,-1)

similarities = cosine_similarity(
    query_embedding,
    embeddings
)[0]

top_indices = similarities.argsort()[::-1][:5]

print("\nTop Similar Products\n")

for rank,index in enumerate(
    top_indices,
    start=1
):

    print(rank)

    print(
        "Category :",
        metadata.iloc[index]["subset_category"]
    )

    print(
        "Similarity :",
        round(similarities[index],4)
    )

    print(
        "Product :",
        metadata.iloc[index]["productDisplayName"]
    )

    print(
        "Image :",
        image_paths[index]
    )

    print("-"*60)