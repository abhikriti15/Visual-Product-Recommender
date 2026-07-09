"""
=========================================================
Cosine Similarity Retrieval
=========================================================

Loads stored embeddings and retrieves
top-K visually similar products.

Author : Abhikriti Saxena
"""

import pickle
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from configs.config import (
    EMBEDDINGS_DIR,
    TOP_K
)

# =====================================================
# Load Stored Data
# =====================================================

embeddings = np.load(
    EMBEDDINGS_DIR / "embeddings.npy"
)

with open(
    EMBEDDINGS_DIR / "image_paths.pkl",
    "rb"
) as f:

    image_paths = pickle.load(f)

metadata = pd.read_csv(
    EMBEDDINGS_DIR / "metadata.csv"
)


# =====================================================
# Retrieve Similar Images
# =====================================================

def retrieve_similar(query_embedding, top_k=TOP_K):

    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    )[0]

    indices = np.argsort(similarities)[::-1]

    # Remove the first result (query image itself)
    indices = indices[1: top_k + 1]

    results = []

    for idx in indices:

        results.append({

            "image_path": image_paths[idx],

            "similarity": float(similarities[idx]),

            "category": metadata.iloc[idx]["subset_category"],

            "product_name": metadata.iloc[idx]["productDisplayName"]

        })

    return results