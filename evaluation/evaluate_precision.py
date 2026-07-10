"""
=========================================================
Evaluate Recommendation Precision@K
=========================================================

Computes Precision@5 for both:
1. Baseline Embeddings
2. Fine-Tuned Embeddings

Author: Abhikriti Saxena
"""

import pickle
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from configs.config import EMBEDDINGS_DIR

# =====================================================
# Load Baseline
# =====================================================

baseline_embeddings = np.load(
    EMBEDDINGS_DIR / "embeddings.npy"
)

baseline_metadata = pd.read_csv(
    EMBEDDINGS_DIR / "metadata.csv"
)

# =====================================================
# Load Fine-Tuned
# =====================================================

finetuned_embeddings = np.load(
    EMBEDDINGS_DIR / "finetuned_embeddings.npy"
)

finetuned_metadata = pd.read_csv(
    EMBEDDINGS_DIR / "finetuned_metadata.csv"
)

print("Data Loaded Successfully!")

# =====================================================
# Precision@K
# =====================================================

def precision_at_k(
    embeddings,
    metadata,
    k=5
):

    scores = []

    for idx in range(len(metadata)):

        query_category = metadata.iloc[idx]["subset_category"]

        similarity = cosine_similarity(
            embeddings[idx].reshape(1, -1),
            embeddings
        )[0]

        # Exclude query image
        top_indices = similarity.argsort()[::-1][1:k+1]

        correct = 0

        for i in top_indices:

            if metadata.iloc[i]["subset_category"] == query_category:
                correct += 1

        scores.append(correct / k)

    return np.mean(scores)

# =====================================================
# Evaluate Baseline
# =====================================================

print("=" * 60)
print("EVALUATING BASELINE")
print("=" * 60)

baseline_precision = precision_at_k(
    baseline_embeddings,
    baseline_metadata,
    k=5
)

print(f"\nBaseline Precision@5 : {baseline_precision:.4f}")

# =====================================================
# Evaluate Fine-Tuned
# =====================================================

print("\n" + "=" * 60)
print("EVALUATING FINE-TUNED MODEL")
print("=" * 60)

finetuned_precision = precision_at_k(
    finetuned_embeddings,
    finetuned_metadata,
    k=5
)

print(f"\nFine-Tuned Precision@5 : {finetuned_precision:.4f}")

# =====================================================
# Comparison
# =====================================================

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

improvement = (
    finetuned_precision - baseline_precision
) * 100

print(f"Baseline Precision@5   : {baseline_precision:.4f}")
print(f"Fine-Tuned Precision@5 : {finetuned_precision:.4f}")
print(f"Improvement            : {improvement:.2f}%")