"""
=========================================================
Compare Baseline vs Fine-Tuned Recommendations
=========================================================
"""

import pickle
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from configs.config import EMBEDDINGS_DIR

# -----------------------------------------------------
# Load Baseline
# -----------------------------------------------------

baseline_embeddings = np.load(
    EMBEDDINGS_DIR / "embeddings.npy"
)

with open(
    EMBEDDINGS_DIR / "image_paths.pkl",
    "rb"
) as f:
    baseline_paths = pickle.load(f)

baseline_metadata = pd.read_csv(
    EMBEDDINGS_DIR / "metadata.csv"
)

# -----------------------------------------------------
# Load Fine-Tuned
# -----------------------------------------------------

finetuned_embeddings = np.load(
    EMBEDDINGS_DIR / "finetuned_embeddings.npy"
)

with open(
    EMBEDDINGS_DIR / "finetuned_image_paths.pkl",
    "rb"
) as f:
    finetuned_paths = pickle.load(f)

finetuned_metadata = pd.read_csv(
    EMBEDDINGS_DIR / "finetuned_metadata.csv"
)

# -----------------------------------------------------
# Query
# -----------------------------------------------------

query_index = 100

baseline_scores = cosine_similarity(
    baseline_embeddings[query_index].reshape(1,-1),
    baseline_embeddings
)[0]

finetuned_scores = cosine_similarity(
    finetuned_embeddings[query_index].reshape(1,-1),
    finetuned_embeddings
)[0]

baseline_top = baseline_scores.argsort()[::-1][1:6]
finetuned_top = finetuned_scores.argsort()[::-1][1:6]

print("="*70)
print("BASELINE RECOMMENDATIONS")
print("="*70)

for i in baseline_top:
    print(
        baseline_metadata.iloc[i]["subset_category"],
        "|",
        round(baseline_scores[i],4),
        "|",
        baseline_metadata.iloc[i]["productDisplayName"]
    )

print("\n")

print("="*70)
print("FINE-TUNED RECOMMENDATIONS")
print("="*70)

for i in finetuned_top:
    print(
        finetuned_metadata.iloc[i]["subset_category"],
        "|",
        round(finetuned_scores[i],4),
        "|",
        finetuned_metadata.iloc[i]["productDisplayName"]
    )