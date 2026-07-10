"""
=========================================================
Visual Comparison of Recommendation Systems
=========================================================

Displays query image along with recommendations from:
1. Baseline ResNet50
2. Fine-Tuned ResNet50

Author: Abhikriti Saxena
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from configs.config import EMBEDDINGS_DIR

# =====================================================
# Load Baseline Data
# =====================================================

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

# =====================================================
# Load Fine-Tuned Data
# =====================================================

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

print("Data Loaded Successfully!")

# =====================================================
# Select Query Image
# =====================================================

query_index = 100

baseline_scores = cosine_similarity(
    baseline_embeddings[query_index].reshape(1, -1),
    baseline_embeddings
)[0]

finetuned_scores = cosine_similarity(
    finetuned_embeddings[query_index].reshape(1, -1),
    finetuned_embeddings
)[0]

baseline_top = baseline_scores.argsort()[::-1][:5]

finetuned_top = finetuned_scores.argsort()[::-1][:5]

# =====================================================
# Create Figure
# =====================================================

fig, axes = plt.subplots(3, 6, figsize=(18, 10))

# Remove all axes
for ax in axes.ravel():
    ax.axis("off")

fig.suptitle(
    "Baseline vs Fine-Tuned Product Recommendation",
    fontsize=20,
    fontweight="bold"
)


# Query Image

axes[0, 2].imshow(Image.open(baseline_paths[query_index]))
axes[0, 2].set_title("Query Image", fontsize=14, fontweight="bold")


# =====================================================
# Baseline Recommendations
# =====================================================

axes[1, 0].text(
    -0.6,
    0.5,
    "Baseline",
    fontsize=16,
    fontweight="bold",
    transform=axes[1,0].transAxes
)

for i, idx in enumerate(baseline_top):
    axes[1, i+1].imshow(Image.open(baseline_paths[idx]))
    axes[1, i+1].set_title(
    f"{baseline_metadata.iloc[idx]['subset_category']}\n{baseline_scores[idx]:.2f}",
    fontsize=9
    )
    axes[1, i+1].axis("off")


# =====================================================
# Fine-Tuned Recommendations
# =====================================================

axes[2, 0].text(
    -0.6,
    0.5,
    "Fine-Tuned",
    fontsize=16,
    fontweight="bold",
    transform=axes[2,0].transAxes
)

for i, idx in enumerate(finetuned_top):
    axes[2, i+1].imshow(Image.open(finetuned_paths[idx]))
    axes[2, i+1].set_title(
        f"{finetuned_metadata.iloc[idx]['subset_category']}\n{finetuned_scores[idx]:.2f}",
        fontsize=9
    )
    axes[2, i+1].axis("off")

plt.tight_layout()

from configs.config import OUTPUTS_DIR

figure_dir = OUTPUTS_DIR / "figures"

figure_dir.mkdir(
    parents=True,
    exist_ok=True
)

plt.tight_layout()

save_path = figure_dir / "comparison.png"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nComparison saved to:\n{save_path}")

plt.show()