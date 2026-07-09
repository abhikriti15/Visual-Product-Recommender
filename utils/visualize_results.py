"""
=========================================================
Visualize Retrieval Results
=========================================================
"""

import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from configs.config import PROJECT_ROOT


def visualize(query_image, results):

    plt.figure(figsize=(18, 5))

    # ----------------------------
    # Query Image
    # ----------------------------

    plt.subplot(1, len(results) + 1, 1)

    img = Image.open(query_image)

    plt.imshow(img)

    plt.title("Query")

    plt.axis("off")

    # ----------------------------
    # Similar Images
    # ----------------------------

    for i, result in enumerate(results):

        image_path = PROJECT_ROOT / result["image_path"]

        img = Image.open(image_path)

        plt.subplot(1, len(results) + 1, i + 2)

        plt.imshow(img)

        plt.title(
            f"{result['category']}\n{result['similarity']:.2f}",
            fontsize=9
        )

        plt.axis("off")

    plt.tight_layout()

    plt.show()