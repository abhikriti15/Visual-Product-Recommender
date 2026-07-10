"""
=========================================================
Siamese Dataset
=========================================================

Creates positive and negative image pairs for
Siamese Network Training.

Author : Abhikriti Saxena
"""

import random
import pandas as pd

from configs.config import (
    SUBSET_DIR,
    RANDOM_STATE
)

random.seed(RANDOM_STATE)


def create_pairs():

    train_csv = SUBSET_DIR / "train.csv"

    df = pd.read_csv(train_csv)

    print("=" * 60)
    print("CREATING SIAMESE PAIRS")
    print("=" * 60)

    print(f"Training Images : {len(df)}")

    # ---------------------------------------
    # Group images by category
    # ---------------------------------------

    grouped = {}

    for _, row in df.iterrows():

        category = row["subset_category"]

        if category not in grouped:

            grouped[category] = []

        grouped[category].append(row)

    print("\nCategories :")

    print(list(grouped.keys()))

    pairs = []

    labels = []

    # ---------------------------------------
    # Positive Pairs
    # ---------------------------------------

    for category in grouped:

        images = grouped[category]

        MAX_POSITIVE_PER_CLASS = 1000

        count = 0

        for i in range(len(images)):

            for j in range(i + 1, len(images)):

                pairs.append(
                    (
                        images[i]["image_path"],
                        images[j]["image_path"]
                    )
                )

                labels.append(1)

                count += 1

                if count >= MAX_POSITIVE_PER_CLASS:
                    break

            if count >= MAX_POSITIVE_PER_CLASS:
                break

    positive_pairs = len(labels)

    print(f"\nPositive Pairs : {positive_pairs}")

    # ---------------------------------------
    # Negative Pairs
    # ---------------------------------------

    categories = list(grouped.keys())

    while len(labels) < positive_pairs * 2:

        cat1, cat2 = random.sample(categories, 2)

        img1 = random.choice(
            grouped[cat1]
        )["image_path"]

        img2 = random.choice(
            grouped[cat2]
        )["image_path"]

        pairs.append(

            (
                img1,
                img2
            )

        )

        labels.append(0)

    print(f"Negative Pairs : {len(labels)-positive_pairs}")

    print(f"Total Pairs    : {len(labels)}")

    return pairs, labels


if __name__ == "__main__":

    pairs, labels = create_pairs()

    print("\nFirst Five Pairs\n")

    for i in range(5):

        print(pairs[i])

        print("Label :", labels[i])

        print("-" * 60)