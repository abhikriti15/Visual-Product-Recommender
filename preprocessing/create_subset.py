"""
=========================================================
Create Balanced Dataset Subset
=========================================================
"""

import shutil
import pandas as pd
from pathlib import Path

from configs.config import (
    PROCESSED_DIR,
    RAW_IMAGE_DIR,
    SUBSET_DIR,
    SELECTED_CATEGORIES,
    SAMPLES_PER_CATEGORY,
    RANDOM_STATE,
)

# =====================================================
# Load Clean Metadata
# =====================================================

CSV_PATH = PROCESSED_DIR / "clean_styles.csv"

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("CREATING BALANCED DATASET")
print("=" * 70)

subset_list = []

# =====================================================
# Create Subset
# =====================================================

for category in SELECTED_CATEGORIES:

    print(f"\nProcessing : {category}")

    category_df = df[df["articleType"] == category]

    available = len(category_df)

    print(f"Available Images : {available}")

    if available == 0:
        print("Skipping...")
        continue

    sample_size = min(available, SAMPLES_PER_CATEGORY)

    sampled_df = category_df.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    )

    category_folder = SUBSET_DIR / category

    category_folder.mkdir(parents=True, exist_ok=True)

    copied = 0

    for _, row in sampled_df.iterrows():

        image_name = f"{row['id']}.jpg"

        source = RAW_IMAGE_DIR / image_name

        destination = category_folder / image_name

        if source.exists():

            shutil.copy2(source, destination)

            copied += 1

    sampled_df = sampled_df.copy()

    sampled_df["subset_category"] = category

    sampled_df["image_path"] = sampled_df["id"].apply(
        lambda x: f"data/subset/{category}/{x}.jpg"
)

    subset_list.append(sampled_df)

    print(f"Copied Images : {copied}")

# =====================================================
# Save Metadata
# =====================================================

subset_df = pd.concat(subset_list, ignore_index=True)

subset_csv = SUBSET_DIR / "subset.csv"

subset_df.to_csv(subset_csv, index=False)

print("\n" + "=" * 70)
print("Subset Creation Completed")
print("=" * 70)

print(f"\nTotal Images : {len(subset_df)}")

print(f"\nMetadata Saved :")

print(subset_csv)