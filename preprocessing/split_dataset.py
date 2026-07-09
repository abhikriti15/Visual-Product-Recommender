"""
=========================================================
Split Dataset into Train / Validation / Test
=========================================================
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from configs.config import SUBSET_DIR, RANDOM_STATE

# ---------------------------------------------
# Load subset metadata
# ---------------------------------------------

subset_csv = SUBSET_DIR / "subset.csv"

df = pd.read_csv(subset_csv)

print("=" * 60)
print("SPLITTING DATASET")
print("=" * 60)

print(f"Total Images : {len(df)}")

# ---------------------------------------------
# Train + Temp
# ---------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["subset_category"],
    random_state=RANDOM_STATE,
)

# ---------------------------------------------
# Validation + Test
# ---------------------------------------------

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["subset_category"],
    random_state=RANDOM_STATE,
)

# ---------------------------------------------
# Save
# ---------------------------------------------

train_df.to_csv(SUBSET_DIR / "train.csv", index=False)
val_df.to_csv(SUBSET_DIR / "val.csv", index=False)
test_df.to_csv(SUBSET_DIR / "test.csv", index=False)

print("\nSaved")

print("Train :", len(train_df))
print("Validation :", len(val_df))
print("Test :", len(test_df))

print("\nCategory Distribution\n")

print(train_df["subset_category"].value_counts().sort_index())