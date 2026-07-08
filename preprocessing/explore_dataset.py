from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = PROJECT_ROOT / "data" / "processed" / "clean_styles.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv(CSV_PATH)

print("=" * 60)
print("DATASET EXPLORATION")
print("=" * 60)

print(f"\nTotal Records : {len(df):,}")

print("\nColumns:\n")
print(df.columns.tolist())

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nMaster Categories:\n")
print(df["masterCategory"].value_counts())

print("\nTop 20 Article Types:\n")
print(df["articleType"].value_counts().head(20))

print("\nTop 20 Sub Categories:\n")
print(df["subCategory"].value_counts().head(20))

# =====================================================
# Master Category Plot
# =====================================================

plt.figure(figsize=(10,6))

df["masterCategory"].value_counts().plot(kind="bar")

plt.title("Master Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(OUTPUT_DIR/"master_category_distribution.png")

plt.close()

# =====================================================
# Article Type Plot
# =====================================================

plt.figure(figsize=(14,7))

df["articleType"].value_counts().head(15).plot(kind="bar")

plt.title("Top 15 Article Types")
plt.xlabel("Article Type")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(OUTPUT_DIR/"article_type_distribution.png")

plt.close()

print("\nPlots saved inside outputs/figures/")