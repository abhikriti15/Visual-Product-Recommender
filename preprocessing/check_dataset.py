from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm

# =====================================================
# Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images"
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "styles.csv"

# =====================================================
# Load Metadata
# =====================================================

print("=" * 60)
print("VISUAL PRODUCT RECOMMENDER")
print("DATASET VERIFICATION")
print("=" * 60)

if not CSV_PATH.exists():
    raise FileNotFoundError(f"styles.csv not found at:\n{CSV_PATH}")

df = pd.read_csv(CSV_PATH, on_bad_lines="skip")

print(f"\nMetadata records : {len(df):,}")

# =====================================================
# Count Images
# =====================================================

image_files = list(IMAGE_DIR.glob("*.jpg"))

print(f"Image files      : {len(image_files):,}")

# =====================================================
# Missing Images
# =====================================================

missing_images = []

for image_id in tqdm(df["id"], desc="Checking image paths"):
    image_path = IMAGE_DIR / f"{image_id}.jpg"

    if not image_path.exists():
        missing_images.append(image_id)

print(f"\nMissing images   : {len(missing_images)}")

# =====================================================
# Corrupted Images
# =====================================================

corrupted = []

for image_path in tqdm(image_files, desc="Checking corrupted images"):
    try:
        with Image.open(image_path) as img:
            img.verify()

    except Exception:
        corrupted.append(image_path.name)

print(f"Corrupted images : {len(corrupted)}")

print("=" * 60)

if missing_images:
    print("\nFirst 10 Missing IDs:")
    print(missing_images[:10])

if corrupted:
    print("\nFirst 10 Corrupted Images:")
    print(corrupted[:10])

print("\nDataset verification completed successfully.")