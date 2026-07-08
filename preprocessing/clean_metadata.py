from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images"
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "styles.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "clean_styles.csv"

print("=" * 60)
print("CLEANING METADATA")
print("=" * 60)

df = pd.read_csv(CSV_PATH, on_bad_lines="skip")

print(f"Original records : {len(df):,}")

# Keep only rows whose image exists
df = df[df["id"].apply(lambda x: (IMAGE_DIR / f"{x}.jpg").exists())]

print(f"Remaining records: {len(df):,}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved cleaned metadata to:\n{OUTPUT_PATH}")