from configs.config import *

print("=" * 60)
print("VISUAL PRODUCT RECOMMENDER")
print("=" * 60)

print("\nProject Root")
print(PROJECT_ROOT)

print("\nSelected Categories")

for category in SELECTED_CATEGORIES:
    print("•", category)

print("\nImage Size:", IMAGE_SIZE)

print("Batch Size:", BATCH_SIZE)

print("Top K:", TOP_K)