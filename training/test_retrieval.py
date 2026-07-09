"""
=========================================================
Test Image Retrieval
=========================================================
"""

from pathlib import Path

from models.resnet50_feature_extractor import (
    ResNet50FeatureExtractor
)

from retrieval.cosine_similarity import retrieve_similar

from utils.visualize_results import visualize

# -----------------------------------------------------

QUERY_FOLDER = Path("data/subset/Tshirts")

query_image = next(QUERY_FOLDER.glob("*.jpg"))

print("Query Image:")

print(query_image)

# -----------------------------------------------------

extractor = ResNet50FeatureExtractor()

embedding = extractor.extract(str(query_image))

results = retrieve_similar(embedding)

# Print Results

for i, result in enumerate(results, start=1):

    print()

    print(i)

    print(result)

# Visualize

visualize(query_image, results)