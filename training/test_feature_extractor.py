from pathlib import Path

from models.resnet50_feature_extractor import ResNet50FeatureExtractor

IMAGE_PATH = Path("data/subset/Tshirts")

image_file = next(IMAGE_PATH.glob("*.jpg"))

extractor = ResNet50FeatureExtractor()

embedding = extractor.extract(str(image_file))

print("=" * 60)

print("Embedding Shape:", embedding.shape)

print()

print("First 10 Values:")

print(embedding[:10])