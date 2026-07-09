"""
=========================================================
ResNet50 Feature Extractor
=========================================================

Loads a pretrained ResNet50 model and converts images
into 2048-dimensional feature embeddings.

"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

from configs.config import IMAGE_SIZE


class ResNet50FeatureExtractor:

    def __init__(self):

        self.model = ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg"
        )

    def extract(self, image_path):

        img = image.load_img(
            image_path,
            target_size=IMAGE_SIZE
        )

        img = image.img_to_array(img)

        img = np.expand_dims(img, axis=0)

        img = preprocess_input(img)

        embedding = self.model.predict(
            img,
            verbose=0
        )

        embedding = embedding.flatten()

        # L2 Normalization
        embedding = embedding / np.linalg.norm(embedding)

        return embedding