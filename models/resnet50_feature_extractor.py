import numpy as np

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

from configs.config import IMAGE_SIZE, MODELS_DIR


class ResNet50FeatureExtractor:

    def __init__(self):

        model = load_model(
            MODELS_DIR / "fashion_resnet50.keras",
            compile=False
        )

        self.model = Model(
            inputs=model.input,
            outputs=model.get_layer(
                "global_average_pooling2d"
            ).output
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
        ).flatten()

        embedding = embedding / np.linalg.norm(embedding)

        return embedding    