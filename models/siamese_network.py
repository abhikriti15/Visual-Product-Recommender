"""
=========================================================
Siamese Network
=========================================================

Learns visual similarity between fashion products.

Author: Abhikriti Saxena
"""

import tensorflow as tf

from tensorflow.keras import Model

from tensorflow.keras.layers import (
    Input,
    Dense,
    Lambda
)

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.layers import (
    GlobalAveragePooling2D
)

# =========================================================
# Encoder Network
# =========================================================

def create_encoder():

    base_model = ResNet50(

        weights=None,

        include_top=False,

        input_shape=(224, 224, 3)

    )

    encoder = Model(

        inputs=base_model.input,

        outputs=GlobalAveragePooling2D()(base_model.output),

        name="Encoder"

    )

    return encoder

# =========================================================
# Load Fine-Tuned Weights
# =========================================================

from tensorflow.keras.models import load_model

def load_finetuned_encoder():

    model = load_model(
        "models/saved_models/fashion_resnet50.keras"
    )

    encoder = Model(

        inputs=model.input,

        outputs=model.get_layer(
            "global_average_pooling2d"
        ).output

    )

    return encoder

# =========================================================
# Test
# =========================================================

if __name__ == "__main__":

    encoder = load_finetuned_encoder()

    encoder.summary()