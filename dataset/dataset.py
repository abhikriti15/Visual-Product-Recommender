"""
=========================================================
TensorFlow Dataset Loader
=========================================================

Loads images from CSV files and creates tf.data.Dataset
objects for training, validation and testing.

Works in:
1. Local VS Code
2. Google Colab
"""

import pandas as pd
import tensorflow as tf

from configs.config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    PROJECT_ROOT,
)

# =========================================================
# Image Loading
# =========================================================

def load_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, IMAGE_SIZE)

    image = tf.cast(image, tf.float32)

    return image, label


# =========================================================
# Create Dataset
# =========================================================

def create_dataset(
    csv_path,
    label_mapping,
    shuffle=True,
    colab=False
):

    df = pd.read_csv(csv_path)

    # --------------------------------------------
    # Image Paths
    # --------------------------------------------

    if colab:

        df["image_path"] = df["id"].apply(
            lambda x: f"/content/myntradataset/images/{x}.jpg"
        )

    else:

        df["image_path"] = df["image_path"].apply(
            lambda x: str(PROJECT_ROOT / x)
        )

    labels = df["subset_category"].map(
        label_mapping
    ).values

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            df["image_path"].values,
            labels
        )
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:

        dataset = dataset.shuffle(
            buffer_size=len(df),
            reshuffle_each_iteration=True
        )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset