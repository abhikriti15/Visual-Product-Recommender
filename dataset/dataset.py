"""
=========================================================
TensorFlow Dataset Loader
=========================================================

Loads images from CSV files and creates tf.data.Dataset
objects for training, validation and testing.

"""

from pathlib import Path

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

    image_path = tf.strings.join(
        [str(PROJECT_ROOT) + "/", image_path]
    )

    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, IMAGE_SIZE)

    image = tf.cast(image, tf.float32)

    image = image / 255.0

    return image, label


# =========================================================
# Create Dataset
# =========================================================

def create_dataset(csv_path, label_mapping, shuffle=True):

    df = pd.read_csv(csv_path)

    image_paths = df["image_path"].values

    labels = df["subset_category"].map(label_mapping).values

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
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