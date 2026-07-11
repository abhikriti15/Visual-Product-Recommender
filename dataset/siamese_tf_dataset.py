"""
=========================================================
TensorFlow Siamese Dataset
=========================================================

Creates tf.data.Dataset for Siamese Network.

Author : Abhikriti Saxena
"""

import tensorflow as tf
import numpy as np

from configs.config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    PROJECT_ROOT
)


# =========================================================
# Load Image
# =========================================================

def load_image(path):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image


# =========================================================
# Parse Pair
# =========================================================

def parse_pair(path1, path2, label):

    image1 = load_image(path1)

    image2 = load_image(path2)

    return (image1, image2), label


# =========================================================
# Dataset
# =========================================================

def create_dataset(
    pairs,
    labels,
    shuffle=True
):

    image1 = []

    image2 = []

    for p1, p2 in pairs:

        image1.append(
            str(PROJECT_ROOT / p1)
        )

        image2.append(
            str(PROJECT_ROOT / p2)
        )

    image1 = np.array(image1)

    image2 = np.array(image2)

    labels = np.array(labels).astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(

        (
            image1,
            image2,
            labels
        )

    )

    dataset = dataset.map(

        parse_pair,

        num_parallel_calls=tf.data.AUTOTUNE

    )

    dataset = dataset.cache()

    if shuffle:

        dataset = dataset.shuffle(
            buffer_size=512
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(1)

    return dataset