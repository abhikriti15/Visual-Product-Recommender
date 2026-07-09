"""
=========================================================
Transfer Learning Training
=========================================================

Fine-tunes ResNet50 on Fashion Product Images.

Author: Abhikriti Saxena
"""

import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from dataset.dataset import create_dataset

from configs.config import (
    SUBSET_DIR,
    MODELS_DIR,
    IMAGE_SIZE,
    LEARNING_RATE,
    EPOCHS
)

import pandas as pd

# =====================================================
# Load Metadata
# =====================================================

train_csv = SUBSET_DIR / "train.csv"

val_csv = SUBSET_DIR / "val.csv"

train_df = pd.read_csv(train_csv)

classes = sorted(train_df["subset_category"].unique())

label_mapping = {
    label: idx
    for idx, label in enumerate(classes)
}

num_classes = len(classes)

print(label_mapping)


train_dataset = create_dataset(
    train_csv,
    label_mapping,
    shuffle=True
)

val_dataset = create_dataset(
    val_csv,
    label_mapping,
    shuffle=False
)


base_model = ResNet50(

    weights="imagenet",

    include_top=False,

    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )

)


for layer in base_model.layers:

    layer.trainable = False

inputs = Input(shape=(
    IMAGE_SIZE[0],
    IMAGE_SIZE[1],
    3
))

x = preprocess_input(inputs)

x = base_model(
    x,
    training=False
)

x = GlobalAveragePooling2D()(x)

x = Dense(
    256,
    activation="relu"
)(x)

x = Dropout(
    0.3
)(x)

outputs = Dense(
    num_classes,
    activation="softmax"
)(x)

model = Model(
    inputs,
    outputs
)

model.compile(

    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

model.summary()

#Train

history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS

)

#Save Model

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

model.save(

    MODELS_DIR /
    "fashion_resnet50.keras"

)

print("\nModel Saved Successfully!")