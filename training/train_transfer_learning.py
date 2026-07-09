"""
=========================================================
Transfer Learning Training
=========================================================

Fine-tunes ResNet50 on Fashion Product Images.

"""

import tensorflow as tf
COLAB = False

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
from configs.config import OUTPUTS_DIR
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
    shuffle=True,
    colab=COLAB
)

val_dataset = create_dataset(
    val_csv,
    label_mapping,
    shuffle=False,
    colab=COLAB
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

# Freeze early layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Fine-tune last layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

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
    512,
    activation="relu"
)(x)

x = Dropout(0.5)(x)

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

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = MODELS_DIR / "fashion_resnet50.keras"

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        verbose=1
    ),

    ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

]

#Train

history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    callbacks=callbacks

)


test_csv = SUBSET_DIR / "test.csv"

test_dataset = create_dataset(
    test_csv,
    label_mapping,
    shuffle=False,
    colab=COLAB
)

test_loss, test_acc = model.evaluate(test_dataset)

print(f"\nTest Accuracy : {test_acc:.4f}")
print(f"Test Loss     : {test_loss:.4f}")

OUTPUTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.legend()

plt.savefig(
    OUTPUTS_DIR / "training_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nTraining Completed Successfully!")

print(f"\nBest Model Saved At:\n{BEST_MODEL_PATH}")