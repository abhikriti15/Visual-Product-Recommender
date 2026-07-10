"""
=========================================================
Train Siamese Network
=========================================================

Trains the Siamese Network using
Contrastive Loss.

Author : Abhikriti Saxena
"""

import tensorflow as tf

from tensorflow.keras import Model

from tensorflow.keras.layers import (
    Input,
    Lambda
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from models.siamese_network import (
    load_finetuned_encoder
)

from dataset.siamese_dataset import (
    create_pairs
)

from dataset.siamese_tf_dataset import (
    create_dataset
)

from configs.config import (
    MODELS_DIR,
    LEARNING_RATE
)

# ==========================================================
# Create Dataset
# ==========================================================

pairs, labels = create_pairs()

dataset = create_dataset(

    pairs,

    labels

)

print("\nDataset Ready!")

print(dataset)

# ==========================================================
# Encoder
# ==========================================================

encoder = load_finetuned_encoder()

print("\nEncoder Loaded!")

# ==========================================================
# Siamese Model
# ==========================================================

input1 = Input(
    shape=(224,224,3)
)

input2 = Input(
    shape=(224,224,3)
)

embedding1 = encoder(
    input1
)

embedding2 = encoder(
    input2
)

distance = Lambda(

    lambda tensors: tf.norm(

        tensors[0]-tensors[1],

        axis=1,

        keepdims=True

    )

)(

    [

        embedding1,

        embedding2

    ]

)

model = Model(

    inputs=[

        input1,

        input2

    ],

    outputs=distance

)

print(model.summary())

# ==========================================================
# Contrastive Loss
# ==========================================================

def contrastive_loss(

    y_true,

    y_pred,

    margin=1.0

):

    y_true = tf.cast(

        y_true,

        tf.float32

    )

    positive = y_true * tf.square(

        y_pred

    )

    negative = (

        1-y_true

    ) * tf.square(

        tf.maximum(

            margin-y_pred,

            0

        )

    )

    return tf.reduce_mean(

        positive+negative

    )

# ==========================================================
# Compile
# ==========================================================

model.compile(

    optimizer=Adam(

        learning_rate=LEARNING_RATE

    ),

    loss=contrastive_loss

)

print("\nModel Compiled!")


# ==========================================================
# Callbacks
# ==========================================================

MODELS_DIR.mkdir(

    parents=True,

    exist_ok=True

)

callbacks=[

    EarlyStopping(

        monitor="loss",

        patience=3,

        restore_best_weights=True

    ),

    ReduceLROnPlateau(

        monitor="loss",

        factor=0.2,

        patience=2,

        verbose=1

    ),

    ModelCheckpoint(

        filepath=MODELS_DIR/

        "siamese_model.keras",

        monitor="loss",

        save_best_only=True,

        verbose=1

    )

]

# ==========================================================
# Train
# ==========================================================

history=model.fit(

    dataset,

    epochs=10,

    callbacks=callbacks

)

print("\nTraining Finished!")

print(

    "\nSaved Model :",

    MODELS_DIR/

    "siamese_model.keras"

)