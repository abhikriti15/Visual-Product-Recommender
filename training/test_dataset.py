from pathlib import Path

from dataset.dataset import create_dataset

from configs.config import SUBSET_DIR

label_mapping = {
    "Tshirts":0,
    "Shirts":1,
    "Casual Shoes":2,
    "Sports Shoes":3,
    "Heels":4,
    "Watches":5,
    "Handbags":6,
    "Kurtas":7,
    "Tops":8,
    "Sandals":9,
}

train_dataset = create_dataset(
    SUBSET_DIR/"train.csv",
    label_mapping
)

for images, labels in train_dataset.take(1):

    print(images.shape)

    print(labels.shape)

    print(labels)

