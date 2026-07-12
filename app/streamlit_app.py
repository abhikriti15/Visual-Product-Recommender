"""
=========================================================
Visual Product Recommendation System
=========================================================

Streamlit UI

Author : Abhikriti Saxena
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tempfile

from PIL import Image

from sklearn.metrics.pairwise import cosine_similarity

from configs.config import (
    EMBEDDINGS_DIR,
)

from models.resnet50_feature_extractor import (
    ResNet50FeatureExtractor
)

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(

    page_title="Visual Product Recommendation",

    page_icon="🛍️",

    layout="wide"

)

# ==========================================================
# Title
# ==========================================================

st.title("🛍️ Visual Product Recommendation System")

st.markdown(
"""
Upload a product image and retrieve visually similar products
using the Fine-Tuned ResNet50 model.
"""
)

# ==========================================================
# Load Resources
# ==========================================================

@st.cache_resource
def load_resources():

    extractor = ResNet50FeatureExtractor()

    embeddings = np.load(
        EMBEDDINGS_DIR / "finetuned_embeddings.npy"
    )

    metadata = pd.read_csv(
        EMBEDDINGS_DIR / "finetuned_metadata.csv"
    )

    with open(
        EMBEDDINGS_DIR / "finetuned_image_paths.pkl",
        "rb"
    ) as f:

        image_paths = pickle.load(f)

    return (
        extractor,
        embeddings,
        metadata,
        image_paths
    )


with st.spinner("Loading Fine-Tuned Model..."):

    extractor, embeddings, metadata, image_paths = load_resources()

st.success("✅ Model Loaded Successfully")


# ==========================================================
# Dataset Information
# ==========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Total Products",
        len(metadata)
    )

with col2:

    st.metric(
        "Embedding Size",
        embeddings.shape[1]
    )

with col3:

    st.metric(
        "Model",
        "ResNet50"
    )
# ==========================================================
# Upload Image
# ==========================================================

st.markdown("---")

st.header("📤 Upload Product Image")

uploaded_file = st.file_uploader(

    "Choose an image",

    type=["jpg", "jpeg", "png"]

)

if uploaded_file is not None:

    uploaded_image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 2])

    with col1:

        st.image(

            uploaded_image,

            caption="Uploaded Image",

            width="stretch"

        )

    with col2:

        st.success("✅ Image Uploaded Successfully")

        st.write("Filename :", uploaded_file.name)

        st.write("Size :", uploaded_image.size)

        st.write("Mode :", uploaded_image.mode)


# ==========================================================
# Save Uploaded Image Temporarily
# ==========================================================

with tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".jpg"
) as temp_file:

    uploaded_image.save(temp_file.name)

    temp_path = temp_file.name

# ==========================================================
# Generate Embedding
# ==========================================================

with st.spinner("Extracting Features..."):

    query_embedding = extractor.extract(
        temp_path
    )

st.success("✅ Feature Embedding Generated")

st.write("Embedding Shape :", query_embedding.shape)

# ==========================================================
# Similarity Search
# ==========================================================

with st.spinner("Finding Similar Products..."):

    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    )[0]

    top_indices = similarities.argsort()[::-1][:5]

st.success("✅ Top 5 Similar Products Found")

# ==========================================================
# Display Recommendations
# ==========================================================

st.markdown("---")

st.header("🎯 Top 5 Recommendations")

cols = st.columns(5)

for col, idx in zip(cols, top_indices):

    with col:

        st.image(
            image_paths[idx],
            width="stretch"
        )

        st.caption(
            metadata.iloc[idx]["subset_category"]
        )

        st.write(
            metadata.iloc[idx]["productDisplayName"]
        )

        st.progress(float(similarities[idx]))

        st.write(
            f"Similarity : {similarities[idx]:.4f}"
        )