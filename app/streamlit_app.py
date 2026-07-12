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
# Custom CSS
# ==========================================================

st.markdown("""
<style>

.main {
    background-color: #f6f8fb;
}

.hero {
    background: linear-gradient(90deg,#4F46E5,#7C3AED);
    padding:30px;
    border-radius:18px;
    color:white;
    text-align:center;
    margin-bottom:25px;
    box-shadow:0px 8px 20px rgba(0,0,0,0.15);
}

.hero h1{
    font-size:42px;
    margin-bottom:10px;
}

.hero p{
    font-size:18px;
    color:#f1f5f9;
}

.metric-card{
    background:#1e293b;
    color:white;
    padding:25px;
    border-radius:20px;
    text-align:center;
    border:1px solid #374151;
    transition:0.3s;
}
.metric-card:hover{
    transform:translateY(-6px);
}

.upload-box{
    background:white;
    border-radius:15px;
    padding:20px;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# Title
# ==========================================================

st.markdown("""
<div class="hero">

<h1>Visual Product Recommendation System</h1>

<p>
Deep Learning • Transfer Learning • Siamese Network
</p>

</div>
""", unsafe_allow_html=True)

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


st.markdown("<br>", unsafe_allow_html=True)

col1,col2,col3 = st.columns(3)

with col1:

    st.markdown("""
    <div class="metric-card">
    <h3>📦 Products</h3>
    <h2>{}</h2>
    </div>
    """.format(len(metadata)), unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="metric-card">
    <h3>🧠 Embedding</h3>
    <h2>2048</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
    <div class="metric-card">
    <h3>🤖 Model</h3>
    <h2>ResNet50</h2>
    </div>
    """, unsafe_allow_html=True)


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

        score = float(similarities[idx])

        st.badge(f"{score*100:.2f}% Match")

st.markdown("---")
st.header("📊 Model Comparison")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Baseline ResNet50",
        "77.01%",
        "Precision@5"
    )

with col2:
    st.metric(
        "Fine-Tuned ResNet50",
        "83.81%",
        "+6.79%"
    )

with col3:
    st.metric(
        "Siamese Network",
        "Loss = 0.010",
        "Training Completed"
    )

st.markdown("---")

st.markdown(
"""
<center>

Developed by **Abhikriti Saxena**

Transfer Learning • ResNet50 • Siamese Network • Streamlit

</center>
""",
unsafe_allow_html=True
)