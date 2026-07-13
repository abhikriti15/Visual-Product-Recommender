"""
=========================================================
Visual Product Recommendation System
=========================================================

Author : Abhikriti Saxena
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import EMBEDDINGS_DIR
from models.resnet50_feature_extractor import ResNet50FeatureExtractor

import base64
import pickle
import tempfile
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from configs.config import EMBEDDINGS_DIR
from models.resnet50_feature_extractor import ResNet50FeatureExtractor


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Visual Product Recommendation System",
    page_icon="🛍️",
    layout="wide"
)

# Fixed thumbnail size used everywhere a product image is rendered.
# Keeping this consistent (and pre-resizing with LANCZOS) is what
# removes the blur you get from letting the browser stretch small
# source images up to an arbitrary column width.
THUMB_SIZE = (400, 400)

# ==========================================================
# CSS
# ==========================================================

st.markdown("""
<style>

header{visibility:hidden;}
footer{visibility:hidden;}

.block-container{
    padding-top:1rem;
    padding-bottom:2rem;
}

/* ---------------- HERO ---------------- */

.hero{
background:linear-gradient(135deg,#4F46E5,#7C3AED);
padding:42px;
border-radius:24px;
text-align:center;
color:white;
margin-bottom:28px;
box-shadow:0 15px 35px rgba(0,0,0,.22);
}

.hero h1{
font-size:48px;
font-weight:700;
margin-bottom:12px;
}

.hero p{
font-size:20px;
color:#E5E7EB;
}

/* ---------------- DASHBOARD ---------------- */

.metric-card{
background:#1E293B;
padding:28px;
border-radius:20px;
border:1px solid #334155;
text-align:center;
transition:.25s;
height:300px;
display:flex;
flex-direction:column;
justify-content:center;
align-items:center;
}

.metric-card:hover{
transform:translateY(-6px);
box-shadow:0 15px 30px rgba(0,0,0,.25);
}

.metric-title{
font-size:24px;
font-weight:600;
color:white;
margin-top:10px;
}

.metric-value{
font-size:58px;
font-weight:700;
color:white;
margin-top:25px;
}

.metric-sub{
font-size:17px;
color:#CBD5E1;
}

/* ---------------- UPLOAD ---------------- */

.upload-card{
background:#1E293B;
padding:24px;
border-radius:18px;
border:1px solid #334155;
}

.upload-title{
font-size:28px;
font-weight:700;
margin-bottom:12px;
}

.details-card{
background:#1E3A5F;
padding:24px;
border-radius:18px;
border:1px solid #2563EB;
height:100%;
}

.details-card h3{
color:white;
margin-bottom:20px;
}

.details-label{
font-weight:600;
color:#60A5FA;
margin-top:12px;
}

.details-value{
color:white;
font-size:16px;
margin-bottom:10px;
}

/* ---------------- PRODUCT CARD ---------------- */

/* Fixed-height card so every one of the 5 recommendation
   columns lines up regardless of title length or image
   aspect ratio. Everything below is sized in fixed px so
   nothing reflows differently per-card. */
.product-card{
background:#1F2937;
border-radius:18px;
padding:14px;
border:1px solid #374151;
transition:.25s;
height:100%;
display:flex;
flex-direction:column;
}

.product-card:hover{
border:1px solid #3B82F6;
box-shadow:0 10px 20px rgba(59,130,246,.25);
}

.product-thumb-wrap{
width:100%;
height:220px;
border-radius:14px;
overflow:hidden;
background:#0F172A;
}

.product-thumb-wrap img{
width:100%;
height:100%;
object-fit:cover;
display:block;
image-rendering:auto;
}

.category{
display:inline-block;
background:#2563EB;
color:white;
padding:4px 12px;
border-radius:20px;
font-size:12px;
font-weight:600;
width:fit-content;
margin-top:12px;
}

.product-title{
color:white;
font-size:16px;
font-weight:600;
line-height:1.4;
margin-top:10px;
height:44px;
display:-webkit-box;
-webkit-line-clamp:2;
-webkit-box-orient:vertical;
overflow:hidden;
}

.score{
font-size:22px;
font-weight:700;
color:#FBBF24;
text-align:center;
margin-top:10px;
}

.footer{
text-align:center;
color:#CBD5E1;
padding:25px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# HERO
# ==========================================================

st.markdown("""
<div class="hero">

<h1>🛍️ Visual Product Recommendation System</h1>

<p>
Deep Learning • Transfer Learning • Fine-Tuned ResNet50 • Siamese Network
</p>

</div>
""", unsafe_allow_html=True)


# ==========================================================
# HELPERS
# ==========================================================

def image_to_base64(img: Image.Image, size: tuple = THUMB_SIZE) -> str:
    """
    Resize an image with high-quality LANCZOS resampling and return
    it as a base64 JPEG string. Pre-resizing this way (instead of
    letting the browser stretch a small source image with
    use_container_width) is what removes the blur/pixelation.
    """
    img = img.convert("RGB")

    # Center-crop to a square first so object-fit:cover in CSS doesn't
    # have to guess how to crop mismatched aspect ratios badly.
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))

    img = img.resize(size, Image.LANCZOS)

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"


@st.cache_data(show_spinner=False)
def load_and_encode_thumb(path: str) -> str:
    """Cached so we don't re-resize the same dataset image every rerun."""
    img = Image.open(path)
    return image_to_base64(img)


# ==========================================================
# LOAD MODEL
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

    return extractor, embeddings, metadata, image_paths


with st.spinner("Loading Recommendation Engine..."):
    extractor, embeddings, metadata, image_paths = load_resources()

st.success("✅ Recommendation Engine Ready")


# ==========================================================
# DASHBOARD
# ==========================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
<div class="metric-card">
<div style="font-size:48px;">📦</div>
<div class="metric-title">Total Products</div>
<div class="metric-value">{len(metadata)}</div>
<div class="metric-sub">Products in Database</div>
</div>
""", unsafe_allow_html=True)

with c2:
    st.markdown("""
<div class="metric-card">
<div style="font-size:48px;">🧠</div>
<div class="metric-title">Embedding Size</div>
<div class="metric-value">2048</div>
<div class="metric-sub">Dimensions</div>
</div>
""", unsafe_allow_html=True)

with c3:
    st.markdown("""
<div class="metric-card">
<div style="font-size:48px;">🤖</div>
<div class="metric-title">AI Model</div>
<div class="metric-value" style="font-size:34px;">Fine-Tuned</div>
<div class="metric-sub">ResNet50</div>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# IMAGE UPLOAD
# ==========================================================

st.markdown("---")
st.header("📤 Upload Product Image")
st.caption(
    "Upload a fashion product image to retrieve the Top-5 visually similar products."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# Guard: everything below needs an uploaded image, so stop cleanly
# instead of crashing on an undefined temp_path (the original bug).
if not uploaded_file:
    st.info("👆 Upload an image above to get recommendations.")
    st.stop()

uploaded_image = Image.open(uploaded_file).convert("RGB")

left, right = st.columns([1, 2])

with left:
    st.markdown(
        f"""
<div class="product-thumb-wrap" style="height:320px;">
<img src="{image_to_base64(uploaded_image, size=(600, 600))}">
</div>
""",
        unsafe_allow_html=True
    )
    st.caption("Uploaded Image")

with right:
    st.markdown(f"""
<div class="details-card">
<h3>Image Details</h3>
<div class="details-label">File Name</div>
<div class="details-value">{uploaded_file.name}</div>
<div class="details-label">Resolution</div>
<div class="details-value">{uploaded_image.size[0]} × {uploaded_image.size[1]}</div>
<div class="details-label">Color Mode</div>
<div class="details-value">{uploaded_image.mode}</div>
<div class="details-label">Status</div>
<div class="details-value">Ready for Recommendation ✅</div>
</div>
""", unsafe_allow_html=True)

with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
    uploaded_image.save(temp.name)
    temp_path = temp.name


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

st.markdown("---")

with st.spinner("🧠 Extracting Deep Visual Features..."):
    query_embedding = extractor.extract(temp_path)

st.success("✅ Feature Extraction Completed")

st.info(
    "The uploaded image has been converted into a deep feature embedding "
    "using the Fine-Tuned ResNet50 model."
)

# ==========================================================
# SIMILARITY SEARCH
# ==========================================================

with st.spinner("🔍 Finding Similar Products..."):
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    )[0]

    top_indices = similarities.argsort()[::-1][:5]

st.success("✅ Top 5 Similar Products Retrieved")

# ==========================================================
# TOP RECOMMENDATIONS
# ==========================================================

st.markdown("---")
st.subheader("🎯 Top 5 Recommendations")

cols = st.columns(5)

for col, idx in zip(cols, top_indices):

    score = similarities[idx] * 100
    category = metadata.iloc[idx]["subset_category"]
    title = metadata.iloc[idx]["productDisplayName"]

    thumb_b64 = load_and_encode_thumb(image_paths[idx])

    with col:
        with st.container(border=True):
            st.markdown(
                f"""
<div class="product-card">

<div class="product-thumb-wrap">
<img src="{thumb_b64}">
</div>

<span class="category">{category}</span>

<div class="product-title">{title}</div>

</div>
""",
                unsafe_allow_html=True
            )

            st.progress(float(similarities[idx]))

            st.markdown(
                f"""<div class="score">⭐ {score:.2f}% Match</div>""",
                unsafe_allow_html=True
            )

# ==========================================================
# MODEL COMPARISON
# ==========================================================

st.markdown("---")
st.subheader("📊 Model Comparison")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Baseline ResNet50", "77.01%", "Precision@5")

with c2:
    st.metric("Fine-Tuned ResNet50", "83.81%", "+6.80%")

with c3:
    st.metric("Siamese Network", "Loss = 0.010", "Training Complete")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div style="text-align:center;color:#CBD5E1;">
<h4>🛍️ Visual Product Recommendation System</h4>
Developed by <b>Abhikriti Saxena</b>
<br>
Transfer Learning • Fine-Tuned ResNet50 • Siamese Network • Streamlit
</div>
""",
unsafe_allow_html=True
)