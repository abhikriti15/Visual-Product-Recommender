# 🛍️ Visual Product Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![ResNet50](https://img.shields.io/badge/Model-ResNet50-success?style=for-the-badge)
![Siamese](https://img.shields.io/badge/Siamese-Network-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

### Deep Learning based Fashion Product Recommendation using Fine-Tuned ResNet50 & Siamese Network

</div>

---

## 📌 Overview

This project is a **Visual Product Recommendation System** that recommends visually similar fashion products from a dataset using **Deep Learning**.

Instead of relying on product names or metadata, the recommendation is performed purely based on **image similarity**.

The system extracts deep visual embeddings using a **Fine-Tuned ResNet50** model and retrieves the most similar products using **Cosine Similarity**.

---

## ✨ Features

- 📷 Upload any fashion product image
- 🧠 Feature extraction using Fine-Tuned ResNet50
- 🔍 Image similarity search using Cosine Similarity
- 🎯 Top-5 visually similar product recommendations
- ⚡ Fast embedding-based retrieval
- 🌐 Interactive Streamlit Web Application
- 📊 Deep Learning powered recommendation engine

---

# 🏗️ Model Pipeline

```text
Input Image
      │
      ▼
Image Preprocessing
      │
      ▼
Fine-Tuned ResNet50
      │
      ▼
2048-D Feature Embedding
      │
      ▼
Cosine Similarity Search
      │
      ▼
Top-5 Similar Products
```

---

# 🖥️ Demo

### Home Page

> Add screenshot here

```
assets/home.png
```

### Recommendation Result

> Add screenshot here

```
assets/result.png
```

---

# 📂 Project Structure

```text
Visual-Product-Recommender/
│
├── app/
│   └── streamlit_app.py
│
├── configs/
│   └── config.py
│
├── data/
│   ├── embeddings/
│   ├── raw/
│   ├── subset/
│   └── processed/
│
├── models/
│   ├── resnet50_feature_extractor.py
│   └── siamese_network.py
│
├── training/
│   ├── train_siamese.py
│   └── generate_embeddings.py
│
├── requirements.txt
│
└── README.md
```

---

# 🧠 Deep Learning Architecture

### Backbone

- Fine-Tuned ResNet50

### Feature Vector

- 2048-D Embeddings

### Similarity Metric

- Cosine Similarity

### Recommendation

- Top-5 Nearest Products

---

# 📊 Model Performance

| Model | Precision@5 |
|--------|------------:|
| Baseline ResNet50 | 77.01% |
| Fine-Tuned ResNet50 | **83.81%** |

### Siamese Network Training Loss

```
0.010
```

---

# 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- ResNet50
- Siamese Network
- NumPy
- Pandas
- Scikit-Learn
- Pillow
- Streamlit

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/abhikriti15/Visual-Product-Recommender.git
```

Go to project

```bash
cd Visual-Product-Recommender
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app/streamlit_app.py
```

---

# 📸 Dataset

Fashion Product Images

Dataset contains multiple fashion categories including:

- 👕 Shirts
- 👚 Tops
- 👖 Tshirts
- 👠 Heels
- 👟 Casual Shoes
- 👟 Sports Shoes
- 👜 Handbags
- 👗 Kurtas
- ⌚ Watches
- 👡 Sandals

---

# 💡 Future Improvements

- FAISS-based Approximate Nearest Neighbor Search
- CLIP Embeddings
- Multi-modal Recommendations
- Category Filtering
- Price-aware Recommendation
- Mobile Deployment
- User Feedback Loop

---

# 👩‍💻 Author

## Abhikriti Saxena

Computer Science Engineering Student

AI • Deep Learning • Computer Vision • Machine Learning

GitHub

https://github.com/abhikriti15

LinkedIn

(Add your LinkedIn URL)

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

---

<div align="center">

Made with ❤️ using Deep Learning & Streamlit

</div>