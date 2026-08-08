import os

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import faiss

from PIL import Image


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/siamese_encoder.keras"
FAISS_PATH = "faiss/fashion.index"
METADATA_PATH = "metadata/products.csv"
IMAGE_DIR = "catalog_images"

IMAGE_SIZE = (224, 224)
TOP_K_DEFAULT = 5


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FashionVision AI",
    page_icon="👗",
    layout="wide"
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# ============================================================
# LOAD FAISS
# ============================================================

@st.cache_resource
def load_faiss():

    return faiss.read_index(
        FAISS_PATH
    )


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_data
def load_metadata():

    return pd.read_csv(
        METADATA_PATH
    )


# Load everything once
model = load_model()
index = load_faiss()
products = load_metadata()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    image = np.array(
        image
    ).astype("float32")

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# EMBEDDING GENERATION
# ============================================================

def generate_embedding(image):

    processed = preprocess_image(
        image
    )

    embedding = model.predict(
        processed,
        verbose=0
    )

    return embedding.astype(
        "float32"
    )


# ============================================================
# SEARCH
# ============================================================

def search_products(
    embedding,
    top_k
):

    scores, indices = index.search(
        embedding,
        top_k
    )

    results = products.iloc[
        indices[0]
    ].copy()

    results["similarity"] = scores[0]

    return results


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">👗 FashionVision AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Image-based fashion recommendation powered by
    Siamese deep learning and FAISS similarity search.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Search Settings")

    top_k = st.slider(
        "Recommendations",
        min_value=3,
        max_value=10,
        value=TOP_K_DEFAULT
    )

    st.divider()

    st.markdown(
        """
        ### How it works

        **1. Upload an image**

        Upload any fashion product image.

        **2. Extract features**

        The trained Siamese encoder converts
        the image into a 256-dimensional embedding.

        **3. Search**

        FAISS searches the precomputed product
        embedding index.

        **4. Recommend**

        The most visually similar products
        are displayed.
        """
    )

    st.divider()

    st.caption(
        f"Catalog: {len(products):,} products"
    )

    st.caption(
        "Embedding: 256 dimensions"
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📸 Upload a fashion product image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    query_image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.divider()

    # --------------------------------------------------------
    # QUERY IMAGE
    # --------------------------------------------------------

    st.subheader("Your Image")

    col1, col2 = st.columns(
        [1, 2]
    )

    with col1:

        st.image(
            query_image,
            width=300
        )

    # --------------------------------------------------------
    # GENERATE RECOMMENDATIONS
    # --------------------------------------------------------

    with st.spinner(
        "Finding visually similar products..."
    ):

        embedding = generate_embedding(
            query_image
        )

        results = search_products(
            embedding,
            top_k
        )

    with col2:

        st.success(
            f"Found {len(results)} similar products"
        )

        st.write(
            f"Embedding dimension: "
            f"{embedding.shape[1]}"
        )

        st.write(
            f"FAISS catalog size: "
            f"{index.ntotal:,}"
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "✨ Recommended Products"
    )

    columns = st.columns(
        min(top_k, 5)
    )

    for i, (_, row) in enumerate(
        results.iterrows()
    ):

        image_path = os.path.join(
            IMAGE_DIR,
            str(row["image_path"])
        )

        with columns[
            i % len(columns)
        ]:

            # Product image
            if os.path.exists(
                image_path
            ):

                st.image(
                    image_path,
                    use_container_width=True
                )

            # Category
            st.markdown(
                f"**{row['subCategory']}**"
            )

            # Product name
            product_name = str(
                row["productDisplayName"]
            )

            st.caption(
                product_name[:80]
            )

            # Similarity
            st.metric(
                "Similarity",
                f"{float(row['similarity']):.2%}"
            )


else:

    st.info(
        "👆 Upload a fashion product image "
        "to get visually similar recommendations."
    )

    st.markdown(
        """
        ### Supported categories

        👜 Bags  
        👖 Bottomwear  
        👗 Dress  
        👡 Sandal  
        👟 Shoes  
        👕 Topwear
        """
    )