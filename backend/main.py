from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "siamese_encoder.keras"
FAISS_PATH = BASE_DIR / "faiss" / "fashion.index"
METADATA_PATH = BASE_DIR / "metadata" / "products.csv"
CATALOG_DIR = BASE_DIR / "catalog_images"


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="FashionVision AI API",
    description="Image-based fashion recommendation API",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SERVE CATALOG IMAGES
# ============================================================

app.mount(
    "/catalog",
    StaticFiles(directory=str(CATALOG_DIR)),
    name="catalog",
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Siamese encoder...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print("Model loaded successfully.")
print("Model input:", model.input_shape)
print("Model output:", model.output_shape)


# ============================================================
# LOAD FAISS
# ============================================================

print("Loading FAISS index...")

index = faiss.read_index(str(FAISS_PATH))

print("FAISS vectors:", index.ntotal)


# ============================================================
# LOAD METADATA
# ============================================================

print("Loading product metadata...")

products = pd.read_csv(METADATA_PATH)

print("Products:", len(products))


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert uploaded image into the format expected
    by the Siamese encoder.
    """

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(image, dtype=np.float32)

    # EfficientNet preprocessing used during training.
    # EfficientNet in Keras expects pixel values in [0, 255].
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ============================================================
# EMBEDDING GENERATION
# ============================================================

def generate_embedding(image: Image.Image) -> np.ndarray:

    processed = preprocess_image(image)

    embedding = model.predict(
        processed,
        verbose=0,
    )

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    # Ensure 2D shape
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    # L2 normalization
    faiss.normalize_L2(embedding)

    return embedding


# ============================================================
# PRODUCT RESULT CREATION
# ============================================================

def create_product_result(
    metadata_index: int,
    similarity: float,
):

    row = products.iloc[metadata_index]

    product_id = int(row["id"])

    image_filename = f"{product_id}.jpg"

    return {
        "id": product_id,
        "product": str(row["productDisplayName"]),
        "category": str(row["subCategory"]),
        "master_category": str(row["masterCategory"]),
        "gender": str(row["gender"]),
        "colour": str(row["baseColour"]),
        "season": str(row["season"]),
        "usage": str(row["usage"]),
        "similarity": round(float(similarity), 4),
        "image": f"/catalog/{image_filename}",
    }


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "application": "FashionVision AI",
        "status": "running",
        "model": "Siamese Encoder",
        "embedding_dimension": 256,
        "catalog_size": int(index.ntotal),
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "faiss_vectors": int(index.ntotal),
        "products": int(len(products)),
    }


# ============================================================
# RECOMMENDATION ENDPOINT
# ============================================================

@app.post("/recommend")
async def recommend(
    file: UploadFile = File(...),
    top_k: int = 5,
):

    # --------------------------------------------------------
    # Validate number of recommendations
    # --------------------------------------------------------

    top_k = max(1, min(top_k, 20))

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    image_bytes = await file.read()

    try:

        image = Image.open(
            __import__("io").BytesIO(image_bytes)
        )

    except Exception:

        return {
            "success": False,
            "error": "Invalid image file.",
        }

    # --------------------------------------------------------
    # Generate embedding
    # --------------------------------------------------------

    embedding = generate_embedding(image)

    # --------------------------------------------------------
    # Search FAISS
    # --------------------------------------------------------

    similarities, indices = index.search(
        embedding,
        top_k,
    )

    # --------------------------------------------------------
    # Build results
    # --------------------------------------------------------

    recommendations = []

    for similarity, metadata_index in zip(
        similarities[0],
        indices[0],
    ):

        if metadata_index < 0:
            continue

        if metadata_index >= len(products):
            continue

        result = create_product_result(
            int(metadata_index),
            float(similarity),
        )

        recommendations.append(result)

    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return {
        "success": True,
        "filename": file.filename,
        "embedding_dimension": int(embedding.shape[1]),
        "recommendations": recommendations,
    }