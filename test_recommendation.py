import os
import numpy as np
import pandas as pd
import tensorflow as tf
import faiss

from PIL import Image


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/siamese_encoder.keras"
FAISS_PATH = "faiss/fashion.index"
METADATA_PATH = "metadata/products.csv"
IMAGE_DIR = "catalog_images"

IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD COMPONENTS
# ============================================================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("✅ Model loaded")


print("Loading FAISS index...")

index = faiss.read_index(
    FAISS_PATH
)

print("✅ FAISS loaded")
print("FAISS vectors:", index.ntotal)


print("Loading metadata...")

products = pd.read_csv(
    METADATA_PATH
)

print("✅ Metadata loaded")
print("Products:", len(products))


# ============================================================
# VERIFY IMAGE COUNT
# ============================================================

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".jpg")
]

print("Catalog images:", len(image_files))


# ============================================================
# SELECT A REAL CATALOG IMAGE
# ============================================================

query_filename = image_files[0]

query_path = os.path.join(
    IMAGE_DIR,
    query_filename
)

print("\nQuery image:")
print(query_path)


# ============================================================
# PREPROCESS
# ============================================================

image = Image.open(
    query_path
).convert("RGB")

image = image.resize(
    IMAGE_SIZE
)

image_array = np.array(
    image
).astype("float32")

image_array = np.expand_dims(
    image_array,
    axis=0
)


# ============================================================
# GENERATE EMBEDDING
# ============================================================

print("\nGenerating embedding...")

embedding = model.predict(
    image_array,
    verbose=0
)

embedding = embedding.astype(
    "float32"
)

print("Embedding shape:", embedding.shape)

print(
    "Embedding norm:",
    np.linalg.norm(embedding)
)


# ============================================================
# FAISS SEARCH
# ============================================================

TOP_K = 5

print("\nSearching FAISS...")

scores, indices = index.search(
    embedding,
    TOP_K
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n==============================")
print("TOP RECOMMENDATIONS")
print("==============================")

for rank, (idx, score) in enumerate(
    zip(indices[0], scores[0]),
    start=1
):

    row = products.iloc[idx]

    print(
        f"\n#{rank}"
    )

    print(
        "Product ID:",
        row["id"]
    )

    print(
        "Category:",
        row["subCategory"]
    )

    print(
        "Product:",
        row["productDisplayName"]
    )

    print(
        "Similarity:",
        round(float(score), 4)
    )

    print(
        "Image:",
        row["image_path"]
    )


print("\n✅ Recommendation pipeline works!")