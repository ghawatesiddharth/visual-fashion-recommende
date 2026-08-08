import tensorflow as tf
import numpy as np

MODEL_PATH = "models/siamese_encoder.keras"

print("Loading Siamese encoder...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("✅ Model loaded successfully")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

# Test with a dummy image
dummy = np.random.rand(
    1, 224, 224, 3
).astype("float32")

embedding = model.predict(
    dummy,
    verbose=0
)

print("Embedding shape:", embedding.shape)
print("Embedding norm:", np.linalg.norm(embedding))