# FashionVision AI
### AI-Powered Visual Fashion Recommendation System

FashionVision AI is a deep-learning-based visual fashion recommendation system that retrieves visually similar fashion products from a catalog using an uploaded image. It converts images into numerical embeddings and uses FAISS to find nearby catalog products.

> **Project type:** Working academic prototype  
> **Core workflow:** Upload an image → extract visual features → search similar products → display recommendations

## Table of Contents

- [Features](#features)
- [Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Training Results](#training-results)
- [Project Structure](#project-structure)
- [Runtime Artifacts](#runtime-artifacts)
- [API](#backend-api)
- [Example Results](#example-recommendation)
- [Installation](#installation-and-setup)
- [Run the Application](#run-the-application)
- [Testing](#testing)
- [Limitations and Future Work](#limitations)
- [Author](#author)
- [License and Dataset Notes](#license-and-dataset-notes)

## Features

- Upload a fashion image through a React web interface.
- Generate visual embeddings with a trained Siamese encoder based on EfficientNetB0.
- Search similar products using FAISS.
- Request the top **5, 10, or 15** recommendations.
- Display product images and available metadata, including category, colour, gender, and similarity score.
- Serve model inference through FastAPI.
- Provide a responsive interface for end-to-end image-based recommendations.

## System Architecture

```text
User
 |
 v
React Frontend
 | Image Upload
 v
FastAPI Backend
 |
 v
Image Preprocessing
 |
 v
Siamese Encoder
 |
 v
256-D L2-Normalized Embedding
 |
 v
FAISS Similarity Search
 |
 v
Product Metadata
 |
 v
Top-K Recommendations
 |
 v
React Recommendation Cards
```

**Components**
1. **React frontend:** Image selection, preview, recommendation count, loading state, and result cards.
2. **FastAPI backend:** Receives the image, preprocesses it, runs inference, searches FAISS, and returns JSON.
3. **Siamese encoder:** Converts an image into a 256-dimensional embedding.
4. **FAISS index:** Retrieves nearby catalog embeddings.
5. **Product metadata:** Supplies product IDs, names, categories, and available attributes.

## Technology Stack

| Area | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript, CSS, Lucide React |
| Backend | Python, FastAPI, Uvicorn, python-multipart |
| Machine learning | TensorFlow, Keras, EfficientNetB0-based encoder, Siamese network, contrastive loss |
| Similarity search | FAISS |
| Data and images | NumPy, Pandas, Pillow |

## Dataset

The project uses the **Fashion Product Images Dataset by Param Aggarwal**, obtained from Kaggle. The original metadata contains approximately **44,424 product records**. A balanced subset of **1,800 products** was selected.

| Category | Products |
|---|---:|
| Bags | 300 |
| Bottomwear | 300 |
| Dress | 300 |
| Sandal | 300 |
| Shoes | 300 |
| Topwear | 300 |
| **Total** | **1,800** |

### Data split

The dataset was split by category using a stratified approach.

| Split | Proportion | Approximate count |
|---|---:|---:|
| Training | 70% | 1,260 |
| Validation | 15% | 270 |
| Testing | 15% | 270 |
| **Total** | **100%** | **1,800** |

Counts are approximate because per-category rounding may affect exact totals.

## Machine Learning Pipeline

### 1. Image preprocessing
The query image is converted to RGB, resized to **224 × 224** pixels, converted into a numerical tensor, and scaled using the preprocessing expected by the model.

### 2. Feature extraction
An EfficientNetB0-based encoder extracts visual features. The retrieval pipeline uses a **256-dimensional embedding**.

### 3. Siamese network
A Siamese architecture learns an embedding space for visual similarity. During training, image pairs and a contrastive-loss objective encourage similar examples to have closer representations and dissimilar examples to be farther apart.

### 4. Embedding normalization
Embeddings are **L2-normalized** before similarity search.

### 5. Catalog embedding generation
Catalog images are passed through the trained encoder. The resulting embedding array was saved as `siamese_embeddings.npy` and has shape **(1800, 256)**.

### 6. FAISS indexing
FAISS indexes the catalog vectors. At inference time, the uploaded image is encoded, nearby vectors are retrieved, and their corresponding product metadata is returned.

```text
Uploaded Image → Preprocessing → Siamese Encoder
       → 256-D Embedding → FAISS Search
       → Top-K Matches → Metadata → UI Cards
```

## Training Results

The final Siamese encoder was trained for three epochs.

| Epoch | Training loss |
|---:|---:|
| 1 | 0.2364 |
| 2 | 0.1332 |
| 3 | 0.1105 |

The loss decreased over the recorded epochs, showing improvement on the training objective. **Training loss is not recommendation accuracy.** A standardized retrieval metric such as Recall@K or Precision@K has not yet been established, so this README does not claim a retrieval-accuracy percentage.

**Saved model:** `models/siamese_encoder.keras`

## Project Structure

```text
visual-fashion-recommender/
├── backend/
│   └── main.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   ├── main.jsx
│   │   └── assets/
│   ├── package.json
│   ├── package-lock.json
│   ├── index.html
│   └── vite.config.js
├── models/
│   └── siamese_encoder.keras
├── faiss/
│   └── fashion.index
├── metadata/
│   └── products.csv
├── catalog_images/       # Local product images; may be excluded from Git
├── test_model.py
├── test_recommendation.py
├── requirements.txt
├── .gitignore
└── README.md
```

Training notebooks and intermediate artifacts may also exist in the development workspace but are not required for normal inference.

## Runtime Artifacts

The backend expects these files at the configured paths:

```text
models/siamese_encoder.keras
faiss/fashion.index
metadata/products.csv
catalog_images/
```

Catalog image filenames must match product IDs, for example `catalog_images/10018.jpg`.

> **Fresh-clone note:** If `catalog_images/` is excluded from Git due to repository size, obtain the prepared images separately and place them in that directory. Without them, product-image URLs will not show the expected catalog pictures.

The normal inference path needs the trained encoder, FAISS index, metadata CSV, and catalog images. The saved embedding array and intermediate training models are not required at runtime when the index has already been built.

## Backend API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Application status and basic model information |
| `GET` | `/health` | Backend health check |
| `POST` | `/recommend?top_k=5` | Accepts an uploaded image and returns similar products |
| `GET` | `/catalog/{product_id}.jpg` | Serves a catalog product image |

The frontend supports `top_k` values of **5, 10, and 15**.

Example endpoint:

```text
http://127.0.0.1:8000/recommend?top_k=5
```

An illustrative result item may look like this; actual values come from the project metadata:

```json
{
  "id": 10018,
  "product": "Example product name",
  "category": "Bottomwear",
  "master_category": "Apparel",
  "gender": "Men",
  "colour": "Black",
  "season": "Summer",
  "usage": "Sports",
  "similarity": 1.0,
  "image": "/catalog/10018.jpg"
}
```

When the backend is running:
- API root: http://127.0.0.1:8000
- Swagger documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## Example Recommendation

A sample test using catalog product ID `10018` returned:

| Rank | Product ID | Category | Similarity score |
|---:|---:|---|---:|
| 1 | 10018 | Bottomwear | 1.0000 |
| 2 | 49956 | Bottomwear | 0.9869 |
| 3 | 44919 | Bottomwear | 0.9844 |
| 4 | 32457 | Bottomwear | 0.9829 |
| 5 | 57097 | Bottomwear | 0.9819 |

These are scores from a sample retrieval run, not percentages of model accuracy. A query using a catalog image may return that same image first.

## Installation and Setup

### Prerequisites
- Python compatible with the dependencies in `requirements.txt`
- Node.js and npm
- Git
- The model, FAISS index, metadata CSV, and catalog images

The project was developed in Windows with Python 3.12.10. TensorFlow and FAISS compatibility can vary across operating systems and Python versions.

### 1. Clone the repository

Replace the placeholder with your actual repository URL.

```powershell
git clone https://github.com/ghawatesiddharth/visual-fashion-recommende.git
cd visual-fashion-recommender
```

### 2. Create and activate a virtual environment

Run from the project root in Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install backend dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If a required package is missing, install it in the active environment and update `requirements.txt` so the setup remains reproducible.

### 4. Restore runtime artifacts
Confirm the model, index, metadata, and `catalog_images/` directory exist at the paths listed above. If images are not included in Git, obtain them separately from the appropriate dataset source.

### 5. Install frontend dependencies

Open a second terminal:

```powershell
cd frontend
npm install
```

## Run the Application

### Start the backend

From the **project root**, with the Python environment activated:

```powershell
python -m uvicorn backend.main:app --reload
```

Backend URL: `http://127.0.0.1:8000`

### Start the frontend

In a separate terminal:

```powershell
cd frontend
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173/`. Keep both terminals running while using the application.

## Application Workflow

1. Start FastAPI.
2. Start the React frontend.
3. Open the frontend in a browser.
4. Upload or drag and drop a fashion image.
5. Preview the selected image.
6. Choose 5, 10, or 15 recommendations.
7. Click **Find Similar Styles**.
8. React sends the image to FastAPI.
9. FastAPI preprocesses the image and runs the encoder.
10. The encoder generates a normalized 256-dimensional embedding.
11. FAISS retrieves nearby catalog embeddings.
12. Product metadata is attached to the results.
13. React displays recommendation cards and similarity scores.

## Testing

Run from the project root with the Python environment activated:

```powershell
python test_model.py
python test_recommendation.py
```

These scripts are intended to check model loading and the recommendation pipeline; the precise checks depend on their current implementation.

Manual integration checks:
- Verify the API root, health endpoint, and Swagger page.
- Upload a valid image and check the response.
- Try each supported `top_k` value.
- Confirm product IDs match the metadata.
- Confirm catalog image URLs load.
- Inspect the browser console and backend terminal if an error occurs.

## Current Capabilities

- Image-based visual fashion search.
- Similar-product recommendations.
- Top-K results: 5, 10, or 15.
- Six selected fashion categories and a 1,800-product catalog.
- 256-dimensional L2-normalized embeddings.
- Siamese encoder and FAISS nearest-neighbor search.
- FastAPI backend and React frontend.
- End-to-end image-to-recommendation workflow.

## Limitations

- The system focuses on visual similarity rather than complete semantic fashion understanding.
- Quality depends on the catalog, image diversity, and training data.
- The current catalog is a selected subset of 1,800 products.
- Live prices, inventory, and external shopping links are not integrated.
- Similarity does not guarantee an exact product match or a suitable replacement.
- Standardized retrieval evaluation using Recall@K or Precision@K has not yet been reported.
- Catalog images must be available locally for product pictures to display.

## Future Improvements

- Expand the catalog and category coverage.
- Evaluate retrieval with Recall@K, Precision@K, and human relevance checks.
- Improve training data and positive/negative pair selection.
- Experiment with ranking and re-ranking strategies.
- Add clothing-attribute detection and outfit recommendations.
- Add user preferences and recommendation history.
- Integrate prices, availability, and shopping links where permitted.
- Add authentication and user profiles.
- Deploy the application to cloud infrastructure.
- Improve logging, monitoring, error handling, and automated tests.

## Design Decisions

### Why a Siamese network?
A conventional classifier predicts a label such as shoes or dresses. A similarity-learning model learns image representations that can be compared with one another. This suits a retrieval workflow where the user may not know the product name.

### Why embeddings?
Embeddings represent images as compact numerical vectors, allowing the system to compare visual representations rather than raw image files.

### Why FAISS?
FAISS provides optimized nearest-neighbor search tools for retrieving candidate products from a vector index more efficiently as a catalog grows.

### Why FastAPI and React?
FastAPI exposes the inference and retrieval pipeline through HTTP endpoints. React provides the interactive image-upload and recommendation interface. Separating them supports independent development and maintenance.

## Project Status

**Status: Working academic prototype.**

Implemented components include dataset preparation, model training, a Siamese encoder, catalog embedding generation, FAISS indexing, metadata integration, a FastAPI recommendation API, a React frontend, and an end-to-end recommendation workflow. Formal retrieval benchmarking and production deployment remain potential future work.

## Author

**Siddharth Ghawate**  
B.Tech Student  
Machine Learning / Data Science Project

## License and Dataset Notes

This repository is intended for academic and educational purposes. Review the original dataset's license and terms on its source page before redistributing the dataset, product images, or derived assets. The code and dataset may be subject to different licensing terms.

---

## GitHub Repository Checklist

Before pushing, verify that the repository includes the essential files:

- `backend/main.py`
- `frontend/src/App.jsx`
- `frontend/src/App.css`
- `frontend/src/index.css`
- `frontend/src/main.jsx`
- `frontend/package.json`
- `frontend/package-lock.json`
- `frontend/index.html`
- `frontend/vite.config.js`
- `models/siamese_encoder.keras`
- `faiss/fashion.index`
- `metadata/products.csv`
- `requirements.txt`
- `test_model.py`
- `test_recommendation.py`
- `.gitignore`
- `README.md`

Do not commit environment folders or generated build/cache files such as `venv/`, `.venv/`, `frontend/node_modules/`, `frontend/dist/`, `frontend/.vite/`, or `__pycache__/`.

If `catalog_images/` is excluded from Git, document how to obtain and restore those images. Do not assume a fresh clone contains them.

### Final Git commands

```powershell
git status --short
git add .
git status
git commit -m "Build FashionVision AI visual recommendation system"
git branch -M main
git remote -v
git push -u origin main
```

If a remote is already configured, do not add it again. If there is no remote, configure it before pushing:

```powershell
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
```
