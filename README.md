FashionVision AI
AI-Powered Visual Fashion Recommendation System
README.md — Complete GitHub Documentation
Copy the Markdown content below into a file named README.md in your project root.
FashionVision AI
AI-Powered Visual Fashion Recommendation System
FashionVision AI is a deep-learning-based visual fashion recommendation system that finds visually similar fashion products from a catalog using an uploaded image.
Instead of relying only on text-based product search, the system uses computer vision to understand the visual appearance of a fashion item and retrieve similar products from a fashion catalog.
Features
•	Upload a fashion image through a modern web interface.
•	Generate visual embeddings using a trained Siamese neural network.
•	Search similar products using FAISS.
•	Return the top 5, 10, or 15 recommendations.
•	Display product category, color, gender, similarity score, and product information.
•	FastAPI backend for ML inference.
•	React frontend for an interactive dashboard.
•	Responsive fashion-tech UI.
System Architecture
    User
      |
      v
    React Frontend
      |
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
    Fashion Product Metadata
      |
      v
    Top-K Visual Recommendations
Technology Stack
Frontend
•	React
•	Vite
•	JavaScript
•	CSS
•	Lucide React
Backend
•	Python
•	FastAPI
•	Uvicorn
•	Pillow
•	NumPy
•	Pandas
•	Python Multipart
Machine Learning
•	TensorFlow
•	Keras
•	EfficientNetB0
•	Siamese Network
•	Contrastive Loss
•	Image Embeddings
•	L2 Normalization
Similarity Search
•	FAISS
Dataset
The project uses the Fashion Product Images Dataset by Param Aggarwal from Kaggle.
The original dataset contains approximately 44,000 fashion products. For this project, a balanced subset of 1,800 products was selected.
Selected Categories
| Category | Samples |
|---|---:|
| Bags | 300 |
| Bottomwear | 300 |
| Dress | 300 |
| Sandal | 300 |
| Shoes | 300 |
| Topwear | 300 |
| **Total** | **1,800** |
Data Split
•	Training: 70%
•	Validation: 15%
•	Testing: 15%
•	Stratified split by category
Machine Learning Pipeline
1. Image Preprocessing
Uploaded images are converted to RGB, resized to 224 x 224 pixels, converted to NumPy arrays, and prepared for model inference.
2. Feature Extraction
An EfficientNetB0-based encoder converts each fashion image into a compact 256-dimensional visual representation.
The embeddings are L2-normalized before similarity search.
3. Siamese Network
A Siamese architecture was trained to learn a visual similarity space. Similar fashion images are encouraged to have closer embeddings.
Contrastive loss was used during training.
4. Training
The final Siamese encoder was trained for 3 epochs.
| Epoch | Training Loss |
|---:|---:|
| 1 | 0.2364 |
| 2 | 0.1332 |
| 3 | 0.1105 |
Final model:
    models/siamese_encoder.keras
5. FAISS Similarity Search
Catalog images are converted into 256-dimensional embeddings and indexed with FAISS.
The recommendation flow is:
    Uploaded Image
          |
          v
    Siamese Encoder
          |
          v
    256-D Embedding
          |
          v
    FAISS Search
          |
          v
    Top-K Similar Products
FAISS index:
    faiss/fashion.index
Project Structure
    visual-fashion-recommender/
    |
    +-- backend/
    |   +-- main.py
    |
    +-- frontend/
    |   +-- src/
    |   |   +-- App.jsx
    |   |   +-- App.css
    |   |   +-- main.jsx
    |   +-- package.json
    |   +-- package-lock.json
    |   +-- index.html
    |   +-- vite.config.js
    |
    +-- models/
    |   +-- siamese_encoder.keras
    |
    +-- faiss/
    |   +-- fashion.index
    |
    +-- metadata/
    |   +-- products.csv
    |
    +-- test_model.py
    +-- test_recommendation.py
    +-- requirements.txt
    +-- .gitignore
    +-- README.md
Backend API
GET /
Returns application status and basic model information.
GET /health
Checks backend health.
POST /recommend
Accepts an uploaded image and returns visually similar fashion products.
Example:
    POST http://127.0.0.1:8000/recommend?top_k=5
Supported frontend values:
    5
    10
    15
GET /catalog/{product_id}.jpg
Serves a catalog product image.
Example Recommendation
A sample test using product ID 10018 returned visually similar Bottomwear products.
| Rank | Product ID | Category | Similarity |
|---:|---:|---|---:|
| 1 | 10018 | Bottomwear | 1.0000 |
| 2 | 49956 | Bottomwear | 0.9869 |
| 3 | 44919 | Bottomwear | 0.9844 |
| 4 | 32457 | Bottomwear | 0.9829 |
| 5 | 57097 | Bottomwear | 0.9819 |
Installation
1. Clone the repository
    git clone <YOUR_GITHUB_REPOSITORY_URL>
    cd visual-fashion-recommender
2. Create Python environment
Windows PowerShell:
    python -m venv venv
    .\venv\Scripts\Activate.ps1
3. Install backend dependencies
    python -m pip install -r requirements.txt
Run Backend
From the project root:
    python -m uvicorn backend.main:app --reload
Backend:
    http://127.0.0.1:8000
Swagger API documentation:
    http://127.0.0.1:8000/docs
Run Frontend
Open another terminal:
    cd frontend
    npm install
    npm run dev
Frontend:
    http://localhost:5173/
Complete Application Workflow
1. Start the FastAPI backend.
2. Start the React frontend.
3. Open the frontend in a browser.
4. Upload a fashion image.
5. Select the number of recommendations.
6. Click "Find Similar Styles".
7. React sends the image to FastAPI.
8. FastAPI preprocesses the image.
9. The Siamese encoder creates a 256-D embedding.
10. FAISS searches the nearest catalog embeddings.
11. Product metadata is retrieved.
12. Recommendations are displayed in the dashboard.
Model Artifacts
Required runtime artifacts:
    models/siamese_encoder.keras
    faiss/fashion.index
    metadata/products.csv
Intermediate training models and unused embeddings are not required for normal application inference.
Catalog Images
The application expects catalog images in:
    catalog_images/
with filenames matching product IDs, for example:
    catalog_images/10018.jpg
    catalog_images/49956.jpg
    catalog_images/44919.jpg
The complete catalog contains 1,800 images. If these images are excluded from GitHub because of repository-size considerations, they must be supplied separately for the recommendation cards to display product images.
Testing
Model loading:
    python test_model.py
Recommendation pipeline:
    python test_recommendation.py
These scripts verify that the trained model and recommendation pipeline work correctly.
Current Capabilities
•	Visual fashion search
•	Similar-product recommendation
•	Top-K recommendations
•	Six fashion categories
•	1,800-product catalog
•	256-dimensional embeddings
•	FAISS vector search
•	Siamese neural network
•	FastAPI backend
•	React frontend
•	End-to-end image recommendation workflow
Limitations
•	The system performs visual similarity search rather than complete semantic fashion understanding.
•	Results depend on the quality and variety of the catalog.
•	The current model uses the selected 1,800-product subset.
•	Live prices, stock availability, and external shopping links are not currently integrated.
•	Recommendations should be interpreted as visually similar products, not guaranteed exact product matches.
Future Improvements
1. Expand the fashion catalog.
2. Add real-time product search.
3. Integrate visual shopping/search services.
4. Add price and availability information.
5. Add external shopping links.
6. Add personalized user preferences.
7. Add recommendation history.
8. Add clothing attribute detection.
9. Add outfit recommendation.
10. Improve ranking and re-ranking.
11. Deploy the application to the cloud.
12. Add authentication and user profiles.
Why Siamese Networks?
A traditional classifier answers:
    What category does this image belong to?
A Siamese network is better suited to:
    How visually similar are these two images?
Fashion recommendation requires finding visually similar products, so learning an embedding space is useful.
Why FAISS?
Comparing an uploaded image with every catalog image becomes inefficient as the catalog grows.
FAISS provides efficient vector similarity search.
The system therefore uses:
    Image
      |
      v
    Neural Network
      |
      v
    Embedding
      |
      v
    FAISS
      |
      v
    Nearest Neighbors
Project Status
Working academic prototype.
Implemented and tested:
•	Dataset preparation
•	Model training
•	Siamese encoder
•	Embedding generation
•	FAISS indexing
•	Metadata integration
•	FastAPI recommendation API
•	React frontend
•	End-to-end recommendation workflow
Author
Siddharth Ghawate
B-Tech Student
Machine Learning / Data Science Project
License
This project is intended for academic and educational purposes.
Dataset rights and usage are subject to the original dataset license and terms provided by its original source.
GitHub — Essential Files Checklist
Keep these essential files in the repository:
•	backend/main.py
•	frontend/src/App.jsx
•	frontend/src/App.css
•	frontend/src/main.jsx
•	frontend/package.json
•	frontend/package-lock.json
•	frontend/index.html
•	frontend/vite.config.js
•	models/siamese_encoder.keras
•	faiss/fashion.index
•	metadata/products.csv
•	test_model.py
•	test_recommendation.py
•	requirements.txt
•	.gitignore
•	README.md
Do not upload:
•	venv/
•	node_modules/
•	temp_* folders
•	*.zip files
•	frontend/dist/
•	frontend/.vite/
•	unused intermediate models/embeddings
Final Git Commands
git status --short
git add .
git status
git commit -m "Build FashionVision AI visual recommendation system"
git branch -M main
git remote -v
git push -u origin main
