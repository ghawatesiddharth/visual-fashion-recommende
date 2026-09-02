import { useState } from "react";
import {
  Upload,
  Sparkles,
  Image as ImageIcon,
  Search,
  RefreshCw,
} from "lucide-react";

import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState([]);
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");

  const handleFile = (selectedFile) => {
    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please upload a valid image.");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResults([]);
    setError("");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);

    const droppedFile = e.dataTransfer.files[0];
    handleFile(droppedFile);
  };

  const getRecommendations = async () => {
    if (!file) {
      setError("Please upload a fashion image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResults([]);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("top_k", String(topK));

      const response = await fetch(
        `${API_URL}/recommend?top_k=${topK}`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error("Recommendation request failed.");
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || "Unable to generate recommendations.");
      }

      setResults(data.recommendations || []);
    } catch (err) {
      console.error(err);
      setError(
        "Could not connect to the recommendation engine. Make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const resetSearch = () => {
    setFile(null);
    setPreview(null);
    setResults([]);
    setError("");
  };

  return (
    <div className="app">

      {/* NAVBAR */}
      <nav className="navbar">
        <div className="brand">
          <div className="brand-icon">
            <Sparkles size={20} />
          </div>

          <div>
            <h2>FashionVision</h2>
            <span>AI STYLE ENGINE</span>
          </div>
        </div>

        <div className="nav-status">
          <span className="status-dot"></span>
          AI Engine Online
        </div>
      </nav>


      {/* HERO */}
      <main>

        <section className="hero">

          <div className="hero-content">

            <div className="eyebrow">
              <Sparkles size={16} />
              VISUAL FASHION SEARCH
            </div>

            <h1>
              Find the look.
              <br />
              <span>Instantly.</span>
            </h1>

            <p>
              Upload a fashion image and let our Siamese AI model
              discover visually similar products from our catalog.
            </p>

          </div>


          {/* UPLOAD CARD */}
          <div
            className={`upload-card ${dragging ? "dragging" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >

            {!preview ? (

              <div className="upload-empty">

                <div className="upload-icon">
                  <Upload size={30} />
                </div>

                <h3>Drop your fashion image here</h3>

                <p>
                  or choose an image from your device
                </p>

                <label className="browse-btn">
                  <ImageIcon size={18} />
                  Browse Image

                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/webp"
                    onChange={(e) =>
                      handleFile(e.target.files[0])
                    }
                    hidden
                  />
                </label>

                <small>
                  JPG, PNG or WEBP • Max 200MB
                </small>

              </div>

            ) : (

              <div className="preview-area">

                <img
                  src={preview}
                  alt="Uploaded fashion"
                  className="preview-image"
                />

                <button
                  className="remove-btn"
                  onClick={resetSearch}
                >
                  <RefreshCw size={16} />
                  Change image
                </button>

              </div>

            )}

          </div>


          {/* SEARCH BUTTON */}

          {file && (
  <div className="search-controls">

    <div className="recommendation-control">
      <label htmlFor="topK">
        SHOW RESULTS
      </label>

      <select
        id="topK"
        value={topK}
        onChange={(e) => setTopK(Number(e.target.value))}
      >
        <option value={5}>5 recommendations</option>
        <option value={10}>10 recommendations</option>
        <option value={15}>15 recommendations</option>
      </select>
    </div>

    <button
      className="search-btn"
      onClick={getRecommendations}
      disabled={loading}
    >
      {loading ? (
        <>
          <div className="spinner"></div>
          Finding similar styles...
        </>
      ) : (
        <>
          <Search size={20} />
          Find Similar Styles
        </>
      )}
    </button>

  </div>
)}

          {error && (
            <div className="error">
              {error}
            </div>
          )}

        </section>


        {/* RESULTS */}

        {results.length > 0 && (

          <section className="results-section">

            <div className="results-heading">

              <div>
                <span className="section-label">
                  AI MATCHES
                </span>

                <h2>
                  Similar styles
                </h2>
              </div>

              <div className="result-count">
                {results.length} matches
              </div>

            </div>


            <div className="results-grid">

              {results.map((item, index) => (

                <div
                  className="product-card"
                  key={`${item.id}-${index}`}
                >

                  <div className="product-image-container">

                    <img
                      src={`${API_URL}${item.image}`}
                      alt={item.product}
                      className="product-image"
                    />

                    <div className="match-score">
                      {(item.similarity * 100).toFixed(1)}%
                    </div>

                  </div>


                  <div className="product-info">

                    <span className="category">
                      {item.category}
                    </span>

                    <h3>
                      {item.product}
                    </h3>

                    <div className="product-meta">

                      <span>
                        {item.colour}
                      </span>

                      <span>
                        {item.gender}
                      </span>

                    </div>

                  </div>

                </div>

              ))}

            </div>

          </section>

        )}

      </main>


      {/* FOOTER */}

      <footer>
        <span>FashionVision AI</span>
        <span>Powered by Siamese Neural Networks + FAISS</span>
      </footer>

    </div>
  );
}

export default App;