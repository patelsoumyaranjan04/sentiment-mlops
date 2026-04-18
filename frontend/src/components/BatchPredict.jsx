// frontend/src/components/BatchPredict.jsx

import { useState } from "react";
import { predictBatch } from "../api/client";
import ResultCard from "./ResultCard";

export default function BatchPredict({ onResults }) {
  const [rawText, setRawText]   = useState("");
  const [results, setResults]   = useState([]);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);

  const reviews = rawText
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .slice(0, 50);

  const handleSubmit = async () => {
    if (!reviews.length) return;
    setLoading(true);
    setError(null);
    setResults([]);
    try {
      const data = await predictBatch(reviews);
      setResults(data.predictions);
      onResults?.(data.predictions.map((r) => ({ ...r, timestamp: new Date().toISOString() })));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const positiveCount = results.filter((r) => r.label === 1).length;
  const negativeCount = results.length - positiveCount;

  return (
    <div className="card">
      <h2 className="card-title">Batch Analysis</h2>
      <p className="card-hint">Enter one review per line (max 50 reviews).</p>

      <textarea
        className="review-input"
        rows={8}
        placeholder={"Great product, very happy!\nTerrible quality, broke immediately.\nDecent for the price."}
        value={rawText}
        onChange={(e) => setRawText(e.target.value)}
        disabled={loading}
      />
      <div className="char-count">{reviews.length} review{reviews.length !== 1 ? "s" : ""} detected</div>

      <button
        className="submit-btn"
        onClick={handleSubmit}
        disabled={loading || reviews.length === 0}
      >
        {loading ? <><span className="spinner" /> Analyzing...</> : `✨ Analyze ${reviews.length} Review${reviews.length !== 1 ? "s" : ""}`}
      </button>

      {error && <div className="error-box">⚠️ {error}</div>}

      {results.length > 0 && (
        <>
          <div className="batch-summary">
            <div className="summary-chip positive-chip">😊 {positiveCount} Positive</div>
            <div className="summary-chip negative-chip">😞 {negativeCount} Negative</div>
            <div className="summary-chip neutral-chip">
              📊 {Math.round((positiveCount / results.length) * 100)}% positive rate
            </div>
          </div>
          <div className="batch-results">
            {results.map((r, i) => (
              <div key={i} className="batch-item">
                <div className="batch-item-text">
                  <span className="batch-index">#{i + 1}</span>
                  <span className="batch-review-text">{r.review.slice(0, 80)}{r.review.length > 80 ? "…" : ""}</span>
                </div>
                <ResultCard result={r} />
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
