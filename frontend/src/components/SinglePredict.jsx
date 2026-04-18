// frontend/src/components/SinglePredict.jsx

import { useState } from "react";
import { predictSingle } from "../api/client";
import ResultCard from "./ResultCard";

const EXAMPLES = [
  "This product is absolutely amazing! Fast delivery and great quality.",
  "Terrible waste of money. Broke after one day. Do not buy.",
  "Decent product for the price. Nothing special but gets the job done.",
  "The best purchase I've made all year. Highly recommend to everyone!",
  "Very disappointed. The item looks nothing like the pictures shown.",
];

export default function SinglePredict({ onResult }) {
  const [text, setText]       = useState("");
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const handleSubmit = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictSingle(text);
      setResult(data);
      onResult?.({ ...data, timestamp: new Date().toISOString() });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleExample = (ex) => {
    setText(ex);
    setResult(null);
    setError(null);
  };

  const wordCount = text.trim().split(/\s+/).filter(Boolean).length;

  return (
    <div className="card">
      <h2 className="card-title">Analyze a Review</h2>

      <div className="examples-row">
        <span className="examples-label">Try an example:</span>
        {EXAMPLES.map((ex, i) => (
          <button key={i} className="example-chip" onClick={() => handleExample(ex)}>
            Example {i + 1}
          </button>
        ))}
      </div>

      <div className="textarea-wrapper">
        <textarea
          className="review-input"
          rows={5}
          placeholder="Paste or type an Amazon product review here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={loading}
          maxLength={5000}
        />
        <div className="char-count">{wordCount} words · {text.length}/5000 chars</div>
      </div>

      <button
        className="submit-btn"
        onClick={handleSubmit}
        disabled={loading || !text.trim()}
      >
        {loading ? (
          <><span className="spinner" /> Analyzing...</>
        ) : (
          "✨ Analyze Sentiment"
        )}
      </button>

      {error && (
        <div className="error-box">
          ⚠️ {error}
        </div>
      )}

      {result && <ResultCard result={result} />}
    </div>
  );
}
