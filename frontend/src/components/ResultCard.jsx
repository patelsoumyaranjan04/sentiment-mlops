// frontend/src/components/ResultCard.jsx

export default function ResultCard({ result }) {
  const isPositive   = result.label === 1;
  const pct          = Math.round(result.confidence * 100);
  const barPositive  = isPositive ? pct : 100 - pct;
  const barNegative  = 100 - barPositive;

  return (
    <div className={`result-card ${isPositive ? "positive" : "negative"}`}>
      <div className="result-header">
        <span className="result-emoji">{isPositive ? "😊" : "😞"}</span>
        <span className="result-label">{isPositive ? "POSITIVE" : "NEGATIVE"}</span>
        <span className="result-confidence">{pct}% confidence</span>
      </div>

      <div className="result-bar-wrapper">
        <div className="bar-label">Negative</div>
        <div className="result-bar">
          <div className="bar-neg" style={{ width: `${barNegative}%` }} />
          <div className="bar-pos" style={{ width: `${barPositive}%` }} />
        </div>
        <div className="bar-label">Positive</div>
      </div>

      <div className="result-meta">
        Inference latency: {result.latency_ms} ms
      </div>
    </div>
  );
}
