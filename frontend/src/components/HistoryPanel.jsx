// frontend/src/components/HistoryPanel.jsx

export default function HistoryPanel({ history, onClear }) {
  if (history.length === 0) {
    return (
      <div className="card history-card">
        <h3 className="card-title">Recent Predictions</h3>
        <p className="history-empty">No predictions yet. Analyze a review to get started.</p>
      </div>
    );
  }

  const positive = history.filter((h) => h.label === 1).length;
  const negative = history.length - positive;

  return (
    <div className="card history-card">
      <div className="history-header">
        <h3 className="card-title">Recent Predictions</h3>
        <button className="clear-btn" onClick={onClear}>Clear</button>
      </div>

      <div className="history-stats">
        <span className="hstat positive">😊 {positive}</span>
        <span className="hstat negative">😞 {negative}</span>
      </div>

      <ul className="history-list">
        {history.map((h, i) => (
          <li key={i} className={`history-item ${h.label === 1 ? "pos" : "neg"}`}>
            <span className="h-emoji">{h.label === 1 ? "😊" : "😞"}</span>
            <div className="h-content">
              <div className="h-text">{h.review.slice(0, 60)}{h.review.length > 60 ? "…" : ""}</div>
              <div className="h-meta">
                {Math.round(h.confidence * 100)}% · {new Date(h.timestamp).toLocaleTimeString()}
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
