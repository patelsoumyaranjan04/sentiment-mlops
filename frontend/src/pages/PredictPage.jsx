// frontend/src/pages/PredictPage.jsx

import { useState } from "react";
import SinglePredict from "../components/SinglePredict";
import BatchPredict from "../components/BatchPredict";
import HistoryPanel from "../components/HistoryPanel";

export default function PredictPage() {
  const [mode, setMode] = useState("single");
  const [history, setHistory] = useState([]);

  const addToHistory = (entry) => {
    setHistory((prev) => [entry, ...prev].slice(0, 20));
  };

  return (
    <div className="predict-page">
      <div className="mode-toggle">
        <button
          className={`mode-btn ${mode === "single" ? "active" : ""}`}
          onClick={() => setMode("single")}
        >
          Single Review
        </button>
        <button
          className={`mode-btn ${mode === "batch" ? "active" : ""}`}
          onClick={() => setMode("batch")}
        >
          Batch Analysis
        </button>
      </div>

      <div className="predict-layout">
        <div className="predict-main">
          {mode === "single" ? (
            <SinglePredict onResult={addToHistory} />
          ) : (
            <BatchPredict onResults={(entries) => entries.forEach(addToHistory)} />
          )}
        </div>
        <div className="predict-sidebar">
          <HistoryPanel history={history} onClear={() => setHistory([])} />
        </div>
      </div>
    </div>
  );
}
