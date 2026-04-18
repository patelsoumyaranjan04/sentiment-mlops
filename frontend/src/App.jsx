// frontend/src/App.jsx
// Main application component

import { useState } from "react";
import PredictPage from "./pages/PredictPage";
import PipelinePage from "./pages/PipelinePage";
import "./App.css";

export default function App() {
  const [page, setPage] = useState("predict");

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">🧠</span>
            <span className="logo-text">SentiAI</span>
            <span className="logo-sub">Amazon Review Sentiment Analyzer</span>
          </div>
          <nav className="nav">
            <button
              className={`nav-btn ${page === "predict" ? "active" : ""}`}
              onClick={() => setPage("predict")}
            >
              🔍 Analyze
            </button>
            <button
              className={`nav-btn ${page === "pipeline" ? "active" : ""}`}
              onClick={() => setPage("pipeline")}
            >
              ⚙️ ML Pipeline
            </button>
          </nav>
        </div>
      </header>

      <main className="main">
        {page === "predict" ? <PredictPage /> : <PipelinePage />}
      </main>

      <footer className="footer">
        <span>DA6401 — Amazon Sentiment MLOps | BiLSTM Model</span>
      </footer>
    </div>
  );
}
