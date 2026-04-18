// frontend/src/pages/PipelinePage.jsx
// Shows ML pipeline stages + live API health status

import { useState, useEffect } from "react";
import { checkHealth, checkReady } from "../api/client";

const PIPELINE_STAGES = [
  {
    id: 1,
    name: "Data Ingestion",
    icon: "📥",
    tool: "Apache Airflow",
    description: "Reads Amazon_review.csv, validates schema, drops nulls and duplicates.",
    outputs: ["Amazon_review_validated.csv"],
    status: "done",
  },
  {
    id: 2,
    name: "Preprocessing",
    icon: "🔧",
    tool: "Python / NLTK",
    description: "Cleans text, lemmatizes, removes stopwords. Splits into train/val/test. Fits Tokenizer and pads sequences.",
    outputs: ["X_train.npy", "X_val.npy", "X_test.npy", "tokenizer.pkl", "baseline_stats.json"],
    status: "done",
  },
  {
    id: 3,
    name: "Experiment Tracking",
    icon: "📊",
    tool: "MLflow",
    description: "Logs hyperparameters, per-epoch metrics, classification report, confusion matrix, and model artifacts.",
    outputs: ["mlruns/", "registered model"],
    status: "done",
  },
  {
    id: 4,
    name: "Model Training",
    icon: "🏋️",
    tool: "TensorFlow / Kaggle GPU",
    description: "Trains Bidirectional LSTM (Embedding → BiLSTM → Dense → Sigmoid). Early stopping on val_loss.",
    outputs: ["bilstm_model/", "best_model.h5"],
    status: "done",
  },
  {
    id: 5,
    name: "Model Serving",
    icon: "🚀",
    tool: "FastAPI + MLflow",
    description: "REST API exposes /predict and /predict/batch. Model loaded from SavedModel or MLflow registry.",
    outputs: ["/predict", "/predict/batch", "/health", "/ready"],
    status: "live",
  },
  {
    id: 6,
    name: "Monitoring",
    icon: "📡",
    tool: "Prometheus + Grafana",
    description: "Tracks prediction counts, latency, positive ratio, input length distribution. Alerts on drift.",
    outputs: ["Grafana dashboard", "Prometheus metrics"],
    status: "live",
  },
  {
    id: 7,
    name: "Containerization",
    icon: "🐳",
    tool: "Docker Compose",
    description: "Frontend, backend, MLflow, Prometheus, Grafana each run as isolated containers.",
    outputs: ["docker-compose.yml"],
    status: "done",
  },
];

const STATUS_CONFIG = {
  done:    { label: "Complete",    cls: "status-done"    },
  live:    { label: "Live",        cls: "status-live"    },
  pending: { label: "Pending",     cls: "status-pending" },
};

export default function PipelinePage() {
  const [health, setHealth]   = useState(null);
  const [ready, setReady]     = useState(null);
  const [checking, setChecking] = useState(false);

  const checkStatus = async () => {
    setChecking(true);
    try {
      const h = await checkHealth();
      setHealth(h.status === "ok");
    } catch {
      setHealth(false);
    }
    try {
      const r = await checkReady();
      setReady(r.status === "ready");
    } catch {
      setReady(false);
    }
    setChecking(false);
  };

  useEffect(() => { checkStatus(); }, []);

  return (
    <div className="pipeline-page">
      <div className="pipeline-header">
        <h2>ML Pipeline Overview</h2>
        <p>End-to-end MLOps pipeline from raw data to live serving.</p>
      </div>

      {/* API Status Panel */}
      <div className="card status-panel">
        <div className="status-panel-header">
          <h3>🔌 API Status</h3>
          <button className="refresh-btn" onClick={checkStatus} disabled={checking}>
            {checking ? "Checking…" : "↻ Refresh"}
          </button>
        </div>
        <div className="status-indicators">
          <div className={`status-indicator ${health === true ? "ok" : health === false ? "fail" : "unknown"}`}>
            <span className="dot" />
            <span>API Liveness</span>
            <span className="status-val">{health === null ? "—" : health ? "UP" : "DOWN"}</span>
          </div>
          <div className={`status-indicator ${ready === true ? "ok" : ready === false ? "fail" : "unknown"}`}>
            <span className="dot" />
            <span>Model Ready</span>
            <span className="status-val">{ready === null ? "—" : ready ? "YES" : "NO"}</span>
          </div>
          <div className="status-indicator ok">
            <span className="dot" />
            <span>MLflow UI</span>
            <a href="http://localhost:5000" target="_blank" rel="noreferrer" className="status-link">
              localhost:5000 ↗
            </a>
          </div>
          <div className="status-indicator ok">
            <span className="dot" />
            <span>Grafana</span>
            <a href="http://localhost:3003" target="_blank" rel="noreferrer" className="status-link">
              localhost:3003 ↗
            </a>
          </div>
          <div className="status-indicator ok">
            <span className="dot" />
            <span>Airflow</span>
            <a href="http://localhost:8080" target="_blank" rel="noreferrer" className="status-link">
              localhost:8080 ↗
            </a>
          </div>
        </div>
      </div>

      {/* Pipeline Stages */}
      <div className="pipeline-stages">
        {PIPELINE_STAGES.map((stage, i) => (
          <div key={stage.id} className="pipeline-stage-row">
            <div className="stage-card card">
              <div className="stage-top">
                <span className="stage-icon">{stage.icon}</span>
                <div className="stage-info">
                  <div className="stage-name">
                    <span className="stage-num">Stage {stage.id}</span>
                    {stage.name}
                  </div>
                  <span className="stage-tool">{stage.tool}</span>
                </div>
                <span className={`stage-status ${STATUS_CONFIG[stage.status].cls}`}>
                  {STATUS_CONFIG[stage.status].label}
                </span>
              </div>
              <p className="stage-desc">{stage.description}</p>
              <div className="stage-outputs">
                <span className="outputs-label">Outputs:</span>
                {stage.outputs.map((o) => (
                  <span key={o} className="output-chip">{o}</span>
                ))}
              </div>
            </div>
            {i < PIPELINE_STAGES.length - 1 && (
              <div className="stage-arrow">↓</div>
            )}
          </div>
        ))}
      </div>

      {/* Tech Stack */}
      <div className="card tech-stack-card">
        <h3>🧱 Full Technology Stack</h3>
        <div className="tech-grid">
          {[
            ["Model",               "TensorFlow / Keras BiLSTM"],
            ["Data Pipeline",       "Apache Airflow"],
            ["Data Versioning",     "DVC + Git LFS"],
            ["Experiment Tracking", "MLflow"],
            ["Model Serving",       "FastAPI + Uvicorn"],
            ["Monitoring",          "Prometheus + Grafana"],
            ["Containerization",    "Docker + Docker Compose"],
            ["Source Control",      "Git + GitHub"],
            ["Testing",             "pytest"],
          ].map(([k, v]) => (
            <div key={k} className="tech-row">
              <span className="tech-key">{k}</span>
              <span className="tech-val">{v}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
