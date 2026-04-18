# High-Level Design (HLD)
## Amazon Reviews Sentiment Analysis — MLOps Pipeline
**Version:** 1.0.0 | **Course:** DA6401

---

## 1. Problem Statement

Binary sentiment classification of Amazon product reviews (positive / negative) using a Bidirectional LSTM deep learning model, served through a production-grade MLOps pipeline.

**ML Metric:** Test Accuracy ≥ 90%, ROC-AUC ≥ 0.90
**Business Metric:** Inference latency < 200ms (p95)

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER LAYER                                │
│                  React Frontend (port 3000)                      │
│         Single Predict │ Batch Predict │ Pipeline Dashboard      │
└────────────────────────┬─────────────────────────────────────────┘
                         │  REST (JSON)
┌────────────────────────▼─────────────────────────────────────────┐
│                     API LAYER                                    │
│              FastAPI Backend (port 8000)                         │
│     /predict  │  /predict/batch  │  /health  │  /ready          │
│               Prometheus metrics at /metrics                     │
└────────────────────────┬─────────────────────────────────────────┘
                         │
         ┌───────────────┼────────────────┐
         │               │                │
┌────────▼──────┐ ┌──────▼──────┐ ┌──────▼──────────┐
│  ML Model     │ │  MLflow     │ │  Monitoring     │
│  BiLSTM       │ │  Registry   │ │  Prometheus     │
│  SavedModel   │ │  (port 5000)│ │  + Grafana      │
└───────────────┘ └─────────────┘ │  (ports 9090,   │
                                  │   3001)         │
                                  └─────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                   │
│    Apache Airflow DAG (port 8080)                                │
│    ingest → preprocess → dvc_add → dvc_push                     │
│    DVC tracks: raw CSV, processed arrays, tokenizer             │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Descriptions

### 3.1 Data Engineering Pipeline (Airflow + DVC)
- **Trigger:** Daily schedule or manual
- **Stage 1 — Ingest:** Reads `Amazon_review.csv`, validates schema (required columns: `review`, `sentiment`; valid sentiment values: {0, 1}), drops nulls and duplicates, saves validated snapshot
- **Stage 2 — Preprocess:** Cleans text (contraction expansion, stopword removal, lemmatization), splits into 70/10/20 train/val/test, fits Keras Tokenizer on training set, pads sequences to `max_length=64`, saves numpy arrays and tokenizer pickle, computes baseline statistics for drift detection
- **DVC:** Tracks all processed artifacts via `dvc.yaml` pipeline; `dvc repro` reproduces the entire pipeline deterministically

### 3.2 Model Training (Kaggle GPU + MLflow)
- **Architecture:** Embedding(vocab, 64) → BiLSTM(100) → Dense(128, relu) → Dropout(0.3) → Dense(1, sigmoid)
- **Training:** Adam optimizer, binary cross-entropy loss, early stopping on val_loss (patience=5)
- **MLflow tracking:** Logs all hyperparameters, per-epoch metrics, classification report, confusion matrix, model artifact, tokenizer artifact
- **Model Registry:** Best model registered as `SentimentBiLSTM` in MLflow; promoted to `Production` stage

### 3.3 Inference API (FastAPI)
- Loads model from local SavedModel path (or MLflow registry as fallback)
- Applies identical preprocessing pipeline at inference time
- Exposes `/predict` (single) and `/predict/batch` (up to 50) endpoints
- Prometheus-instrumented via `prometheus-fastapi-instrumentator`
- `/health` and `/ready` probes for container orchestration

### 3.4 Frontend (React + Vite)
- Single-page application with two views: **Analyze** and **ML Pipeline**
- Analyze view: single review input with example prompts, batch mode (one review per line), prediction history sidebar
- Pipeline view: real-time API status, visual pipeline stage diagram, tech stack summary, links to MLflow/Grafana/Airflow UIs

### 3.5 Monitoring (Prometheus + Grafana)
- **Metrics tracked:** prediction counts (by sentiment), inference latency (p50/p95/p99), positive prediction ratio, input text length distribution, HTTP request rates and error rates
- **Grafana dashboard:** Pre-provisioned via JSON, auto-connects to Prometheus datasource
- **Alerting:** API down, error rate > 5%, sentiment ratio drift (outside 10%–90%), p99 latency > 500ms

### 3.6 Containerization (Docker Compose)
Five isolated services: `frontend`, `backend`, `mlflow`, `prometheus`, `grafana`
Loose coupling: frontend communicates with backend only via REST API

---

## 4. Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Model architecture | Bidirectional LSTM | Captures forward and backward context in reviews; good balance of accuracy and inference speed |
| API framework | FastAPI | Automatic OpenAPI docs, Pydantic validation, async support, easy Prometheus integration |
| Frontend framework | React + Vite | Fast HMR for development, lightweight production build, component-based UI |
| Experiment tracking | MLflow | Self-hosted (no cloud), integrates with Keras natively, provides model registry |
| Data pipeline orchestration | Airflow + DVC | Airflow for scheduling/visualization; DVC for reproducibility and artifact versioning |
| Monitoring | Prometheus + Grafana | Industry standard, self-hosted, pre-built dashboard provisioning |
| Containerization | Docker Compose | Meets project requirements, ensures environment parity, loose coupling between services |

---

## 5. Data Flow

```
Raw CSV
  → [Airflow: ingest]  → validated CSV
  → [Airflow: preprocess] → train/val/test .npy + tokenizer.pkl
  → [DVC add/push] → artifacts versioned in DVC remote
  → [Kaggle: train] → bilstm_model/ + mlruns/
  → [MLflow registry] → SentimentBiLSTM@Production
  → [FastAPI: serve] → /predict endpoint
  → [React: UI] → user sees sentiment + confidence
  → [Prometheus: scrape] → metrics stored
  → [Grafana: visualize] → real-time dashboard
```

---

## 6. Security Considerations
- All containers communicate on an internal Docker network; only required ports are exposed to the host
- Sensitive config values (passwords, keys) should be moved to `.env` files (not committed to Git)
- Input validation via Pydantic schemas prevents malformed payloads
- CORS restricted to known origins in production deployment
