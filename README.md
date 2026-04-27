
Soumya Ranjan Patel || DA25M029


# SentiAI — Amazon Review Sentiment Analysis

> Binary sentiment classification of Amazon product reviews using a Bidirectional LSTM,
> served through a production-grade MLOps pipeline running entirely on-device.

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | 85.76% |
| ROC-AUC | 0.936 |
| F1 Score | 0.861 |
| Optimal Threshold | 0.4879 |
| p95 Inference Latency | < 200ms |

---

## 🧱 Tech Stack

| Layer | Tool |
|---|---|
| Model | TensorFlow / Keras BiLSTM |
| Data Pipeline | Apache Airflow + DVC |
| Experiment Tracking | MLflow (SQLite) |
| API Serving | FastAPI + Uvicorn |
| Frontend | React + Vite + nginx |
| Monitoring | Prometheus + Grafana |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Compose |

---

## 🚀 Quick Start

### Prerequisites
- Docker + Docker Compose
- conda (for local development)
- Git + DVC

### 1. Clone and setup environment
```bash
git clone https://github.com/[YOUR_GITHUB_USERNAME]/sentiment-mlops.git
cd sentiment-mlops
conda env create -f environment.yml
conda activate sentiment-mlops
```

### 2. Pull data artifacts
```bash
dvc pull
```

### 3. Start all services
```bash
docker compose up --build
```

### 4. Access the application

| Service | URL | Credentials |
|---|---|---|
| Frontend | http://localhost:3002 | — |
| API + Swagger | http://localhost:8000/docs | — |
| MLflow | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3003 | admin / admin |
| Airflow | http://localhost:8080 | admin / [see logs] |

---

## 🔄 DVC Pipeline

```bash
dvc repro      # run full pipeline
dvc dag        # visualize pipeline
dvc metrics show  # view model metrics
```

Pipeline stages:
```
ingest → preprocess → featurize → train
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v --tb=short
# Expected: 36 passed
```

---

## 📁 Project Structure

```
sentiment-mlops/
├── airflow/dags/          # Airflow DAG for data pipeline
├── configs/config.yaml    # Central configuration
├── data/processed/        # Model artifacts (DVC tracked)
├── docker/                # Dockerfiles + monitoring configs
├── frontend/              # React web application
├── mlruns/                # MLflow tracking (SQLite)
├── scripts/               # Utility scripts
├── src/
│   ├── api/               # FastAPI inference server
│   ├── data/              # Ingestion + preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # Training + evaluation
│   ├── monitoring/        # Prometheus metrics + drift detection
│   └── utils/             # Config loader + logger
└── tests/                 # Unit + integration test suites
```

---

## 📖 Documentation

| Document | Location |
|---|---|
| Architecture + HLD | `Architecture & HLD.pdf` |
| API Spec + LLD | `LLD.pdf` |
| Test Plan | `Test Plan & Test Cases.pdf` |
| User Manual | `USER_MANUAL.pdf` |

---

