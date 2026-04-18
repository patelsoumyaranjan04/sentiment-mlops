# 🛒 Amazon Reviews Sentiment Analysis — MLOps Pipeline

Binary sentiment classification (positive / negative) of Amazon product reviews  
using a **Bidirectional LSTM** model, wrapped in a full **MLOps lifecycle**.

---

## 📐 Architecture Overview

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  React UI   │───▶│  FastAPI Backend │───▶│  MLflow Model    │
│  (Frontend) │    │  (Inference API) │    │  Registry        │
└─────────────┘    └──────────────────┘    └──────────────────┘
                           │                        │
                    ┌──────▼──────┐         ┌──────▼──────┐
                    │  Prometheus │         │  Airflow    │
                    │  + Grafana  │         │  DAG        │
                    └─────────────┘         └─────────────┘
```

## 🧱 Tech Stack

| Layer | Tool |
|---|---|
| Model | TensorFlow/Keras BiLSTM |
| Experiment Tracking | MLflow |
| Data Version Control | DVC |
| Data Pipeline | Apache Airflow |
| API Serving | FastAPI |
| Frontend | React |
| Monitoring | Prometheus + Grafana |
| Containerization | Docker + Docker Compose |
| Source Control | Git + Git LFS |

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone <repo-url>
cd sentiment-mlops

# 2. Create environment
conda env create -f environment.yml
conda activate sentiment-mlops

# 3. Pull data with DVC
dvc pull

# 4. Start all services
docker compose up --build
```

## 📁 Project Structure

```
sentiment-mlops/
├── airflow/           # Airflow DAGs for data pipeline
├── configs/           # Central YAML configuration
├── data/
│   ├── raw/           # Raw CSV (DVC tracked)
│   └── processed/     # Tokenizer, train/test splits
├── docker/            # Dockerfiles
├── docs/              # HLD, LLD, test plan, user manual
├── frontend/          # React web application
├── mlflow_project/    # MLproject file + conda env
├── notebooks/         # Reference/EDA notebooks
├── scripts/           # Utility scripts
├── src/
│   ├── data/          # Data ingestion & preprocessing
│   ├── features/      # Feature engineering
│   ├── models/        # Model training & evaluation
│   ├── api/           # FastAPI inference server
│   ├── monitoring/    # Prometheus exporters
│   └── utils/         # Config loader, logger
└── tests/             # Unit & integration tests
```

## 📊 Model Performance

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Bidirectional LSTM | ~94% | ~0.94 |
| XGBoost (baseline) | ~91% | ~0.91 |

## 📖 Documentation

See the `docs/` folder for:
- Architecture Diagram
- High-Level Design (HLD)
- Low-Level Design (LLD) with API specs
- Test Plan & Test Cases
- User Manual
