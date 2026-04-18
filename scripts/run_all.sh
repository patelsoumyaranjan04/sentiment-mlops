#!/bin/bash
# =================================================================
# MASTER SETUP SCRIPT — sentiment-mlops
# Run from the project root after all files are in place.
# =================================================================
set -e

echo ""
echo "============================================================"
echo "  SENTIMENT MLOPS — Full Project Setup"
echo "============================================================"
echo ""

# ---- [1] Verify project structure ----
echo "[1/8] Verifying project structure..."
REQUIRED_DIRS=(
  "src/data" "src/models" "src/api" "src/monitoring" "src/utils"
  "airflow/dags" "configs" "data/raw" "data/processed"
  "docker" "frontend/src" "tests/unit" "tests/integration"
  "docs" "mlflow_project"
)
for d in "${REQUIRED_DIRS[@]}"; do
  if [ ! -d "$d" ]; then
    echo "  Creating missing dir: $d"
    mkdir -p "$d"
  fi
done
echo "  ✅ Structure OK"

# ---- [2] Check raw data ----
echo ""
echo "[2/8] Checking raw data..."
if [ ! -f "data/raw/Amazon_review.csv" ]; then
  echo "  ⚠️  WARNING: data/raw/Amazon_review.csv not found!"
  echo "  Download from: https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews"
  echo "  Place it at: data/raw/Amazon_review.csv"
  echo "  Skipping data pipeline for now..."
else
  echo "  ✅ Raw data found"
fi

# ---- [3] Conda environment ----
echo ""
echo "[3/8] Setting up conda environment..."
if conda env list | grep -q "sentiment-mlops"; then
  echo "  ✅ Environment 'sentiment-mlops' already exists"
else
  conda env create -f environment.yml
  echo "  ✅ Environment created"
fi
echo "  → Run: conda activate sentiment-mlops"

# ---- [4] DVC init ----
echo ""
echo "[4/8] Initializing DVC..."
if [ ! -d ".dvc" ]; then
  conda run -n sentiment-mlops dvc init
  conda run -n sentiment-mlops dvc remote add -d localremote ~/dvc-storage
  echo "  ✅ DVC initialized with local remote at ~/dvc-storage"
else
  echo "  ✅ DVC already initialized"
fi

# ---- [5] Run data pipeline (if raw data exists) ----
echo ""
echo "[5/8] Running data pipeline..."
if [ -f "data/raw/Amazon_review.csv" ]; then
  conda run -n sentiment-mlops python -m src.data.ingest
  conda run -n sentiment-mlops python -m src.data.preprocess
  conda run -n sentiment-mlops dvc add data/processed/ data/raw/Amazon_review_validated.csv
  conda run -n sentiment-mlops dvc push
  echo "  ✅ Data pipeline complete"
else
  echo "  ⏭️  Skipped (no raw data)"
fi

# ---- [6] Run unit tests ----
echo ""
echo "[6/8] Running unit tests..."
conda run -n sentiment-mlops pytest tests/unit/ -v --tb=short
echo "  ✅ Unit tests complete"

# ---- [7] MLflow server ----
echo ""
echo "[7/8] Starting MLflow server (background)..."
if ! pgrep -f "mlflow server" > /dev/null; then
  conda run -n sentiment-mlops mlflow server \
    --host 0.0.0.0 --port 5000 \
    --backend-store-uri ./mlruns \
    --default-artifact-root ./mlartifacts &
  sleep 3
  echo "  ✅ MLflow running at http://localhost:5000"
else
  echo "  ✅ MLflow already running"
fi

# ---- [8] Docker Compose ----
echo ""
echo "[8/8] Starting Docker Compose services..."
echo "  NOTE: Requires bilstm_model/ and tokenizer.pkl in data/processed/"
echo "  (Download these from Kaggle after training)"
echo ""
read -p "  Start Docker Compose now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  docker compose up --build -d
  echo ""
  echo "  Waiting for services to be healthy..."
  sleep 15
  echo ""
  echo "  ✅ Services started:"
  echo "     Frontend  → http://localhost:3000"
  echo "     API       → http://localhost:8000"
  echo "     API Docs  → http://localhost:8000/docs"
  echo "     MLflow    → http://localhost:5000"
  echo "     Prometheus→ http://localhost:9090"
  echo "     Grafana   → http://localhost:3001  (admin/admin)"
  echo "     Airflow   → http://localhost:8080  (admin/admin)"
fi

echo ""
echo "============================================================"
echo "  SETUP COMPLETE"
echo "============================================================"
echo ""
echo "NEXT STEPS:"
echo "  1. If not done: Download Amazon_review.csv → data/raw/"
echo "  2. If not done: Run Kaggle training → download bilstm_model/ + tokenizer.pkl → data/processed/"
echo "  3. conda activate sentiment-mlops"
echo "  4. pytest tests/ -v   (run all tests)"
echo "  5. docker compose up --build   (start all services)"
echo ""
