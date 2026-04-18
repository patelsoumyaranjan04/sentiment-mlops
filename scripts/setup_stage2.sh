#!/bin/bash
# ============================================================
# Stage 2: Data Engineering Pipeline Setup
# Run from the project root: sentiment-mlops/
# ============================================================

# ---- 1. Copy new files into the project ----
# Copy these into your project (from wherever you saved the stage2 outputs):
#
#   src/data/ingest.py          -> sentiment-mlops/src/data/ingest.py
#   src/data/preprocess.py      -> sentiment-mlops/src/data/preprocess.py
#   airflow/dags/sentiment_data_pipeline.py -> sentiment-mlops/airflow/dags/
#   dvc.yaml                    -> sentiment-mlops/dvc.yaml
#   docker/docker-compose.airflow.yml -> sentiment-mlops/docker/
#   tests/unit/test_data_pipeline.py -> sentiment-mlops/tests/unit/

# ---- 2. Place raw data ----
# Download Amazon_review.csv from Kaggle:
# https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews
# Place it at:  sentiment-mlops/data/raw/Amazon_review.csv

# ---- 3. Test the pipeline locally (no Airflow needed yet) ----
conda activate sentiment-mlops
cd sentiment-mlops

python -m src.data.ingest
python -m src.data.preprocess

# ---- 4. Run unit tests ----
pytest tests/unit/test_data_pipeline.py -v

# ---- 5. DVC pipeline (alternative to Airflow for local runs) ----
dvc repro         # runs ingest -> preprocess stages defined in dvc.yaml
dvc dag           # visualize the pipeline DAG in terminal
dvc push          # push artifacts to your DVC remote

# ---- 6. Git commit ----
git add src/data/ airflow/ dvc.yaml dvc.lock tests/unit/test_data_pipeline.py
git commit -m "feat: add data ingestion and preprocessing pipeline (Airflow + DVC)"
git push

# ---- 7. Start Airflow (optional, for UI visualization) ----
# Requires Docker installed:
docker compose -f docker/docker-compose.airflow.yml up airflow-init
docker compose -f docker/docker-compose.airflow.yml up -d

# Then open: http://localhost:8080
# Login: admin / admin
# Enable the DAG: sentiment_data_pipeline
# Trigger it manually to see it run

echo "Stage 2 complete!"
