#!/bin/bash
# ============================================================
# Stage 1: Full Repository & Environment Setup Script
# Run this on your Ubuntu VirtualBox
# ============================================================
set -e

echo "=== [1/6] Creating project folder structure ==="
mkdir -p sentiment-mlops && cd sentiment-mlops
git init

mkdir -p \
  data/raw \
  data/processed \
  airflow/dags \
  airflow/plugins \
  src/data \
  src/features \
  src/models \
  src/api \
  src/monitoring \
  src/utils \
  frontend \
  mlflow_project \
  docker \
  tests/unit \
  tests/integration \
  docs \
  notebooks \
  configs \
  scripts \
  logs

# Keep empty dirs in git
find . -type d -empty -exec touch {}/.gitkeep \;

echo "=== [2/6] Copying config files from stage1 output ==="
# Copy files from wherever you downloaded them
# (Adjust source path to wherever you saved the stage1 outputs)
# cp /path/to/stage1/.gitignore .
# cp /path/to/stage1/environment.yml .
# cp /path/to/stage1/README.md .
# cp /path/to/stage1/configs/config.yaml configs/
# cp /path/to/stage1/src/utils/config_loader.py src/utils/
# cp /path/to/stage1/src/utils/logger.py src/utils/

echo "=== [3/6] Creating conda environment ==="
conda env create -f environment.yml
echo "Run: conda activate sentiment-mlops"

echo "=== [4/6] Initializing DVC ==="
# Run these AFTER activating the conda env:
# dvc init
# dvc remote add -d myremote gdrive://<your-gdrive-folder-id>
# (Or use a local remote for now:)
# dvc remote add -d localremote /tmp/dvc-storage

echo "=== [5/6] Creating __init__.py files ==="
touch src/__init__.py
touch src/data/__init__.py
touch src/features/__init__.py
touch src/models/__init__.py
touch src/api/__init__.py
touch src/monitoring/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py

echo "=== [6/6] Initial git commit ==="
# git add .
# git commit -m "chore: initial project scaffold"
# git remote add origin https://github.com/<your-username>/sentiment-mlops.git
# git push -u origin main

echo ""
echo "✅ Stage 1 complete! Next steps:"
echo "  1. conda activate sentiment-mlops"
echo "  2. dvc init"
echo "  3. Set up DVC remote (GDrive or local)"
echo "  4. Push to GitHub"
echo "  5. Proceed to Stage 2: Data Engineering Pipeline"
