#!/bin/bash
# ============================================================
# Install Apache Airflow into the already-active conda env
# Run AFTER: conda activate sentiment-mlops
# ============================================================

AIRFLOW_VERSION=2.7.3
PYTHON_VERSION=3.10
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

echo "=== Installing Apache Airflow ${AIRFLOW_VERSION} with constraints ==="
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

echo "=== Initializing Airflow DB ==="
export AIRFLOW_HOME=~/Desktop/sentiment-mlops/airflow
airflow db init

echo ""
echo "✅ Airflow installed. To start:"
echo "  export AIRFLOW_HOME=~/Desktop/sentiment-mlops/airflow"
echo "  airflow standalone"