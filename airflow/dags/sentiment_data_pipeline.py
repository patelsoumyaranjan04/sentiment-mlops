"""
airflow/dags/sentiment_data_pipeline.py
----------------------------------------
Airflow DAG that orchestrates:
  1. data_ingest   — validate raw CSV, drop nulls/duplicates
  2. data_preprocess — clean text, split, tokenize, pad, save arrays
  3. dvc_add        — track processed artifacts with DVC
  4. dvc_push       — push artifacts to DVC remote

Schedule: daily (can be triggered manually)
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# ------------------------------------------------------------------ #
#  Default args                                                        #
# ------------------------------------------------------------------ #
default_args = {
    "owner": "sentiment-mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Project root — adjust if Airflow home differs
PROJECT_ROOT = Path("/opt/airflow/project")  # inside Docker
# If running Airflow locally (not Docker), change to your absolute path:
# PROJECT_ROOT = Path("/home/<your-user>/sentiment-mlops")


# ------------------------------------------------------------------ #
#  Python callables                                                    #
# ------------------------------------------------------------------ #

def run_ingest(**context):
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data.ingest import ingest
    from src.utils.config_loader import load_config
    cfg = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    df = ingest(cfg)
    context["ti"].xcom_push(key="row_count", value=len(df))


def run_preprocess(**context):
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data.preprocess import preprocess
    from src.utils.config_loader import load_config
    cfg = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    preprocess(cfg)


# ------------------------------------------------------------------ #
#  DAG definition                                                      #
# ------------------------------------------------------------------ #

with DAG(
    dag_id="sentiment_data_pipeline",
    description="Data ingestion and preprocessing pipeline for Amazon Sentiment",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["sentiment", "nlp", "mlops"],
) as dag:

    # ---- Task 1: Ingest ----
    ingest_task = PythonOperator(
        task_id="data_ingest",
        python_callable=run_ingest,
        provide_context=True,
        doc_md="""
        ## Data Ingestion
        Reads `data/raw/Amazon_review.csv`, validates schema,
        drops nulls/duplicates, saves validated snapshot.
        """,
    )

    # ---- Task 2: Preprocess ----
    preprocess_task = PythonOperator(
        task_id="data_preprocess",
        python_callable=run_preprocess,
        provide_context=True,
        doc_md="""
        ## Data Preprocessing
        Cleans text, splits into train/val/test,
        fits tokenizer, pads sequences, saves numpy arrays.
        Computes baseline stats for drift detection.
        """,
    )

    # ---- Task 3: DVC add processed artifacts ----
    dvc_add_task = BashOperator(
        task_id="dvc_add",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "dvc add data/processed/ data/raw/Amazon_review_validated.csv"
        ),
        doc_md="Tracks processed data artifacts with DVC.",
    )

    # ---- Task 4: DVC push to remote ----
    dvc_push_task = BashOperator(
        task_id="dvc_push",
        bash_command=f"cd {PROJECT_ROOT} && dvc push",
        doc_md="Pushes DVC-tracked artifacts to the configured remote.",
    )

    # ---- Dependencies ----
    ingest_task >> preprocess_task >> dvc_add_task >> dvc_push_task
