"""
scripts/register_model.py
--------------------------
Registers the latest MLflow run's model in the MLflow
Model Registry and transitions it to Production stage.

Run from project root:
    python scripts/register_model.py

This implements proper MLflow model versioning:
  - Registers model as "SentimentBiLSTM"
  - Creates a new version
  - Transitions to Production stage
  - Archives any previous Production version
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("register_model")


def register_model():
    import os
    import mlflow
    from mlflow.tracking import MlflowClient

    cfg = load_config()

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{Path('mlruns/mlflow.db').resolve()}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    model_name = cfg["mlflow"]["registered_model_name"]

    # Get the latest run from the experiment
    runs = client.search_runs(
        experiment_ids=["1"],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        logger.error("No runs found. Train the model first.")
        return

    run = runs[0]
    run_id = run.info.run_id
    logger.info(f"Latest run: {run_id}")
    logger.info(f"Metrics: {run.data.metrics}")

    # Register the model
    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Registering model from: {model_uri}")

    try:
        # Create registered model if it doesn't exist
        try:
            client.create_registered_model(
                name=model_name,
                description="Bidirectional LSTM for Amazon review sentiment classification.",
                tags={
                    "framework":    "tensorflow",
                    "task":         "binary_classification",
                    "dataset":      "amazon_reviews",
                    "architecture": "BiLSTM",
                }
            )
            logger.info(f"Created registered model: {model_name}")
        except Exception:
            logger.info(f"Model '{model_name}' already exists, creating new version.")

        # Create a new version
        version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
            description=(
                f"BiLSTM model — "
                f"accuracy={run.data.metrics.get('test_accuracy', 0):.4f}, "
                f"roc_auc={run.data.metrics.get('test_roc_auc', 0):.4f}"
            ),
        )
        logger.info(f"Created model version: {version.version}")

        # Archive previous Production versions
        for mv in client.search_model_versions(f"name='{model_name}'"):
            if mv.current_stage == "Production" and mv.version != version.version:
                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Archived",
                )
                logger.info(f"Archived previous version: {mv.version}")

        # Transition new version to Production
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Production",
        )
        logger.info(
            f"✅ Model '{model_name}' version {version.version} "
            f"transitioned to Production"
        )

        # Add tags
        client.set_model_version_tag(
            name=model_name,
            version=version.version,
            key="test_accuracy",
            value=str(run.data.metrics.get("test_accuracy", "")),
        )
        client.set_model_version_tag(
            name=model_name,
            version=version.version,
            key="test_roc_auc",
            value=str(run.data.metrics.get("test_roc_auc", "")),
        )

        print(f"\n{'='*50}")
        print(f"Model registered: {model_name}")
        print(f"Version:          {version.version}")
        print(f"Stage:            Production")
        print(f"Run ID:           {run_id}")
        print(f"Test Accuracy:    {run.data.metrics.get('test_accuracy', 'N/A'):.4f}")
        print(f"Test ROC-AUC:     {run.data.metrics.get('test_roc_auc', 'N/A'):.4f}")
        print(f"{'='*50}\n")

    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        raise


if __name__ == "__main__":
    register_model()