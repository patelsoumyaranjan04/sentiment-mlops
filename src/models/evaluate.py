"""
src/models/evaluate.py
-----------------------
Runs error analysis on the test set:
- Finds misclassified examples
- Analyzes by review length
- Saves error analysis report to MLflow

Run: python -m src.models.evaluate
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("evaluate")


def evaluate():
    cfg      = load_config()
    proc_dir = Path(cfg["data"]["processed_dir"])

    # Load test data
    X_test  = np.load(proc_dir / "X_test.npy")
    y_test  = np.load(proc_dir / "y_test.npy")
    test_df = pd.read_csv(proc_dir / "test.csv")

    # Load model
    import tensorflow as tf
    model = tf.keras.models.load_model(cfg["model"]["model_save_path"])

    # Load threshold
    threshold_path = proc_dir / "optimal_threshold.json"
    threshold = 0.5
    if threshold_path.exists():
        threshold = json.loads(threshold_path.read_text())["threshold"]
    logger.info(f"Using threshold: {threshold}")

    # Predict
    y_prob = model.predict(X_test, batch_size=128).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    # Add predictions to dataframe
    test_df["y_true"]      = y_test
    test_df["y_pred"]      = y_pred
    test_df["y_prob"]      = y_prob
    test_df["correct"]     = (y_test == y_pred)
    test_df["review_len"]  = test_df["clean_review"].str.split().str.len()
    test_df["confidence"]  = np.where(y_pred == 1, y_prob, 1 - y_prob)

    # Misclassified examples
    errors = test_df[~test_df["correct"]].copy()
    logger.info(f"Total errors: {len(errors)} / {len(test_df)} "
                f"({len(errors)/len(test_df)*100:.1f}%)")

    # False positives and negatives
    fp = errors[errors["y_pred"] == 1]  # predicted positive, actually negative
    fn = errors[errors["y_pred"] == 0]  # predicted negative, actually positive
    logger.info(f"False Positives: {len(fp)}, False Negatives: {len(fn)}")

    # Error analysis by review length
    bins   = [0, 10, 20, 40, 80, 200]
    labels = ["0-10", "10-20", "20-40", "40-80", "80+"]
    test_df["length_bin"] = pd.cut(
        test_df["review_len"], bins=bins, labels=labels
    )
    length_analysis = test_df.groupby("length_bin", observed=True).agg(
        total=("correct", "count"),
        correct=("correct", "sum"),
        accuracy=("correct", "mean"),
    ).round(4)
    logger.info(f"\nAccuracy by review length:\n{length_analysis}")

    # High confidence errors (model was very wrong)
    high_conf_errors = errors[errors["confidence"] > 0.8].sort_values(
        "confidence", ascending=False
    )
    logger.info(f"High confidence errors (conf > 0.8): {len(high_conf_errors)}")

    # Save error analysis report
    report = {
        "total_test":           len(test_df),
        "total_errors":         len(errors),
        "error_rate":           round(len(errors) / len(test_df), 4),
        "false_positives":      len(fp),
        "false_negatives":      len(fn),
        "high_conf_errors":     len(high_conf_errors),
        "threshold_used":       threshold,
        "accuracy_by_length":   length_analysis.reset_index().to_dict(orient="records"),
        "worst_fp_examples":    fp.nlargest(5, "confidence")[
            ["clean_review", "y_prob"]
        ].to_dict(orient="records"),
        "worst_fn_examples":    fn.nsmallest(5, "confidence")[
            ["clean_review", "y_prob"]
        ].to_dict(orient="records"),
    }

    report_path = proc_dir / "error_analysis.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Error analysis saved to {report_path}")

    # Log to MLflow if tracking URI is set
    try:
        import mlflow
        import os
        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI",
            f"sqlite:///{Path('mlruns/mlflow.db').resolve()}"
        )
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs("1")
        if runs:
            run_id = runs[0].info.run_id
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(str(report_path), artifact_path="evaluation")
                mlflow.log_metrics({
                    "error_rate":       report["error_rate"],
                    "false_positives":  report["false_positives"],
                    "false_negatives":  report["false_negatives"],
                })
            logger.info(f"Error analysis logged to MLflow run: {run_id}")
    except Exception as e:
        logger.warning(f"Could not log to MLflow: {e}")

    logger.info("✅ Error analysis complete")
    return report


if __name__ == "__main__":
    evaluate()