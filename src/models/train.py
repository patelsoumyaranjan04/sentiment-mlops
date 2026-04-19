"""
src/models/train.py
--------------------
Trains the Bidirectional LSTM for Amazon Sentiment classification
entirely on-device. Tracks every run with MLflow.

Run:
    python -m src.models.train

Or via DVC (recommended):
    dvc repro train
"""

import os
import sys
import json
import pickle
from pathlib import Path

import numpy as np
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("train")


# ------------------------------------------------------------------ #
#  Model builder                                                       #
# ------------------------------------------------------------------ #

def build_model(vocab_size: int, cfg: dict) -> tf.keras.Model:
    m_cfg = cfg["model"]
    model = Sequential([
        Embedding(vocab_size, m_cfg["embedding_dim"], mask_zero=True),
        Bidirectional(LSTM(
            m_cfg["lstm_units"],
            dropout=m_cfg["dropout"],
            recurrent_dropout=m_cfg["recurrent_dropout"],
            return_sequences=False,
        )),
        Dense(m_cfg["dense_units"], activation="relu"),
        Dropout(0.3),
        Dense(1, activation=m_cfg["output_activation"]),
    ])
    model.compile(
        optimizer=m_cfg["optimizer"],
        loss=m_cfg["loss"],
        metrics=m_cfg["metrics"],
    )
    return model


# ------------------------------------------------------------------ #
#  Training                                                            #
# ------------------------------------------------------------------ #

def train(cfg: dict | None = None) -> None:
    if cfg is None:
        cfg = load_config()

    proc_cfg = cfg["preprocessing"]
    m_cfg    = cfg["model"]
    mlf_cfg  = cfg["mlflow"]
    data_cfg = cfg["data"]
    proc_dir = Path(data_cfg["processed_dir"])

    # ---- Load processed data ----
    logger.info("Loading preprocessed data ...")
    X_train = np.load(proc_dir / "X_train.npy")
    X_val   = np.load(proc_dir / "X_val.npy")
    X_test  = np.load(proc_dir / "X_test.npy")
    y_train = np.load(proc_dir / "y_train.npy")
    y_val   = np.load(proc_dir / "y_val.npy")
    y_test  = np.load(proc_dir / "y_test.npy")

    with open(proc_cfg["tokenizer_save_path"], "rb") as f:
        tokenizer = pickle.load(f)

    vocab_size = len(tokenizer.word_index) + 1
    logger.info(f"vocab_size={vocab_size}, train={X_train.shape}, val={X_val.shape}")

    # ---- MLflow setup ----
    # Use SQLite by default — works fully on-device without a separate server.
    # Override with MLFLOW_TRACKING_URI env var to point at the Docker MLflow server.
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{Path('mlruns/mlflow.db').resolve()}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlf_cfg["experiment_name"])
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        # ---- Log hyperparameters ----
        params = {
            "vocab_size":              vocab_size,
            "embedding_dim":           m_cfg["embedding_dim"],
            "lstm_units":              m_cfg["lstm_units"],
            "dense_units":             m_cfg["dense_units"],
            "dropout":                 m_cfg["dropout"],
            "recurrent_dropout":       m_cfg["recurrent_dropout"],
            "batch_size":              m_cfg["batch_size"],
            "max_sequence_length":     proc_cfg["max_sequence_length"],
            "optimizer":               m_cfg["optimizer"],
            "epochs":                  m_cfg["epochs"],
            "early_stopping_patience": m_cfg["early_stopping_patience"],
            "test_size":               proc_cfg["test_size"],
            "val_size":                proc_cfg["val_size"],
            "random_state":            proc_cfg["random_state"],
        }
        mlflow.log_params(params)

        # Log baseline stats
        baseline_path = proc_dir / "baseline_stats.json"
        if baseline_path.exists():
            mlflow.log_artifact(str(baseline_path), artifact_path="data_stats")

        # ---- Build model ----
        model = build_model(vocab_size, cfg)
        model.summary()

        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        mlflow.log_text("\n".join(stringlist), "model_summary.txt")

        # ---- Callbacks ----
        checkpoint_dir = proc_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=m_cfg["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(checkpoint_dir / "best_model.h5"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        # ---- Train ----
        logger.info("Starting training ...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=m_cfg["epochs"],
            batch_size=m_cfg["batch_size"],
            callbacks=callbacks,
            verbose=1,
        )

        # ---- Log per-epoch metrics ----
        for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
            history.history["loss"],
            history.history["accuracy"],
            history.history["val_loss"],
            history.history["val_accuracy"],
        )):
            mlflow.log_metrics({
                "train_loss":     loss,
                "train_accuracy": acc,
                "val_loss":       val_loss,
                "val_accuracy":   val_acc,
            }, step=epoch)

        # ---- Evaluate on test set ----
        logger.info("Evaluating on test set ...")
        y_prob = model.predict(X_test, batch_size=m_cfg["batch_size"]).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "test_accuracy":  float(accuracy_score(y_test, y_pred)),
            "test_f1":        float(f1_score(y_test, y_pred)),
            "test_precision": float(precision_score(y_test, y_pred)),
            "test_recall":    float(recall_score(y_test, y_pred)),
            "test_roc_auc":   float(roc_auc_score(y_test, y_prob)),
        }
        mlflow.log_metrics(metrics)

        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        # ---- Save metrics JSON for DVC tracking ----
        metrics_path = proc_dir / "train_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")

        # ---- Log classification report ----
        report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"])
        mlflow.log_text(report, "classification_report.txt")
        logger.info(f"\n{report}")

        # ---- Log confusion matrix ----
        cm = confusion_matrix(y_test, y_pred).tolist()
        mlflow.log_text(json.dumps({"confusion_matrix": cm}), "confusion_matrix.json")

        # ---- Save model ----
        model_save_path = Path(m_cfg["model_save_path"])
        model.save(str(model_save_path))
        logger.info(f"Model saved to {model_save_path}")

        # Log model artifact to MLflow
        sample_input  = X_test[:5]
        sample_output = model.predict(sample_input)
        signature = infer_signature(sample_input, sample_output)

        mlflow.keras.log_model(
            model,
            artifact_path="model",
            signature=signature,
        )

        # Log tokenizer
        tok_json = proc_dir / "tokenizer.json"
        if tok_json.exists():
            mlflow.log_artifact(str(tok_json), artifact_path="tokenizer")

        logger.info(f"✅ Training complete. Run ID: {run_id}")
        logger.info(f"   Test Accuracy : {metrics['test_accuracy']:.4f}")
        logger.info(f"   Test ROC-AUC  : {metrics['test_roc_auc']:.4f}")


if __name__ == "__main__":
    train()