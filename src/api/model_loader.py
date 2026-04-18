"""
src/api/model_loader.py
------------------------
Loads the trained Keras model and tokenizer for inference.
Supports both local SavedModel path and MLflow model URI.
"""

import os
import pickle
from pathlib import Path

import tensorflow as tf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

logger = get_logger("model_loader")


def load_model_and_tokenizer(cfg: dict):
    """
    Load model + tokenizer.
    Priority:
      1. Local SavedModel at cfg["model"]["model_save_path"]
      2. MLflow model URI at cfg["api"]["model_uri"]  (if MLflow tracking server is up)
    """
    model     = _load_model(cfg)
    tokenizer = _load_tokenizer(cfg)
    return model, tokenizer


def _load_model(cfg: dict):
    local_path = Path(cfg["model"]["model_save_path"])

    if local_path.exists():
        logger.info(f"Loading model from local path: {local_path}")
        model = tf.keras.models.load_model(str(local_path))
        logger.info("Model loaded from local SavedModel.")
        return model

    # Fallback: MLflow model registry
    try:
        import mlflow.keras
        mlflow_uri = cfg["mlflow"]["tracking_uri"]
        model_uri  = cfg["api"]["model_uri"]
        logger.info(f"Local model not found. Trying MLflow URI: {model_uri}")
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)
        model = mlflow.keras.load_model(model_uri)
        logger.info("Model loaded from MLflow registry.")
        return model
    except Exception as e:
        raise RuntimeError(
            f"Could not load model from local path '{local_path}' "
            f"or MLflow registry. Error: {e}"
        )


def _load_tokenizer(cfg: dict):
    tok_path = Path(cfg["preprocessing"]["tokenizer_save_path"])
    json_path = tok_path.parent / "tokenizer.json"

    # Prefer JSON format — fully portable, no keras_preprocessing dependency
    if json_path.exists():
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        tokenizer = tokenizer_from_json(json_path.read_text())
        logger.info(f"Tokenizer loaded from JSON at {json_path} (vocab_size={len(tokenizer.word_index)+1})")
        return tokenizer

    # Fallback: pickle — patch module path for Kaggle-exported tokenizers
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tok_path} or {json_path}. "
            "Run scripts/fix_tokenizer.py first."
        )
    with open(tok_path, "rb") as f:
        raw = f.read()

    # Patch keras_preprocessing → tensorflow.keras.preprocessing if needed
    patched = raw.replace(
        b"keras_preprocessing.text",
        b"tensorflow.keras.preprocessing.text"
    )
    tokenizer = pickle.loads(patched)
    logger.info(f"Tokenizer loaded from pickle at {tok_path} (vocab_size={len(tokenizer.word_index)+1})")
    return tokenizer