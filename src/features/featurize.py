"""
src/features/featurize.py
--------------------------
Feature engineering stage — separate from text cleaning.
Takes clean text CSVs and produces:
  - Fitted tokenizer (SimpleTokenizer JSON)
  - Padded numpy arrays (X_train, X_val, X_test)
  - Label arrays (y_train, y_val, y_test)

This stage is versioned separately from text preprocessing
so tokenizer changes don't force re-cleaning.

Run: python -m src.features.featurize
Or via DVC: dvc repro featurize
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
from src.data.preprocess import SimpleTokenizer, pad_sequences

logger = get_logger("featurize")


def featurize(cfg: dict | None = None) -> None:
    if cfg is None:
        cfg = load_config()

    proc_cfg    = cfg["preprocessing"]
    data_cfg    = cfg["data"]
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    proc_dir    = PROJECT_ROOT / data_cfg["processed_dir"]
    max_len     = proc_cfg["max_sequence_length"]

    # Load clean text splits
    logger.info("Loading clean text splits...")
    train_df = pd.read_csv(proc_dir / "train.csv")
    val_df   = pd.read_csv(proc_dir / "val.csv")
    test_df  = pd.read_csv(proc_dir / "test.csv")

    X_train = train_df["clean_review"].values
    X_val   = val_df["clean_review"].values
    X_test  = test_df["clean_review"].values
    y_train = train_df["sentiment"].values
    y_val   = val_df["sentiment"].values
    y_test  = test_df["sentiment"].values

    # Fit tokenizer on training data only
    logger.info("Fitting tokenizer on training data...")
    tokenizer = SimpleTokenizer()
    tokenizer.fit_on_texts(X_train)
    vocab_size = len(tokenizer.word_index) + 1
    logger.info(f"Vocabulary size: {vocab_size}")

    # Encode and pad sequences
    def encode(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=max_len)

    logger.info("Encoding and padding sequences...")
    X_train_pad = encode(X_train)
    X_val_pad   = encode(X_val)
    X_test_pad  = encode(X_test)

    # Save numpy arrays
    np.save(proc_dir / "X_train.npy", X_train_pad)
    np.save(proc_dir / "X_val.npy",   X_val_pad)
    np.save(proc_dir / "X_test.npy",  X_test_pad)
    np.save(proc_dir / "y_train.npy", y_train)
    np.save(proc_dir / "y_val.npy",   y_val)
    np.save(proc_dir / "y_test.npy",  y_test)

    # Save tokenizer as JSON
    json_path = proc_dir / "tokenizer.json"
    json_path.write_text(tokenizer.to_json())
    logger.info(f"Tokenizer saved to {json_path}")

    # Save tokenizer as pickle for compatibility
    tok_path = PROJECT_ROOT / proc_cfg["tokenizer_save_path"]
    with open(tok_path, "wb") as f:
        pickle.dump(tokenizer, f)

    # Save feature metadata
    meta = {
        "vocab_size":          vocab_size,
        "max_sequence_length": max_len,
        "train_samples":       len(X_train),
        "val_samples":         len(X_val),
        "test_samples":        len(X_test),
    }
    with open(proc_dir / "feature_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Feature metadata: {meta}")
    logger.info("✅ Featurization complete")


if __name__ == "__main__":
    featurize()