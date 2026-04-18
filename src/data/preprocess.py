"""
src/data/preprocess.py
----------------------
Text cleaning, lemmatization, train/val/test splitting,
tokenizer fitting, sequence padding, and baseline drift stats.

Run:
    python -m src.data.preprocess
"""

import sys
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("preprocess")

# Download required NLTK data
for pkg in ["stopwords", "wordnet", "punkt", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))


# ------------------------------------------------------------------ #
#  Text cleaning helpers                                               #
# ------------------------------------------------------------------ #

def clean_text(text: str) -> str:
    """Basic normalization: handle contractions, symbols, lowercase."""
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"@", " at ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = text.lower()
    return text


def preprocess_text(text: str) -> str:
    """Clean + remove stopwords + lemmatize."""
    text = clean_text(text)
    tokens = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in STOP_WORDS and len(w) > 1
    ]
    return " ".join(tokens)


# ------------------------------------------------------------------ #
#  Baseline statistics for drift detection                            #
# ------------------------------------------------------------------ #

def compute_baseline_stats(texts: list[str]) -> dict:
    """
    Compute statistical baseline (mean/std of text length, vocab size)
    to be stored for later drift detection comparison.
    """
    lengths = [len(t.split()) for t in texts]
    stats = {
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "vocab_size": int(len(set(" ".join(texts).split()))),
        "total_samples": len(texts),
    }
    logger.info("Baseline drift stats computed:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    return stats


# ------------------------------------------------------------------ #
#  Main pipeline                                                       #
# ------------------------------------------------------------------ #

def preprocess(cfg: dict | None = None) -> None:
    if cfg is None:
        cfg = load_config()

    proc_cfg = cfg["preprocessing"]
    data_cfg = cfg["data"]

    # ---- Load validated raw data ----
    raw_path = Path(data_cfg["raw_dir"]) / "Amazon_review_validated.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Run ingest.py first. File not found: {raw_path}")

    logger.info(f"Loading validated data from {raw_path}")
    df = pd.read_csv(raw_path)

    # ---- Clean text ----
    logger.info("Cleaning and preprocessing text ...")
    df["clean_review"] = df["review"].apply(preprocess_text)

    # Remove empty strings after cleaning
    df = df[df["clean_review"].str.strip().str.len() > 0].reset_index(drop=True)
    logger.info(f"Rows after cleaning: {len(df)}")

    # ---- Compute baseline stats (for drift detection) ----
    baseline = compute_baseline_stats(df["clean_review"].tolist())

    # ---- Train / Val / Test split ----
    test_size = proc_cfg["test_size"]
    val_size = proc_cfg["val_size"]
    seed = proc_cfg["random_state"]

    X = df["clean_review"].values
    y = df["sentiment"].values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, random_state=seed, stratify=y_temp
    )

    logger.info(f"Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # ---- Tokenizer ----
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    vocab_size = len(tokenizer.word_index) + 1
    logger.info(f"Vocabulary size: {vocab_size}")

    max_len = proc_cfg["max_sequence_length"]

    def encode(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")

    X_train_pad = encode(X_train)
    X_val_pad   = encode(X_val)
    X_test_pad  = encode(X_test)

    # ---- Save outputs ----
    proc_dir = Path(data_cfg["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Save splits as CSV (text + label, for reference)
    for split, X_s, y_s in [
        ("train", X_train, y_train),
        ("val",   X_val,   y_val),
        ("test",  X_test,  y_test),
    ]:
        pd.DataFrame({"clean_review": X_s, "sentiment": y_s}).to_csv(
            proc_dir / f"{split}.csv", index=False
        )

    # Save padded numpy arrays
    np.save(proc_dir / "X_train.npy", X_train_pad)
    np.save(proc_dir / "X_val.npy",   X_val_pad)
    np.save(proc_dir / "X_test.npy",  X_test_pad)
    np.save(proc_dir / "y_train.npy", y_train)
    np.save(proc_dir / "y_val.npy",   y_val)
    np.save(proc_dir / "y_test.npy",  y_test)

    # Save tokenizer
    tok_path = Path(proc_cfg["tokenizer_save_path"])
    tok_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tok_path, "wb") as f:
        pickle.dump(tokenizer, f)
    logger.info(f"Tokenizer saved to {tok_path}")

    # Save baseline stats
    import json
    with open(proc_dir / "baseline_stats.json", "w") as f:
        json.dump({**baseline, "vocab_size_tokenizer": vocab_size}, f, indent=2)
    logger.info(f"Baseline stats saved to {proc_dir / 'baseline_stats.json'}")

    logger.info("✅ Preprocessing complete.")


if __name__ == "__main__":
    preprocess()
