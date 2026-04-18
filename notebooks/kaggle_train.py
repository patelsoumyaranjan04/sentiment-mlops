"""
notebooks/kaggle_train.py
--------------------------
Self-contained training script for Kaggle GPU notebooks.
Paste this entire file into a Kaggle Code cell.

After running:
  1. Download from /kaggle/working/:
       - bilstm_model/        (SavedModel directory)
       - tokenizer.pkl
       - mlruns/              (MLflow tracking data)
       - classification_report.txt
  2. Place bilstm_model/ and tokenizer.pkl in data/processed/
  3. Place mlruns/ in project root

Dataset required on Kaggle:
  https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews
  (Add as input dataset, path: /kaggle/input/amazonn-reviews/Amazon_review.csv)
"""

# ============================================================
# CELL 1 — Install dependencies
# ============================================================
# !pip install mlflow loguru emoji nltk -q

# ============================================================
# CELL 2 — Imports & Config
# ============================================================
import os, re, pickle, json
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, f1_score, precision_score, recall_score,
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature

for pkg in ["stopwords", "wordnet", "punkt", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

# ---- Hyperparameters (mirrors configs/config.yaml) ----
CFG = {
    "raw_file":               "/kaggle/input/amazonn-reviews/Amazon_review.csv",
    "output_dir":             "/kaggle/working",
    "test_size":              0.20,
    "val_size":               0.10,
    "random_state":           42,
    "max_sequence_length":    64,
    "embedding_dim":          64,
    "lstm_units":             100,
    "dense_units":            128,
    "dropout":                0.3,
    "batch_size":             64,
    "epochs":                 10,
    "early_stopping_patience":5,
    "optimizer":              "adam",
    "loss":                   "binary_crossentropy",
    "experiment_name":        "amazon-sentiment-bilstm",
    "registered_model_name":  "SentimentBiLSTM",
}

# ============================================================
# CELL 3 — Preprocessing
# ============================================================
lemmatizer = WordNetLemmatizer()
STOP_WORDS  = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"@", " at ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return text.lower()

def preprocess_text(text):
    text = clean_text(text)
    tokens = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in STOP_WORDS and len(w) > 1
    ]
    return " ".join(tokens)

print("Loading data ...")
df = pd.read_csv(CFG["raw_file"])[["review", "sentiment"]].dropna().drop_duplicates()
df["sentiment"] = df["sentiment"].astype(int)
print(f"Rows: {len(df)}, Positive: {(df.sentiment==1).sum()}, Negative: {(df.sentiment==0).sum()}")

print("Preprocessing text (this takes a few minutes) ...")
df["clean_review"] = df["review"].apply(preprocess_text)
df = df[df["clean_review"].str.strip().str.len() > 0].reset_index(drop=True)
print(f"After cleaning: {len(df)} rows")

# Splits
X, y = df["clean_review"].values, df["sentiment"].values
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=CFG["test_size"], random_state=CFG["random_state"], stratify=y
)
val_rel = CFG["val_size"] / (1 - CFG["test_size"])
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_rel, random_state=CFG["random_state"], stratify=y_temp
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Tokenize & pad
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocab size: {vocab_size}")

def encode(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=CFG["max_sequence_length"],
                         padding="post", truncating="post")

X_train_pad = encode(X_train)
X_val_pad   = encode(X_val)
X_test_pad  = encode(X_test)

# Save tokenizer
tok_path = os.path.join(CFG["output_dir"], "tokenizer.pkl")
with open(tok_path, "wb") as f:
    pickle.dump(tokenizer, f)
print(f"Tokenizer saved to {tok_path}")

# Baseline stats
lengths = [len(t.split()) for t in X_train]
baseline = {
    "mean_length":  float(np.mean(lengths)),
    "std_length":   float(np.std(lengths)),
    "min_length":   int(np.min(lengths)),
    "max_length":   int(np.max(lengths)),
    "vocab_size":   vocab_size,
    "total_samples":len(X_train),
}
with open(os.path.join(CFG["output_dir"], "baseline_stats.json"), "w") as f:
    json.dump(baseline, f, indent=2)
print("Baseline stats:", baseline)

# ============================================================
# CELL 4 — Model + MLflow Training
# ============================================================
mlflow.set_tracking_uri(f"file://{CFG['output_dir']}/mlruns")
mlflow.set_experiment(CFG["experiment_name"])

with mlflow.start_run() as run:
    print(f"MLflow run_id: {run.info.run_id}")
    mlflow.log_params({**CFG, "vocab_size": vocab_size})

    # Build model
    model = Sequential([
        Embedding(vocab_size, CFG["embedding_dim"]),
        Bidirectional(LSTM(CFG["lstm_units"], dropout=0, recurrent_dropout=0)),
        Dense(CFG["dense_units"], activation="relu"),
        Dropout(CFG["dropout"]),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=CFG["optimizer"], loss=CFG["loss"], metrics=["accuracy"])
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=CFG["early_stopping_patience"],
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(CFG["output_dir"], "best_model.h5"),
                        monitor="val_loss", save_best_only=True, verbose=1),
    ]

    # Train
    history = model.fit(
        X_train_pad, y_train,
        validation_data=(X_val_pad, y_val),
        epochs=CFG["epochs"],
        batch_size=CFG["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    # Per-epoch metrics
    for epoch, vals in enumerate(zip(
        history.history["loss"], history.history["accuracy"],
        history.history["val_loss"], history.history["val_accuracy"],
    )):
        mlflow.log_metrics(dict(zip(
            ["train_loss","train_accuracy","val_loss","val_accuracy"], vals
        )), step=epoch)

    # Test evaluation
    y_prob = model.predict(X_test_pad, batch_size=CFG["batch_size"]).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    test_metrics = {
        "test_accuracy":  float(accuracy_score(y_test, y_pred)),
        "test_f1":        float(f1_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall":    float(recall_score(y_test, y_pred)),
        "test_roc_auc":   float(roc_auc_score(y_test, y_prob)),
    }
    mlflow.log_metrics(test_metrics)

    report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"])
    print(report)
    mlflow.log_text(report, "classification_report.txt")
    with open(os.path.join(CFG["output_dir"], "classification_report.txt"), "w") as f:
        f.write(report)

    # Save & register model
    model_path = os.path.join(CFG["output_dir"], "bilstm_model")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    sample_in  = X_test_pad[:5]
    sample_out = model.predict(sample_in)
    sig = infer_signature(sample_in, sample_out)
    mlflow.keras.log_model(
        model, artifact_path="model",
        signature=sig,
        registered_model_name=CFG["registered_model_name"],
    )
    mlflow.log_artifact(tok_path, artifact_path="tokenizer")
    mlflow.log_artifact(
        os.path.join(CFG["output_dir"], "baseline_stats.json"),
        artifact_path="data_stats"
    )

    print("\n===== FINAL METRICS =====")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"MLflow run_id: {run.info.run_id}")
    print("Download bilstm_model/, tokenizer.pkl, mlruns/ from /kaggle/working/")
