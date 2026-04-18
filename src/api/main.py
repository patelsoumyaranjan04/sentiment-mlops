"""
src/api/main.py
----------------
FastAPI inference server for Amazon Sentiment BiLSTM.

Endpoints:
  GET  /health     — liveness check
  GET  /ready      — readiness check (model loaded?)
  POST /predict    — single review prediction
  POST /predict/batch — batch predictions
  GET  /metrics    — Prometheus metrics (via instrumentator)

Run locally:
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import pickle
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.api.model_loader import load_model_and_tokenizer
from src.monitoring.metrics import (
    PREDICTION_COUNTER,
    PREDICTION_LATENCY,
    POSITIVE_RATIO_GAUGE,
    INPUT_LENGTH_HISTOGRAM,
)

logger = get_logger("api")
cfg = load_config()

# Global state
_state: dict = {"model": None, "tokenizer": None, "ready": False}


# ------------------------------------------------------------------ #
#  Lifespan — load model on startup                                    #
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model and tokenizer ...")
    try:
        model, tokenizer = load_model_and_tokenizer(cfg)
        _state["model"]     = model
        _state["tokenizer"] = tokenizer
        _state["ready"]     = True
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
    yield
    logger.info("Shutting down API.")


# ------------------------------------------------------------------ #
#  App                                                                 #
# ------------------------------------------------------------------ #

app = FastAPI(
    title="Amazon Sentiment API",
    description="Bidirectional LSTM sentiment classifier for Amazon product reviews.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus auto-instrumentation
Instrumentator().instrument(app).expose(app)


# ------------------------------------------------------------------ #
#  Schemas                                                             #
# ------------------------------------------------------------------ #

class PredictRequest(BaseModel):
    review: Annotated[str, Field(min_length=1, max_length=5000,
                                  example="This product is absolutely amazing!")]


class PredictResponse(BaseModel):
    review:     str
    sentiment:  str          # "positive" | "negative"
    label:      int          # 1 | 0
    confidence: float        # probability of positive class
    latency_ms: float


class BatchPredictRequest(BaseModel):
    reviews: Annotated[list[str], Field(min_length=1, max_length=50)]


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    total:       int
    latency_ms:  float


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _preprocess_single(text: str, tokenizer, max_len: int) -> np.ndarray:
    import re
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    # Minimal cleaning (same as training)
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text).lower()
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    return padded


def _run_inference(texts: list[str]) -> list[dict]:
    model    = _state["model"]
    tokenizer = _state["tokenizer"]
    max_len  = cfg["preprocessing"]["max_sequence_length"]

    all_padded = []
    lengths    = []
    for t in texts:
        padded = _preprocess_single(t, tokenizer, max_len)
        all_padded.append(padded)
        lengths.append(len(t.split()))

    X = np.vstack(all_padded)
    probs = model.predict(X, verbose=0).flatten()

    results = []
    for i, (text, prob) in enumerate(zip(texts, probs)):
        label     = int(prob >= 0.5)
        sentiment = "positive" if label == 1 else "negative"

        # Prometheus metrics
        PREDICTION_COUNTER.labels(sentiment=sentiment).inc()
        INPUT_LENGTH_HISTOGRAM.observe(lengths[i])

        results.append({
            "review":     text,
            "sentiment":  sentiment,
            "label":      label,
            "confidence": round(float(prob), 4),
        })

    pos_count = sum(1 for r in results if r["label"] == 1)
    POSITIVE_RATIO_GAUGE.set(pos_count / len(results))

    return results


# ------------------------------------------------------------------ #
#  Routes                                                              #
# ------------------------------------------------------------------ #

@app.get("/health", tags=["Health"])
def health():
    """Liveness probe — always returns OK if the process is running."""
    return {"status": "ok"}


@app.get("/ready", tags=["Health"])
def ready():
    """Readiness probe — returns OK only when the model is loaded."""
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="Model not yet loaded")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    """Predict sentiment for a single review."""
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="Model not ready")

    t0 = time.perf_counter()
    try:
        results = _run_inference([req.review])
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency = round((time.perf_counter() - t0) * 1000, 2)
    PREDICTION_LATENCY.observe(latency / 1000)

    return PredictResponse(**results[0], latency_ms=latency)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(req: BatchPredictRequest):
    """Predict sentiment for a batch of reviews (max 50)."""
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="Model not ready")

    t0 = time.perf_counter()
    try:
        results = _run_inference(req.reviews)
    except Exception as e:
        logger.error(f"Batch inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency = round((time.perf_counter() - t0) * 1000, 2)
    PREDICTION_LATENCY.observe(latency / 1000)

    predictions = [PredictResponse(**r, latency_ms=latency) for r in results]
    return BatchPredictResponse(
        predictions=predictions,
        total=len(predictions),
        latency_ms=latency,
    )
