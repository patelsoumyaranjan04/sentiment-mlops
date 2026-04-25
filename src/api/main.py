"""
src/api/main.py
----------------
FastAPI inference server for Amazon Sentiment BiLSTM.

Endpoints:
  GET  /health        — liveness check
  GET  /ready         — readiness check (model loaded?)
  POST /predict       — single review prediction
  POST /predict/batch — batch predictions
  GET  /drift         — current drift detection status
  GET  /metrics       — Prometheus metrics

Run locally:
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conlist
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
    MODEL_LOADED_GAUGE,
    DRIFT_DETECTED_GAUGE,
    DRIFT_KS_STATISTIC,
    DRIFT_P_VALUE,
    DRIFT_MEAN_SHIFT,
    DRIFT_WINDOW_MEAN,
)
from src.monitoring.drift_detector import DriftDetector

logger = get_logger("api")
cfg    = load_config()

# Global state
_state: dict = {
    "model":     None,
    "tokenizer": None,
    "ready":     False,
    "threshold": 0.5,
    "drift":     None,  # DriftDetector instance
}


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
        MODEL_LOADED_GAUGE.set(1)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        MODEL_LOADED_GAUGE.set(0)

    # Load optimal threshold
    proc_dir       = Path(cfg["data"]["processed_dir"])
    threshold_path = proc_dir / "optimal_threshold.json"
    if threshold_path.exists():
        _state["threshold"] = json.loads(threshold_path.read_text())["threshold"]
        logger.info(f"Loaded optimal threshold: {_state['threshold']:.4f}")
    else:
        logger.info("Using default threshold: 0.5")

    # Initialize drift detector
    baseline_path = proc_dir / "baseline_stats.json"
    _state["drift"] = DriftDetector(
        baseline_path=baseline_path,
        window_size=500,
        ks_threshold=0.05,
    )
    logger.info("Drift detector initialized.")

    yield
    logger.info("Shutting down API.")
    MODEL_LOADED_GAUGE.set(0)


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

Instrumentator().instrument(app).expose(app)


# ------------------------------------------------------------------ #
#  Schemas                                                             #
# ------------------------------------------------------------------ #

class PredictRequest(BaseModel):
    review: Annotated[str, Field(
        min_length=1,
        max_length=5000,
        example="This product is absolutely amazing!"
    )]


class PredictResponse(BaseModel):
    review:     str
    sentiment:  str
    label:      int
    confidence: float
    latency_ms: float
    threshold:  float


class BatchPredictRequest(BaseModel):
    reviews: conlist(str, min_items=1, max_items=50)

class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    total:       int
    latency_ms:  float


class DriftResponse(BaseModel):
    drift_detected: bool
    ks_statistic:   float | None
    p_value:        float | None
    window_size:    int
    window_mean:    float | None
    baseline_mean:  float | None
    mean_shift:     float | None
    message:        str


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _preprocess_single(text: str, tokenizer, max_len: int) -> np.ndarray:
    import re
    from src.data.preprocess import pad_sequences
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text).lower()
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    return padded


def _run_inference(texts: list[str]) -> list[dict]:
    if not texts:
        raise ValueError("No texts provided for inference")
    model     = _state["model"]
    tokenizer = _state["tokenizer"]
    threshold = _state["threshold"]
    max_len   = cfg["preprocessing"]["max_sequence_length"]
    detector  = _state["drift"]

    all_padded = []
    lengths    = []
    for t in texts:
        padded = _preprocess_single(t, tokenizer, max_len)
        all_padded.append(padded)
        word_count = len(t.split())
        lengths.append(word_count)

        # Update drift detector window
        if detector:
            detector.update(t)

    X     = np.vstack(all_padded)
    probs = model.predict(X, verbose=0).flatten()

    # Update drift metrics every 50 predictions
    if detector and len(detector.window) % 50 == 0:
        drift_result = detector.check_drift()
        DRIFT_DETECTED_GAUGE.set(1 if drift_result["drift_detected"] else 0)
        if drift_result["ks_statistic"] is not None:
            DRIFT_KS_STATISTIC.set(drift_result["ks_statistic"])
            DRIFT_P_VALUE.set(drift_result["p_value"])
        if drift_result["mean_shift"] is not None:
            DRIFT_MEAN_SHIFT.set(drift_result["mean_shift"])
        if drift_result["window_mean"] is not None:
            DRIFT_WINDOW_MEAN.set(drift_result["window_mean"])
        if drift_result["drift_detected"]:
            logger.warning(f"Drift alert: {drift_result['message']}")

    results = []
    for i, (text, prob) in enumerate(zip(texts, probs)):
        label     = int(prob >= threshold)
        sentiment = "positive" if label == 1 else "negative"

        PREDICTION_COUNTER.labels(sentiment=sentiment).inc()
        INPUT_LENGTH_HISTOGRAM.observe(lengths[i])

        results.append({
            "review":     text,
            "sentiment":  sentiment,
            "label":      label,
            "confidence": round(float(prob), 4),
            "threshold":  round(threshold, 4),
        })

    pos_count = sum(1 for r in results if r["label"] == 1)
    POSITIVE_RATIO_GAUGE.set(pos_count / len(results))

    return results


# ------------------------------------------------------------------ #
#  Routes                                                              #
# ------------------------------------------------------------------ #

@app.get("/health", tags=["Health"])
def health():
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/ready", tags=["Health"])
def ready():
    """Readiness probe."""
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
    latency     = round((time.perf_counter() - t0) * 1000, 2)
    PREDICTION_LATENCY.observe(latency / 1000)
    predictions = [PredictResponse(**r, latency_ms=latency) for r in results]
    return BatchPredictResponse(
        predictions=predictions,
        total=len(predictions),
        latency_ms=latency,
    )


@app.get("/drift", response_model=DriftResponse, tags=["Monitoring"])
def drift_status():
    """
    Returns the current data drift detection status.
    Compares the rolling window of incoming requests
    against the training baseline using the KS test.
    """
    detector = _state.get("drift")
    if detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")
    
    result = detector.check_drift()
    
    return DriftResponse(
        drift_detected=result.get("drift_detected", False),
        ks_statistic=result.get("ks_statistic"),
        p_value=result.get("p_value"),
        window_size=result.get("window_size", 0),
        window_mean=result.get("window_mean"),
        baseline_mean=result.get("baseline_mean"),
        mean_shift=result.get("mean_shift"),
        message=result.get("message", ""),
    )