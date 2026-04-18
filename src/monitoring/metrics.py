"""
src/monitoring/metrics.py
--------------------------
Custom Prometheus metrics for the sentiment API.
Imported by src/api/main.py.
"""

from prometheus_client import Counter, Histogram, Gauge

# ---- Prediction counter: labelled by sentiment ----
PREDICTION_COUNTER = Counter(
    name="sentiment_predictions_total",
    documentation="Total number of sentiment predictions, labelled by sentiment.",
    labelnames=["sentiment"],
)

# ---- Inference latency histogram ----
PREDICTION_LATENCY = Histogram(
    name="sentiment_prediction_latency_seconds",
    documentation="Latency of individual inference calls in seconds.",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# ---- Positive prediction ratio (rolling gauge) ----
POSITIVE_RATIO_GAUGE = Gauge(
    name="sentiment_positive_ratio",
    documentation="Ratio of positive predictions in the last batch.",
)

# ---- Input text length histogram (drift proxy) ----
INPUT_LENGTH_HISTOGRAM = Histogram(
    name="sentiment_input_length_words",
    documentation="Word count of incoming review texts.",
    buckets=[5, 10, 20, 30, 50, 75, 100, 150, 200],
)

# ---- Model load status ----
MODEL_LOADED_GAUGE = Gauge(
    name="sentiment_model_loaded",
    documentation="1 if model is loaded and ready, 0 otherwise.",
)
