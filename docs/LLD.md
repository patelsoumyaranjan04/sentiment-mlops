# Low-Level Design (LLD)
## Amazon Reviews Sentiment Analysis — API & Component Specification
**Version:** 1.0.0 | **Course:** DA6401

---

## 1. API Endpoint Definitions

Base URL: `http://localhost:8000`
All request/response bodies are `application/json`.

---

### 1.1 GET `/health`
**Purpose:** Liveness probe — confirms the process is running.
**Auth:** None

**Response 200:**
```json
{ "status": "ok" }
```

---

### 1.2 GET `/ready`
**Purpose:** Readiness probe — confirms the model is loaded and accepting traffic.
**Auth:** None

**Response 200:**
```json
{ "status": "ready" }
```

**Response 503:**
```json
{ "detail": "Model not yet loaded" }
```

---

### 1.3 POST `/predict`
**Purpose:** Predict sentiment for a single Amazon review text.
**Auth:** None

**Request Body:**
```json
{
  "review": "string (1–5000 chars, required)"
}
```

**Response 200:**
```json
{
  "review":     "string — echoed input",
  "sentiment":  "positive | negative",
  "label":      "integer — 1 (positive) or 0 (negative)",
  "confidence": "float — probability of positive class [0.0, 1.0]",
  "latency_ms": "float — end-to-end inference time in milliseconds"
}
```

**Response 422 (Validation Error):**
```json
{
  "detail": [
    {
      "loc":  ["body", "review"],
      "msg":  "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Response 503:**
```json
{ "detail": "Model not ready" }
```

**Example curl:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "Absolutely love this product!"}'
```

**Example response:**
```json
{
  "review":     "Absolutely love this product!",
  "sentiment":  "positive",
  "label":      1,
  "confidence": 0.9423,
  "latency_ms": 18.4
}
```

---

### 1.4 POST `/predict/batch`
**Purpose:** Predict sentiment for multiple reviews in one call (max 50).
**Auth:** None

**Request Body:**
```json
{
  "reviews": ["string", "string", "..."]
}
```
Constraints: array length 1–50; each string 1–5000 chars.

**Response 200:**
```json
{
  "predictions": [
    {
      "review":     "string",
      "sentiment":  "positive | negative",
      "label":      0,
      "confidence": 0.1234,
      "latency_ms": 35.2
    }
  ],
  "total":       2,
  "latency_ms":  35.2
}
```

**Response 422:** Same structure as `/predict` validation error.

---

### 1.5 GET `/metrics`
**Purpose:** Expose Prometheus metrics for scraping.
**Auth:** None
**Content-Type:** `text/plain; version=0.0.4`

Key metrics exposed:
```
sentiment_predictions_total{sentiment="positive|negative"}
sentiment_prediction_latency_seconds_bucket{le="..."}
sentiment_positive_ratio
sentiment_input_length_words_bucket{le="..."}
http_requests_total{method, handler, status}
http_request_duration_seconds_bucket{...}
```

---

## 2. Module-Level Specifications

### 2.1 `src/data/ingest.py`

| Function | Signature | Description |
|---|---|---|
| `validate_schema` | `(df: DataFrame) -> None` | Raises `ValueError` if required columns missing or invalid sentiment values |
| `log_statistics` | `(df: DataFrame) -> dict` | Returns stats dict with row count, null counts, duplicate count, class distribution |
| `ingest` | `(cfg: dict) -> DataFrame` | Full ingestion pipeline; returns cleaned DataFrame |

### 2.2 `src/data/preprocess.py`

| Function | Signature | Description |
|---|---|---|
| `clean_text` | `(text: str) -> str` | Contraction expansion, URL removal, special char removal, lowercase |
| `preprocess_text` | `(text: str) -> str` | Applies `clean_text` + stopword removal + lemmatization |
| `compute_baseline_stats` | `(texts: list[str]) -> dict` | Returns mean/std/min/max word length, vocab size, total samples |
| `preprocess` | `(cfg: dict) -> None` | Full pipeline: clean → split → tokenize → pad → save arrays + tokenizer |

### 2.3 `src/models/train.py`

| Function | Signature | Description |
|---|---|---|
| `build_model` | `(vocab_size: int, cfg: dict) -> keras.Model` | Constructs and compiles the BiLSTM model |
| `train` | `(cfg: dict) -> None` | Full training loop with MLflow tracking; saves model and tokenizer |

### 2.4 `src/api/main.py`

| Component | Type | Description |
|---|---|---|
| `app` | `FastAPI` | Main application instance with CORS and Prometheus instrumentation |
| `lifespan` | `async context manager` | Loads model + tokenizer on startup |
| `_state` | `dict` | In-process state: `model`, `tokenizer`, `ready` flag |
| `_preprocess_single` | `(text, tokenizer, max_len) -> np.ndarray` | Tokenizes and pads a single text for inference |
| `_run_inference` | `(texts: list[str]) -> list[dict]` | Batch inference with Prometheus metric updates |

### 2.5 `src/api/model_loader.py`

| Function | Signature | Description |
|---|---|---|
| `load_model_and_tokenizer` | `(cfg: dict) -> tuple[Model, Tokenizer]` | Loads from local path; falls back to MLflow URI |

### 2.6 `src/monitoring/metrics.py`

| Metric | Type | Labels | Description |
|---|---|---|---|
| `sentiment_predictions_total` | Counter | `sentiment` | Total predictions by class |
| `sentiment_prediction_latency_seconds` | Histogram | — | Per-call inference latency |
| `sentiment_positive_ratio` | Gauge | — | Rolling positive class ratio |
| `sentiment_input_length_words` | Histogram | — | Word count of incoming texts |
| `sentiment_model_loaded` | Gauge | — | 1 when model is ready |

---

## 3. Data Schemas

### 3.1 Raw Data
| Column | Type | Values | Description |
|---|---|---|---|
| `review` | string | Any text | Amazon product review text |
| `sentiment` | int | 0 or 1 | 0 = negative, 1 = positive |

### 3.2 Processed Artifacts
| File | Shape | dtype | Description |
|---|---|---|---|
| `X_train.npy` | (N_train, 64) | int32 | Padded token sequences |
| `X_val.npy` | (N_val, 64) | int32 | Validation sequences |
| `X_test.npy` | (N_test, 64) | int32 | Test sequences |
| `y_train.npy` | (N_train,) | int32 | Binary labels |
| `tokenizer.pkl` | — | pickle | Fitted Keras Tokenizer |
| `baseline_stats.json` | — | JSON | Drift detection baseline |

---

## 4. Airflow DAG Specification

**DAG ID:** `sentiment_data_pipeline`
**Schedule:** `@daily`
**Catchup:** False

| Task ID | Operator | Upstream | Description |
|---|---|---|---|
| `data_ingest` | PythonOperator | — | Runs `src.data.ingest.ingest()` |
| `data_preprocess` | PythonOperator | `data_ingest` | Runs `src.data.preprocess.preprocess()` |
| `dvc_add` | BashOperator | `data_preprocess` | `dvc add data/processed/ data/raw/...` |
| `dvc_push` | BashOperator | `dvc_add` | `dvc push` |

---

## 5. DVC Pipeline (`dvc.yaml`)

| Stage | Command | Key Deps | Key Outputs |
|---|---|---|---|
| `ingest` | `python -m src.data.ingest` | `Amazon_review.csv`, `ingest.py` | `Amazon_review_validated.csv` |
| `preprocess` | `python -m src.data.preprocess` | validated CSV, `preprocess.py` | `X_*.npy`, `tokenizer.pkl`, `baseline_stats.json` |

---

## 6. Docker Services

| Service | Image | Internal Port | Host Port | Key Volumes |
|---|---|---|---|---|
| `backend` | Custom (Dockerfile.backend) | 8000 | 8000 | `data/processed`, `mlruns` |
| `frontend` | Custom (Dockerfile.frontend + nginx) | 80 | 3000 | — |
| `mlflow` | python:3.10-slim | 5000 | 5000 | `mlruns` |
| `prometheus` | prom/prometheus:2.47.0 | 9090 | 9090 | `prometheus.yml` |
| `grafana` | grafana/grafana:10.1.5 | 3000 | 3001 | provisioning configs |
