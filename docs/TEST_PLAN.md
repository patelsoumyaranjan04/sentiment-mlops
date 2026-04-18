# Test Plan & Test Cases
## Amazon Reviews Sentiment Analysis — MLOps Pipeline
**Version:** 1.0.0 | **Course:** DA6401

---

## 1. Test Plan

### 1.1 Scope
This test plan covers the data pipeline, ML model, inference API, and frontend of the Amazon Sentiment MLOps project.

### 1.2 Acceptance Criteria
| Component | Criterion |
|---|---|
| Data Pipeline | All unit tests pass; validated CSV has 0 nulls, 0 duplicates |
| Model | Test accuracy ≥ 88%, ROC-AUC ≥ 0.88 |
| API | All integration tests pass; `/predict` p95 latency < 200ms |
| Frontend | App loads without errors; both pages render; predictions display correctly |
| Docker | All 5 services start healthy within 60 seconds |

### 1.3 Test Types
- **Unit tests** — individual function behavior, isolated with mocks
- **Integration tests** — API endpoints end-to-end (in-process via TestClient)
- **Manual tests** — UI walkthrough, Docker Compose startup, Grafana dashboard

### 1.4 Test Environment
- OS: Ubuntu 22.04 (VirtualBox)
- Python: 3.10 (conda: sentiment-mlops)
- Run command: `pytest tests/ -v --tb=short`

---

## 2. Test Cases

### 2.1 Data Pipeline Unit Tests (`tests/unit/test_data_pipeline.py`)

| TC ID | Test Name | Description | Expected Result | Status |
|---|---|---|---|---|
| DP-01 | `test_removes_urls` | `clean_text` strips URLs | "http" absent in output | PASS |
| DP-02 | `test_lowercases` | `clean_text` lowercases all | output == output.lower() | PASS |
| DP-03 | `test_expands_wont` | `clean_text` expands "won't" | "will not" in output | PASS |
| DP-04 | `test_expands_cant` | `clean_text` expands "can't" | "can not" in output | PASS |
| DP-05 | `test_removes_special_chars` | `clean_text` strips `!?` | no `!` or `?` in output | PASS |
| DP-06 | `test_removes_stopwords` | `preprocess_text` strips "this", "is", "a" | stopwords absent | PASS |
| DP-07 | `test_lemmatizes` | `preprocess_text` returns string | isinstance(result, str) | PASS |
| DP-08 | `test_returns_string` | `preprocess_text` always returns str | isinstance always true | PASS |
| DP-09 | `test_empty_string` | `preprocess_text("")` handles empty input | returns str without error | PASS |
| DP-10 | `test_returns_dict` | `compute_baseline_stats` returns dict | isinstance(stats, dict) | PASS |
| DP-11 | `test_required_keys` | baseline stats has all 6 keys | all keys present | PASS |
| DP-12 | `test_total_samples` | baseline stats total_samples correct | equals len(input) | PASS |
| DP-13 | `test_mean_length_positive` | mean_length > 0 | mean_length > 0 | PASS |
| DP-14 | `test_min_leq_max` | min_length ≤ max_length | logical constraint | PASS |
| DP-15 | `test_raises_on_missing_column` | `validate_schema` raises on missing col | ValueError raised | PASS |
| DP-16 | `test_raises_on_invalid_sentiment` | `validate_schema` rejects value 5 | ValueError raised | PASS |
| DP-17 | `test_passes_valid_df` | `validate_schema` accepts valid df | no exception raised | PASS |

### 2.2 Model Unit Tests (`tests/unit/test_model.py`)

| TC ID | Test Name | Description | Expected Result | Status |
|---|---|---|---|---|
| ML-01 | `test_model_compiles` | `build_model` returns compiled model | model is not None | PASS |
| ML-02 | `test_model_output_shape` | predict on (4, 10) → (4, 1) | shape == (4, 1) | PASS |
| ML-03 | `test_model_output_sigmoid_range` | all outputs in [0, 1] | all 0.0 ≤ p ≤ 1.0 | PASS |
| ML-04 | `test_model_has_correct_layers` | has Embedding, Bidirectional, Dense | layer names match | PASS |
| ML-05 | `test_model_trainable` | model.fit runs without error | no exception | PASS |

### 2.3 API Integration Tests (`tests/integration/test_api.py`)

| TC ID | Test Name | Description | Expected Result | Status |
|---|---|---|---|---|
| API-01 | `test_health_returns_200` | GET /health → 200 | status_code == 200 | PASS |
| API-02 | `test_health_body` | GET /health body | `{"status": "ok"}` | PASS |
| API-03 | `test_ready_when_model_loaded` | GET /ready when ready | `{"status": "ready"}` | PASS |
| API-04 | `test_predict_returns_200` | POST /predict → 200 | status_code == 200 | PASS |
| API-05 | `test_predict_response_schema` | response has all required fields | all keys present | PASS |
| API-06 | `test_predict_sentiment_values` | sentiment is positive/negative | in allowed set | PASS |
| API-07 | `test_predict_confidence_range` | confidence in [0.0, 1.0] | constraint holds | PASS |
| API-08 | `test_predict_empty_review_rejected` | empty review → 422 | status_code == 422 | PASS |
| API-09 | `test_predict_missing_field_rejected` | no review field → 422 | status_code == 422 | PASS |
| API-10 | `test_predict_latency_is_positive` | latency_ms > 0 | constraint holds | PASS |
| API-11 | `test_batch_returns_200` | POST /predict/batch → 200 | status_code == 200 | PASS |
| API-12 | `test_batch_response_count` | 3 reviews → total == 3 | total and len match | PASS |
| API-13 | `test_batch_empty_list_rejected` | empty reviews list → 422 | status_code == 422 | PASS |
| API-14 | `test_batch_all_have_sentiment` | each prediction has sentiment | all in allowed set | PASS |

---

## 3. Test Report Template

Fill this in after running `pytest tests/ -v --tb=short > test_report.txt`:

```
Test Run Date  : _______________
Environment    : Ubuntu 22.04, Python 3.10, conda sentiment-mlops
Command        : pytest tests/ -v --tb=short

Total Test Cases : 26
Passed           : ___
Failed           : ___
Skipped          : ___

Pass Rate        : ____%

Failed Tests (if any):
  - TC ID: ___  | Reason: ___
```

---

## 4. Manual Test Checklist

### 4.1 Docker Compose Startup
- [ ] `docker compose up --build` completes without errors
- [ ] All 5 containers are healthy: `docker compose ps`
- [ ] Frontend accessible at http://localhost:3000
- [ ] API docs accessible at http://localhost:8000/docs
- [ ] MLflow UI accessible at http://localhost:5000
- [ ] Prometheus accessible at http://localhost:9090
- [ ] Grafana accessible at http://localhost:3001 (admin/admin)

### 4.2 Frontend UI
- [ ] Analyze page loads; textarea visible
- [ ] Example chips populate the textarea when clicked
- [ ] Single predict returns a result card with sentiment, confidence bar, latency
- [ ] Batch mode: multiple reviews produce per-review results and summary stats
- [ ] History panel accumulates predictions and shows correct counts
- [ ] Pipeline page loads; API status indicators show green when backend is up
- [ ] Links to MLflow, Grafana, Airflow are visible

### 4.3 Grafana Dashboard
- [ ] "Sentiment API Monitoring" dashboard loads automatically
- [ ] All panels show data after sending a few requests
- [ ] Prediction counter increases with each request
- [ ] Latency histogram shows p50/p95/p99 lines
