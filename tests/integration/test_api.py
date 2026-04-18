"""
tests/integration/test_api.py
------------------------------
Integration tests for the FastAPI inference server.
Requires the backend to be running on localhost:8000.
OR uses TestClient for in-process testing (preferred for CI).

Run:
    pytest tests/integration/test_api.py -v
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ------------------------------------------------------------------ #
#  Fixtures: mock model + tokenizer so we don't need real artifacts   #
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def mock_model():
    model = MagicMock()
    # Simulate model.predict returning probabilities
    model.predict.return_value = np.array([[0.85], [0.12], [0.92]])
    return model


@pytest.fixture(scope="module")
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.word_index = {f"word{i}": i for i in range(1000)}
    tokenizer.texts_to_sequences.return_value = [[1, 2, 3, 4, 5]]
    return tokenizer


@pytest.fixture(scope="module")
def client(mock_model, mock_tokenizer):
    """Create a FastAPI TestClient with mocked model loading."""
    with patch("src.api.model_loader.load_model_and_tokenizer",
               return_value=(mock_model, mock_tokenizer)):
        from fastapi.testclient import TestClient
        from src.api.main import app, _state
        _state["model"]     = mock_model
        _state["tokenizer"] = mock_tokenizer
        _state["ready"]     = True
        return TestClient(app)


# ------------------------------------------------------------------ #
#  Health & Readiness                                                  #
# ------------------------------------------------------------------ #

class TestHealthEndpoints:
    def test_health_returns_200(self, client):
        res = client.get("/health")
        assert res.status_code == 200

    def test_health_body(self, client):
        res = client.get("/health")
        assert res.json() == {"status": "ok"}

    def test_ready_when_model_loaded(self, client):
        res = client.get("/ready")
        assert res.status_code == 200
        assert res.json()["status"] == "ready"


# ------------------------------------------------------------------ #
#  Single Prediction                                                   #
# ------------------------------------------------------------------ #

class TestSinglePredict:
    def test_predict_returns_200(self, client):
        res = client.post("/predict", json={"review": "This product is amazing!"})
        assert res.status_code == 200

    def test_predict_response_schema(self, client):
        res = client.post("/predict", json={"review": "Great quality item."})
        body = res.json()
        assert "sentiment" in body
        assert "label" in body
        assert "confidence" in body
        assert "latency_ms" in body

    def test_predict_sentiment_values(self, client):
        res = client.post("/predict", json={"review": "Excellent product!"})
        body = res.json()
        assert body["sentiment"] in ("positive", "negative")
        assert body["label"] in (0, 1)

    def test_predict_confidence_range(self, client):
        res = client.post("/predict", json={"review": "Good but not great."})
        body = res.json()
        assert 0.0 <= body["confidence"] <= 1.0

    def test_predict_empty_review_rejected(self, client):
        res = client.post("/predict", json={"review": ""})
        assert res.status_code == 422

    def test_predict_missing_field_rejected(self, client):
        res = client.post("/predict", json={})
        assert res.status_code == 422

    def test_predict_latency_is_positive(self, client):
        res = client.post("/predict", json={"review": "Fast delivery, happy customer."})
        assert res.json()["latency_ms"] > 0


# ------------------------------------------------------------------ #
#  Batch Prediction                                                    #
# ------------------------------------------------------------------ #

class TestBatchPredict:
    def test_batch_returns_200(self, client, mock_model):
        mock_model.predict.return_value = np.array([[0.9], [0.1]])
        res = client.post("/predict/batch", json={
            "reviews": ["Amazing!", "Terrible!"]
        })
        assert res.status_code == 200

    def test_batch_response_count(self, client, mock_model):
        mock_model.predict.return_value = np.array([[0.9], [0.1], [0.7]])
        res = client.post("/predict/batch", json={
            "reviews": ["Great!", "Awful!", "Okay"]
        })
        body = res.json()
        assert body["total"] == 3
        assert len(body["predictions"]) == 3

    def test_batch_empty_list_rejected(self, client):
        res = client.post("/predict/batch", json={"reviews": []})
        assert res.status_code == 422

    def test_batch_all_have_sentiment(self, client, mock_model):
        mock_model.predict.return_value = np.array([[0.8], [0.2]])
        res = client.post("/predict/batch", json={
            "reviews": ["Good product", "Broken item"]
        })
        for pred in res.json()["predictions"]:
            assert pred["sentiment"] in ("positive", "negative")
