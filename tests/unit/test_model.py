"""
tests/unit/test_model.py
-------------------------
Unit tests for model building and training utilities.
Run: pytest tests/unit/test_model.py -v
"""

import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture
def dummy_cfg():
    return {
        "model": {
            "embedding_dim": 16,
            "lstm_units": 8,
            "dense_units": 16,
            "dropout": 0.1,
            "recurrent_dropout": 0.0,
            "output_activation": "sigmoid",
            "loss": "binary_crossentropy",
            "optimizer": "adam",
            "metrics": ["accuracy"],
            "batch_size": 4,
            "epochs": 1,
            "early_stopping_patience": 1,
            "model_save_path": "/tmp/test_model",
        },
        "preprocessing": {
            "max_sequence_length": 10,
            "tokenizer_save_path": "/tmp/test_tokenizer.pkl",
            "test_size": 0.2,
            "val_size": 0.1,
            "random_state": 42,
        },
    }


class TestBuildModel:
    def test_model_compiles(self, dummy_cfg):
        from src.models.train import build_model
        model = build_model(vocab_size=500, cfg=dummy_cfg)
        assert model is not None

    def test_model_output_shape(self, dummy_cfg):
        from src.models.train import build_model
        model = build_model(vocab_size=500, cfg=dummy_cfg)
        batch = np.zeros((4, 10), dtype=np.int32)
        out = model.predict(batch, verbose=0)
        assert out.shape == (4, 1)

    def test_model_output_sigmoid_range(self, dummy_cfg):
        from src.models.train import build_model
        model = build_model(vocab_size=500, cfg=dummy_cfg)
        batch = np.random.randint(0, 500, (8, 10))
        out = model.predict(batch, verbose=0).flatten()
        assert all(0.0 <= p <= 1.0 for p in out)

    def test_model_has_correct_layers(self, dummy_cfg):
        from src.models.train import build_model
        import tensorflow as tf
        model = build_model(vocab_size=500, cfg=dummy_cfg)
        layer_types = [type(l).__name__ for l in model.layers]
        assert "Embedding" in layer_types
        assert "Bidirectional" in layer_types
        assert "Dense" in layer_types

    def test_model_trainable(self, dummy_cfg):
        from src.models.train import build_model
        model = build_model(vocab_size=100, cfg=dummy_cfg)
        X = np.random.randint(0, 100, (8, 10))
        y = np.random.randint(0, 2, (8,))
        # Should not raise
        model.fit(X, y, epochs=1, verbose=0)
