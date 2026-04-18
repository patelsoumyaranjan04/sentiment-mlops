"""
tests/unit/test_data_pipeline.py
---------------------------------
Unit tests for ingest and preprocess modules.
Run: pytest tests/unit/test_data_pipeline.py -v
"""

import sys
import json
import pickle
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.preprocess import clean_text, preprocess_text, compute_baseline_stats


# ------------------------------------------------------------------ #
#  Tests: clean_text                                                   #
# ------------------------------------------------------------------ #

class TestCleanText:
    def test_removes_urls(self):
        assert "http" not in clean_text("visit http://example.com now")

    def test_lowercases(self):
        result = clean_text("Hello World")
        assert result == result.lower()

    def test_expands_wont(self):
        assert "will not" in clean_text("I won't do it")

    def test_expands_cant(self):
        assert "can not" in clean_text("I can't stop")

    def test_removes_special_chars(self):
        result = clean_text("hello!!! world???")
        assert "!" not in result and "?" not in result


# ------------------------------------------------------------------ #
#  Tests: preprocess_text                                              #
# ------------------------------------------------------------------ #

class TestPreprocessText:
    def test_removes_stopwords(self):
        result = preprocess_text("this is a great product")
        # stopwords like 'this', 'is', 'a' should be removed
        assert "this" not in result.split()
        assert "is" not in result.split()

    def test_lemmatizes(self):
        result = preprocess_text("running dogs are playing")
        # 'dogs' -> 'dog', 'running' -> various, but basic check
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_string(self):
        assert isinstance(preprocess_text("some review text"), str)

    def test_empty_string(self):
        result = preprocess_text("")
        assert isinstance(result, str)


# ------------------------------------------------------------------ #
#  Tests: compute_baseline_stats                                       #
# ------------------------------------------------------------------ #

class TestBaselineStats:
    def setup_method(self):
        self.texts = [
            "great product love it",
            "terrible quality waste of money",
            "okay nothing special",
            "amazing fast delivery",
            "bad packaging broken item",
        ]

    def test_returns_dict(self):
        stats = compute_baseline_stats(self.texts)
        assert isinstance(stats, dict)

    def test_required_keys(self):
        stats = compute_baseline_stats(self.texts)
        for key in ["mean_length", "std_length", "min_length", "max_length",
                    "vocab_size", "total_samples"]:
            assert key in stats, f"Missing key: {key}"

    def test_total_samples(self):
        stats = compute_baseline_stats(self.texts)
        assert stats["total_samples"] == len(self.texts)

    def test_mean_length_positive(self):
        stats = compute_baseline_stats(self.texts)
        assert stats["mean_length"] > 0

    def test_min_leq_max(self):
        stats = compute_baseline_stats(self.texts)
        assert stats["min_length"] <= stats["max_length"]


# ------------------------------------------------------------------ #
#  Tests: ingest validation                                            #
# ------------------------------------------------------------------ #

class TestIngestValidation:
    def test_raises_on_missing_column(self):
        from src.data.ingest import validate_schema
        df_bad = pd.DataFrame({"review": ["text"], "label": [1]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df_bad)

    def test_raises_on_invalid_sentiment(self):
        from src.data.ingest import validate_schema
        df_bad = pd.DataFrame({"review": ["text"], "sentiment": [5]})
        with pytest.raises(ValueError, match="Unexpected sentiment values"):
            validate_schema(df_bad)

    def test_passes_valid_df(self):
        from src.data.ingest import validate_schema
        df_good = pd.DataFrame({"review": ["text"], "sentiment": [1]})
        validate_schema(df_good)  # Should not raise
