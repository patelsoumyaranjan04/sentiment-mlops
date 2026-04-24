"""
src/monitoring/drift_detector.py
---------------------------------
Statistical drift detection using Kolmogorov-Smirnov test.
Compares incoming request distributions against baseline stats
computed during preprocessing.

Used by the FastAPI inference server to detect data drift
in real time and expose results via Prometheus metrics.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from collections import deque

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

logger = get_logger("drift_detector")


class DriftDetector:
    """
    Detects data drift by comparing a rolling window of incoming
    request statistics against the training baseline.

    Uses the Kolmogorov-Smirnov (KS) test for distribution comparison.
    A p-value below the threshold indicates drift.
    """

    def __init__(
        self,
        baseline_path: str | Path,
        window_size: int = 500,
        ks_threshold: float = 0.05,
    ):
        self.window_size  = window_size
        self.ks_threshold = ks_threshold
        self.window       = deque(maxlen=window_size)
        self.drift_count  = 0
        self.total_checks = 0

        # Load baseline statistics
        self.baseline = self._load_baseline(baseline_path)
        logger.info(
            f"DriftDetector initialized — "
            f"window={window_size}, threshold={ks_threshold}"
        )

    def _load_baseline(self, path: str | Path) -> dict:
        path = Path(path)
        if not path.exists():
            logger.warning(f"Baseline stats not found at {path}. Drift detection disabled.")
            return {}
        with open(path) as f:
            baseline = json.load(f)
        logger.info(f"Baseline loaded: mean_length={baseline.get('mean_length'):.2f}, "
                   f"std={baseline.get('std_length'):.2f}")
        return baseline

    def update(self, text: str) -> None:
        """Add a new incoming review to the rolling window."""
        word_count = len(text.split())
        self.window.append(word_count)

    def check_drift(self) -> dict:
        """
        Run KS test comparing rolling window against baseline.
        Returns drift report dict.
        Only runs when window has enough samples.
        """
        result = {
            "drift_detected": False,
            "ks_statistic":   None,
            "p_value":        None,
            "window_size":    len(self.window),
            "window_mean":    None,
            "baseline_mean":  self.baseline.get("mean_length"),
            "message":        "Insufficient samples",
        }

        if not self.baseline:
            result["message"] = "No baseline available"
            return result

        if len(self.window) < self.window_size:
            result["message"] = (
                f"Building window: {len(self.window)}/{self.window_size} samples"
            )
            return result

        # Generate baseline distribution from stored stats
        baseline_mean = self.baseline["mean_length"]
        baseline_std  = self.baseline["std_length"]
        baseline_samples = np.random.normal(
            loc=baseline_mean,
            scale=baseline_std,
            size=self.window_size
        ).clip(min=1)

        window_array = np.array(self.window)
        window_mean  = float(np.mean(window_array))

        # KS test
        ks_stat, p_value = stats.ks_2samp(window_array, baseline_samples)

        drift_detected = p_value < self.ks_threshold
        self.total_checks += 1
        if drift_detected:
            self.drift_count += 1

        result.update({
            "drift_detected": drift_detected,
            "ks_statistic":   round(float(ks_stat), 4),
            "p_value":        round(float(p_value), 4),
            "window_mean":    round(window_mean, 2),
            "baseline_mean":  round(baseline_mean, 2),
            "mean_shift":     round(window_mean - baseline_mean, 2),
            "message": (
                f"DRIFT DETECTED (p={p_value:.4f} < {self.ks_threshold})"
                if drift_detected
                else f"No drift (p={p_value:.4f} >= {self.ks_threshold})"
            ),
        })

        if drift_detected:
            logger.warning(
                f"DATA DRIFT DETECTED — "
                f"KS={ks_stat:.4f}, p={p_value:.4f}, "
                f"window_mean={window_mean:.2f}, "
                f"baseline_mean={baseline_mean:.2f}"
            )

        return result

    @property
    def drift_rate(self) -> float:
        if self.total_checks == 0:
            return 0.0
        return self.drift_count / self.total_checks