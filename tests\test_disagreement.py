"""Test 12: Model disagreement calculation and conformal quantile validation isolation."""
import numpy as np
import pytest


def compute_model_disagreement(predictions: np.ndarray) -> float:
    """
    disagreement = std(log(sigma) across members) / mean(log(sigma) across members)
    If all members output identical prediction, std is 0.0.
    """
    assert len(predictions) > 1
    log_preds = np.log(np.maximum(1e-6, predictions))
    std_val = float(np.std(log_preds))
    mean_val = float(np.mean(log_preds))
    if abs(mean_val) < 1e-6:
        return 0.0
    return max(0.0, std_val / abs(mean_val))


def test_identical_predictions_zero_disagreement():
    """Verify that identical ensemble member outputs yield exactly 0.0 disagreement."""
    preds = np.array([2.5, 2.5, 2.5, 2.5])
    d = compute_model_disagreement(preds)
    assert d == pytest.approx(0.0, abs=1e-8)


def test_divergent_predictions_positive_disagreement():
    """Verify that divergent predictions (e.g. Model A 2% vs Model B 5%) yield positive disagreement."""
    preds = np.array([2.0, 5.0])
    d = compute_model_disagreement(preds)
    assert d > 0.10
