"""Test 4: Hybrid residual model exact fallback to HAR baseline when alpha = 0."""
import numpy as np
import pytest


def hybrid_predict(har_pred: np.ndarray, residual_pred: np.ndarray, alpha: float, eps: float = 1e-4) -> np.ndarray:
    """
    Hybrid residual combination:
      y_hat = max(eps, har_pred + alpha * residual_pred)
    """
    raw = har_pred + alpha * residual_pred
    return np.maximum(eps, raw)


def test_alpha_zero_exact_har_fallback():
    """Verify that when alpha=0, hybrid output matches HAR output bitwise/exactly above eps."""
    har_preds = np.array([1.25, 2.50, 0.85, 3.10, 0.05])
    residual_preds = np.array([0.45, -0.80, 1.20, -0.30, 0.50])

    # With alpha = 0
    hybrid_out = hybrid_predict(har_preds, residual_preds, alpha=0.0, eps=1e-4)
    expected = np.maximum(1e-4, har_preds)

    np.testing.assert_array_almost_equal(hybrid_out, expected, decimal=10)
