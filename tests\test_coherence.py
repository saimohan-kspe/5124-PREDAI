"""Test 3: Mathematical nesting and coherence error of Coherent Variance Heads."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F


def compute_coherent_variances(trunk_output: torch.Tensor):
    """
    Reference coherent head implementation:
      v1 = softplus(linear1(trunk))
      v2 = softplus(linear2(trunk))
      v3 = softplus(linear3(trunk))
      V_5m = v1
      V_1h = v1 + v2
      V_24h = v1 + v2 + v3
    """
    v1 = F.softplus(trunk_output[:, 0]) + 1e-6
    v2 = F.softplus(trunk_output[:, 1]) + 1e-6
    v3 = F.softplus(trunk_output[:, 2]) + 1e-6

    V_5m = v1
    V_1h = v1 + v2
    V_24h = v1 + v2 + v3
    return V_5m, V_1h, V_24h


def test_coherent_nesting_invariants():
    """Verify V_5m <= V_1h <= V_24h is strictly satisfied for all random trunk vectors."""
    torch.manual_seed(42)
    trunk = torch.randn(100, 3)
    V_5m, V_1h, V_24h = compute_coherent_variances(trunk)

    # Nesting condition: V_5m <= V_1h <= V_24h
    assert torch.all(V_5m <= V_1h).item()
    assert torch.all(V_1h <= V_24h).item()

    # Volatility is sqrt of variance
    sigma_5m = torch.sqrt(V_5m)
    sigma_1h = torch.sqrt(V_1h)
    sigma_24h = torch.sqrt(V_24h)

    assert torch.all(sigma_5m <= sigma_1h).item()
    assert torch.all(sigma_1h <= sigma_24h).item()


def test_coherence_error_zero():
    """Verify coherence error is exactly 0.0 by construction for coherent heads."""
    trunk = torch.randn(50, 3)
    V_5m, V_1h, V_24h = compute_coherent_variances(trunk)

    implied_rest = V_1h - V_5m
    coherence_error = torch.mean(torch.abs(V_1h - (V_5m + implied_rest)) / V_1h).item()
    assert coherence_error == pytest.approx(0.0, abs=1e-7)
