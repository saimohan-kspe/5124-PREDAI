"""Test 7: Verification that proxies are masked at 15:25 close origin and unmasked at PRE_OPEN."""
import numpy as np
import pytest


def get_proxy_vector(origin_kind: str, raw_proxy_returns: np.ndarray):
    """
    Constructs the 10-dimensional proxy vector:
      5 returns + 5 mask indicators.
      If origin_kind == 'SESSION_CLOSE' (15:25):
        returns = 0.0, mask = 0.0
      If origin_kind == 'PRE_OPEN' (09:10):
        returns = raw_returns, mask = 1.0
    """
    assert len(raw_proxy_returns) == 5
    if origin_kind == "SESSION_CLOSE":
        # At 15:25 IST, overnight proxies are not yet realized
        rets = np.zeros(5, dtype=float)
        mask = np.zeros(5, dtype=float)
    elif origin_kind == "PRE_OPEN":
        # At 09:10 IST, overnight proxy returns are known
        rets = np.array(raw_proxy_returns, dtype=float)
        mask = np.ones(5, dtype=float)
    else:
        raise ValueError(f"Unknown origin kind: {origin_kind}")

    return np.concatenate([rets, mask])


def test_proxy_mask_at_close_and_preopen():
    """Verify that 15:25 origin has mask=0 and returns=0, while PRE_OPEN is unmasked."""
    mock_returns = np.array([0.012, -0.008, 0.003, 0.002, 0.045])

    vec_close = get_proxy_vector("SESSION_CLOSE", mock_returns)
    assert len(vec_close) == 10
    # First 5 are zeroed
    assert np.all(vec_close[:5] == 0.0)
    # Mask elements are 0
    assert np.all(vec_close[5:] == 0.0)

    vec_preopen = get_proxy_vector("PRE_OPEN", mock_returns)
    assert len(vec_preopen) == 10
    # Values match mock returns
    np.testing.assert_array_equal(vec_preopen[:5], mock_returns)
    # Mask elements are 1
    assert np.all(vec_preopen[5:] == 1.0)
