"""Test 10: Liquidity trap detector proxy calculation and uncertainty scoring."""
import math
import pytest


def corwin_schultz_proxy(high: float, low: float) -> float:
    """Simple High-Low range spread estimator proxy: (High - Low) / ((High + Low) / 2)."""
    mid = (high + low) / 2.0
    return max(0.0, (high - low) / max(1e-4, mid))


def compute_execution_uncertainty(spread_proxy: float, volume_ratio: float) -> float:
    """
    Higher spread and lower volume ratio (thin liquidity) increases execution uncertainty score (0-100).
    volume_ratio = current_volume / median_volume
    """
    # Inverse relation with volume, direct with spread
    score = 100.0 * (1.0 - math.exp(-2.0 * spread_proxy / max(0.1, volume_ratio)))
    return min(100.0, max(0.0, score))


def test_liquidity_proxy_non_negative():
    """Verify spread proxy is always non-negative."""
    spread = corwin_schultz_proxy(105.0, 95.0)
    assert spread >= 0.0


def test_wide_range_low_volume_raises_uncertainty():
    """Verify wide price range combined with low volume raises execution uncertainty."""
    # Calm bar: narrow range, normal volume
    calm_spread = corwin_schultz_proxy(100.5, 99.5)
    calm_score = compute_execution_uncertainty(calm_spread, volume_ratio=1.0)

    # Stressed thin bar: wide range, low volume (e.g. 0.2x of normal)
    stress_spread = corwin_schultz_proxy(105.0, 95.0)
    stress_score = compute_execution_uncertainty(stress_spread, volume_ratio=0.2)

    assert stress_score > calm_score
