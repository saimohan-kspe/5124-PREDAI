"""Test 9: Domino effect propagation, self-impact identity, and beta monotonicity."""
import numpy as np
import pytest


def propagate_domino_shock(
    shock_symbol: str,
    shock_pct: float,
    symbols: list,
    corr_matrix: np.ndarray,
    vols: np.ndarray,
    corr_threshold: float = 0.3,
):
    """
    Beta propagation: beta_{j|i} = corr_{ij} * (sigma_j / sigma_i)
    impact_j = beta_{j|i} * shock_pct
    """
    i = symbols.index(shock_symbol)
    sigma_i = vols[i]
    impacts = {}
    for j, sym in enumerate(symbols):
        if i == j:
            impacts[sym] = shock_pct
        else:
            c = corr_matrix[i, j]
            if abs(c) >= corr_threshold:
                beta = c * (vols[j] / sigma_i)
                impacts[sym] = beta * shock_pct
            else:
                impacts[sym] = 0.0
    return impacts


def test_domino_self_impact_and_ranking():
    """Verify self-impact is exactly the initial shock and impact ranks monotonically with beta."""
    symbols = ["RELIANCE", "TCS", "INFY"]
    corr = np.array([
        [1.0, 0.4, 0.2],
        [0.4, 1.0, 0.8],
        [0.2, 0.8, 1.0],
    ])
    vols = np.array([0.20, 0.20, 0.25])

    # Shock TCS by -5%
    shock_pct = -0.05
    impacts = propagate_domino_shock("TCS", shock_pct, symbols, corr, vols, corr_threshold=0.3)

    # Self-impact must be exactly the shock
    assert impacts["TCS"] == -0.05

    # INFY has higher correlation (0.8) and higher vol than RELIANCE (corr 0.4)
    # So INFY must suffer larger impact than RELIANCE
    assert abs(impacts["INFY"]) > abs(impacts["RELIANCE"])
