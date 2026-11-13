"""Test 8: Overnight What-If simulator exact values for 1,00,000 INR portfolio."""
import pytest


def simulate_fixed_scenarios(portfolio_value: float, shock_pct: float) -> float:
    """Computes hypothetical opening portfolio value: V = V_0 * (1 + shock_pct)."""
    return portfolio_value * (1.0 + shock_pct)


def test_whatif_exact_calculations():
    """Verify 1,00,000 INR portfolio yields exactly 99000, 97000, 95000, 92000 INR."""
    base_val = 100000.0

    assert simulate_fixed_scenarios(base_val, -0.01) == 99000.0
    assert simulate_fixed_scenarios(base_val, -0.03) == 97000.0
    assert simulate_fixed_scenarios(base_val, -0.05) == 95000.0
    assert simulate_fixed_scenarios(base_val, -0.08) == 92000.0
