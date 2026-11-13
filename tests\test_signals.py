"""Test 6: Trading strategy signals, no-trade band, bounds, and gap rule execution."""
import pytest


def generate_signal(
    sigma_hat_24h: float,
    prev_fraction: float,
    overnight_proxy_move: float = 0.0,
    risk_budget: float = 0.01,
    f_max: float = 1.0,
    gap_theta: float = 0.02,
    gap_kappa: float = 0.5,
    band: float = 0.10,
):
    """
    Reference volatility sizing and signal generation logic:
      f_raw = min(f_max, R / sigma_hat_24h)
      if |overnight proxy move| > theta: f_target = kappa * f_raw
      else: f_target = f_raw
      if f_target > prev_fraction + band: BUY
      elif f_target < prev_fraction - band: SELL
      else: HOLD (target remains prev_fraction to avoid churn)
    """
    f_raw = min(f_max, max(0.0, risk_budget / max(1e-4, sigma_hat_24h)))
    if abs(overnight_proxy_move) > gap_theta:
        f_target = gap_kappa * f_raw
        reason = "GAP_SHOCK_PROTECTION"
    else:
        f_target = f_raw
        reason = "NORMAL_VOL_SIZING"

    f_target = min(f_max, max(0.0, f_target))

    if f_target > prev_fraction + band:
        sig = "BUY"
        actual_f = f_target
    elif f_target < prev_fraction - band:
        sig = "SELL"
        actual_f = f_target
    else:
        sig = "HOLD"
        actual_f = prev_fraction

    return sig, actual_f, reason


def test_signals_bounds_and_no_trade_band():
    """Verify target fraction strictly in [0, f_max] and changes within band emit HOLD."""
    # When prev is 0.50 and new is 0.55 (change is 0.05 < 0.10 band) -> HOLD
    sig, f_act, _ = generate_signal(
        sigma_hat_24h=0.019,  # 0.01 / 0.019 ~ 0.526
        prev_fraction=0.50,
        band=0.10,
    )
    assert sig == "HOLD"
    assert f_act == 0.50

    # When new target is 0.70 (change 0.20 > 0.10) -> BUY
    sig, f_act, _ = generate_signal(
        sigma_hat_24h=0.014,  # 0.01 / 0.014 ~ 0.714
        prev_fraction=0.50,
        band=0.10,
    )
    assert sig == "BUY"
    assert 0.0 <= f_act <= 1.0


def test_gap_rule_fires_on_proxy_move():
    """Verify that proxy move > 2% cuts exposure in half via kappa multiplier."""
    sig_normal, f_norm, _ = generate_signal(
        sigma_hat_24h=0.01,
        prev_fraction=0.0,
        overnight_proxy_move=0.01,  # 1% < 2%
    )
    sig_gap, f_gap, reason = generate_signal(
        sigma_hat_24h=0.01,
        prev_fraction=0.0,
        overnight_proxy_move=0.025,  # 2.5% > 2%
    )
    assert f_norm == 1.0
    assert f_gap == 0.5
    assert reason == "GAP_SHOCK_PROTECTION"
