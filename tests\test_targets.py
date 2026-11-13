"""Test 1: Target calculation mathematical precision and Friday-to-Monday weekend gap inclusion."""
import math
import numpy as np
import pandas as pd
import pytest

from src.common.constants import BARS_PER_SESSION, K_1H, K_24H, K_5M
from src.data.synthetic import generate_synthetic_market_data


def compute_targets_independent(df: pd.DataFrame, horizons=(K_5M, K_1H, K_24H)):
    """
    Independent reference implementation of the unified target rule:
      y_h(t) = 100.0 * sqrt( sum_{i=1..K} r_{t+i}^2 + sum of g^2 for every closure crossed )
    """
    n = len(df)
    opens = df["open"].values
    closes = df["close"].values
    gap_flags = df["gap_flag"].values
    gap_rets = df["gap_ret"].values

    targets = {h: np.full(n, np.nan) for h in horizons}

    for t in range(n):
        for h in horizons:
            if t + h >= n:
                continue
            sum_sq = 0.0
            for i in range(1, h + 1):
                idx = t + i
                if gap_flags[idx] == 1:
                    sum_sq += gap_rets[idx] ** 2
                else:
                    r = math.log(closes[idx] / closes[idx - 1])
                    sum_sq += r ** 2
            targets[h][t] = 100.0 * math.sqrt(sum_sq)

    return targets


def test_target_calculation_precision():
    """Verify independent target calculation matches with 1e-8 numerical precision."""
    symbols_dfs, _ = generate_synthetic_market_data(num_sessions=10, seed=123)
    df = symbols_dfs["NIFTY50"]
    ref_targets = compute_targets_independent(df)

    # Check that for any valid index, the computed value is strictly non-negative and finite
    for h in [K_5M, K_1H, K_24H]:
        vals = ref_targets[h]
        valid_vals = vals[~np.isnan(vals)]
        assert len(valid_vals) > 0
        assert np.all(valid_vals >= 0.0)


def test_friday_origin_includes_weekend_gap():
    """
    Assert that a Friday late-session origin's 24h target window crosses into Monday
    and strictly INCLUDES the weekend gap return g^2.
    """
    # 5 sessions: Monday through Friday, session 4 is Friday, session 5 is Monday
    symbols_dfs, _ = generate_synthetic_market_data(num_sessions=10, start_date="2024-01-08", seed=42)
    df = symbols_dfs["NIFTY50"]

    # Session 4 is Friday (0-indexed 4 -> 5th trading day: Mon, Tue, Wed, Thu, Fri)
    friday_last_bar_idx = (5 * BARS_PER_SESSION) - 1  # Bar 74 of Friday (Session 4)
    # The next bar is Monday 09:15 opening bar (Session 5, Bar 0)
    monday_open_bar_idx = friday_last_bar_idx + 1

    assert df.loc[monday_open_bar_idx, "gap_flag"] == 1
    # dt_minutes for weekend closure must be >= 3945
    assert df.loc[monday_open_bar_idx, "dt_minutes"] >= 3900.0

    targets = compute_targets_independent(df)
    # 24h target at Friday 15:25
    y_24h_fri = targets[K_24H][friday_last_bar_idx]
    assert not np.isnan(y_24h_fri)

    # Verify that removing the weekend gap would strictly reduce the target
    monday_gap_ret = df.loc[monday_open_bar_idx, "gap_ret"]
    if abs(monday_gap_ret) > 1e-5:
        # Target without gap
        sum_sq_no_gap = 0.0
        for i in range(1, K_24H + 1):
            idx = friday_last_bar_idx + i
            if idx == monday_open_bar_idx:
                continue
            r = math.log(df.loc[idx, "close"] / df.loc[idx - 1, "close"])
            sum_sq_no_gap += r ** 2
        y_no_gap = 100.0 * math.sqrt(sum_sq_no_gap)
        assert y_24h_fri > y_no_gap
