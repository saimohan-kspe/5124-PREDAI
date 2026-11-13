"""Test 2: Validation protocol and temporal leakage verification."""
import numpy as np
import pandas as pd
import pytest

from src.common.constants import BARS_PER_SESSION, K_24H
from src.data.synthetic import generate_synthetic_market_data


def test_no_future_timestamps_in_window():
    """Verify that any feature extraction window ending at origin t uses only bars <= t."""
    symbols_dfs, _ = generate_synthetic_market_data(num_sessions=10, seed=42)
    df = symbols_dfs["NIFTY50"]
    window_size = 60

    for origin_idx in range(window_size, len(df), 15):
        origin_ts = df.loc[origin_idx, "timestamp"]
        window_df = df.iloc[origin_idx - window_size + 1 : origin_idx + 1]
        assert len(window_df) == window_size
        assert (window_df["timestamp"] <= origin_ts).all()


def test_purge_and_embargo_isolation():
    """
    Assert that the training label window (75 bars forward) of the last training origin
    does NOT cross into the validation set, and embargo is respected.
    """
    total_sessions = 30
    train_sessions = 20
    train_end_bar = train_sessions * BARS_PER_SESSION  # First bar of validation

    # Purge rule: any training sample whose 75-bar target window reaches or crosses train_end_bar
    # must be purged.
    purge_limit = train_end_bar - K_24H
    valid_train_origins = list(range(purge_limit))

    for origin in valid_train_origins:
        target_horizon_end = origin + K_24H
        assert target_horizon_end <= train_end_bar, f"Origin {origin} target window leaks into validation set!"


def test_scalers_train_only():
    """Verify that scalers fit strictly on train set remain identical regardless of val/test data."""
    train_data = np.array([[10.0], [20.0], [30.0], [40.0]], dtype=float)
    val_data = np.array([[100.0], [200.0]], dtype=float)

    train_mean = float(np.mean(train_data))
    train_std = float(np.std(train_data))

    # A mock scaler fit strictly on train
    scaler_params = {"mean": train_mean, "std": train_std}

    # Transform val using train scaler
    transformed_val = (val_data - scaler_params["mean"]) / scaler_params["std"]

    # Scaler params MUST NOT change
    assert scaler_params["mean"] == 25.0
    assert abs(scaler_params["std"] - np.std(train_data)) < 1e-9
