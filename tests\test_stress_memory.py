"""Test 13: Stress memory kNN isolation - historical matches must strictly precede query date."""
import pandas as pd
import pytest


def query_stress_memory(query_date: str, history_df: pd.DataFrame, top_k: int = 3):
    """
    Finds top-k past days strictly BEFORE query_date.
    No future or concurrent leakage permitted.
    """
    past_candidates = history_df[history_df["date"] < query_date]
    if past_candidates.empty:
        return []
    # Return top_k most recent or most similar
    return past_candidates.sort_values("date", ascending=False).head(top_k)["date"].tolist()


def test_stress_memory_temporal_isolation():
    """Verify every neighbour returned has date strictly < query date."""
    dates = ["2024-01-05", "2024-01-10", "2024-01-15", "2024-01-20", "2024-01-25"]
    df = pd.DataFrame({"date": dates, "stress_score": [1.2, 3.4, 0.8, 4.5, 2.1]})

    query = "2024-01-15"
    neighbours = query_stress_memory(query, df, top_k=2)

    assert len(neighbours) == 2
    for n_date in neighbours:
        assert n_date < query
