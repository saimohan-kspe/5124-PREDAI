"""Test 14: Risk blind spot detection for stale feed, scheduled events, and concentration."""
from datetime import datetime, timedelta
import pytest


def check_blindspots(
    current_time: datetime,
    last_bar_time: datetime,
    upcoming_events: list,
    weights: dict,
    stale_minutes_limit: float = 15.0,
):
    """Flags data staleness, concentration, and upcoming scheduled events."""
    flags = []

    # Check staleness
    delay = (current_time - last_bar_time).total_seconds() / 60.0
    if delay > stale_minutes_limit:
        flags.append("STALE_FEED")

    # Check scheduled events
    curr_date_str = current_time.strftime("%Y-%m-%d")
    for event in upcoming_events:
        if event.get("date") == curr_date_str:
            flags.append(f"EVENT_TODAY_{event.get('title')}")

    # Check single-stock concentration
    for sym, w in weights.items():
        if w > 0.35:
            flags.append(f"CONCENTRATION_EXCESS_{sym}")

    return flags


def test_blindspots_stale_feed_and_budget_event():
    """Verify stale feed (>15 min) and Union Budget event raise corresponding flags."""
    now = datetime(2024, 2, 1, 9, 30)
    last_bar = now - timedelta(minutes=25)  # 25 min stale

    events = [{"date": "2024-02-01", "title": "Union Budget Presentation"}]
    weights = {"RELIANCE": 0.40, "TCS": 0.20}

    flags = check_blindspots(now, last_bar, events, weights)

    assert "STALE_FEED" in flags
    assert "EVENT_TODAY_Union Budget Presentation" in flags
    assert "CONCENTRATION_EXCESS_RELIANCE" in flags
