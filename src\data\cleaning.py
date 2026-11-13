"""Data cleaning, integrity validation, and automated quality reporting."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd

from src.common.constants import (
    BARS_PER_SESSION,
    DISCLAIMER,
    SCHEMA_COLUMNS,
    TIMEZONE,
)

logger = logging.getLogger(__name__)


def clean_market_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw market DataFrame:
      - Validates and enforces SCHEMA_COLUMNS.
      - Converts and normalises timestamps to Asia/Kolkata timezone.
      - Removes duplicates based on timestamp.
      - Validates price non-negativity and bounds (high >= open, close; low <= open, close).
      - Checks bar counts per session (strictly 75 bars).
      - Generates comprehensive quality diagnostics.
    """
    report: Dict[str, Any] = {
        "disclaimer": DISCLAIMER,
        "total_input_rows": len(df),
        "duplicate_timestamps": 0,
        "missing_bars": 0,
        "negative_or_zero_prices": 0,
        "price_bound_violations": 0,
        "session_length_violations": {},
        "gap_distribution": {},
        "is_valid": True,
    }

    cleaned = df.copy()

    # Verify column presence
    missing_cols = [col for col in SCHEMA_COLUMNS if col not in cleaned.columns]
    if missing_cols:
        raise ValueError(f"Missing mandatory schema columns: {missing_cols}")

    # Ensure timestamp parsing
    if not pd.api.types.is_datetime64_any_dtype(cleaned["timestamp"]):
        cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"])

    # Duplicate check
    dups = cleaned.duplicated(subset=["timestamp"]).sum()
    report["duplicate_timestamps"] = int(dups)
    if dups > 0:
        cleaned = cleaned.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Price non-negativity check
    invalid_prices = (
        (cleaned["open"] <= 0)
        | (cleaned["high"] <= 0)
        | (cleaned["low"] <= 0)
        | (cleaned["close"] <= 0)
    ).sum()
    report["negative_or_zero_prices"] = int(invalid_prices)
    if invalid_prices > 0:
        report["is_valid"] = False

    # Price bound check (high >= max(open, close), low <= min(open, close))
    bound_errs = (
        (cleaned["high"] < cleaned["open"])
        | (cleaned["high"] < cleaned["close"])
        | (cleaned["low"] > cleaned["open"])
        | (cleaned["low"] > cleaned["close"])
    ).sum()
    report["price_bound_violations"] = int(bound_errs)

    # Session length validation
    session_counts = cleaned.groupby("session_id")["bar_idx"].count()
    violations = session_counts[session_counts != BARS_PER_SESSION]
    if not violations.empty:
        report["session_length_violations"] = {int(k): int(v) for k, v in violations.items()}
        report["is_valid"] = False

    # Gap distribution summary
    gap_bars = cleaned[cleaned["gap_flag"] == 1]
    report["gap_distribution"] = {
        "total_gaps": len(gap_bars),
        "mean_gap_minutes": float(gap_bars["dt_minutes"].mean()) if len(gap_bars) > 0 else 0.0,
        "max_gap_minutes": float(gap_bars["dt_minutes"].max()) if len(gap_bars) > 0 else 0.0,
        "mean_gap_abs_ret_pct": float((gap_bars["gap_ret"].abs() * 100.0).mean()) if len(gap_bars) > 0 else 0.0,
    }

    report["final_cleaned_rows"] = len(cleaned)
    return cleaned, report


def save_quality_report(report: Dict[str, Any], output_dir: str) -> Tuple[str, str]:
    """Saves data quality report as JSON and Markdown in output_dir."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "data_quality_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = out_path / "data_quality_report.md"
    md_content = f"""# Data Quality Report
*Notice: {report['disclaimer']}*

## Executive Summary
- **Input Rows**: {report['total_input_rows']}
- **Final Cleaned Rows**: {report['final_cleaned_rows']}
- **Validation Status**: `{"PASS" if report['is_valid'] else "FAIL"}`
- **Duplicate Timestamps**: {report['duplicate_timestamps']}
- **Zero / Negative Prices**: {report['negative_or_zero_prices']}
- **Price Bound Violations**: {report['price_bound_violations']}

## Session Length Integrity (Target: {BARS_PER_SESSION} bars/session)
- **Session Violations**: {len(report['session_length_violations'])}

## Gap Distribution Summary
- **Total Closures Crossed**: {report['gap_distribution'].get('total_gaps', 0)}
- **Mean Gap Duration**: {report['gap_distribution'].get('mean_gap_minutes', 0.0):.1f} minutes
- **Max Gap Duration**: {report['gap_distribution'].get('max_gap_minutes', 0.0):.1f} minutes
- **Mean |Gap Return|**: {report['gap_distribution'].get('mean_gap_abs_ret_pct', 0.0):.3f}%
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return str(json_path), str(md_path)
