"""NSE trading calendar, session timing, and closure duration calculations."""
import math
from datetime import datetime, time, timedelta
from typing import List, Tuple
import pandas as pd

from src.common.constants import (
    BAR_DURATION_MINUTES,
    BARS_PER_SESSION,
    MAX_DELTA_LOG,
    OVERNIGHT_MINUTES,
    TIMEZONE,
    WEEKEND_MINUTES,
)

# Standard NSE exchange holidays (example sample for 2024 calendar)
NSE_HOLIDAYS_2024 = {
    "2024-01-22",  # Special Holiday
    "2024-01-26",  # Republic Day
    "2024-03-08",  # Mahashivratri
    "2024-03-25",  # Holi
    "2024-03-29",  # Good Friday
    "2024-04-11",  # Id-Ul-Fitr
    "2024-04-17",  # Ram Navami
    "2024-05-01",  # Maharashtra Day
    "2024-06-17",  # Bakri Id
    "2024-07-17",  # Muharram
    "2024-08-15",  # Independence Day
    "2024-10-02",  # Mahatma Gandhi Jayanti
    "2024-11-01",  # Diwali Laxmi Pujan
    "2024-11-15",  # Gurunanak Jayanti
    "2024-12-25",  # Christmas
}


def is_trading_day(dt: datetime, holidays: set = NSE_HOLIDAYS_2024) -> bool:
    """Returns True if weekday (Mon-Fri) and not an exchange holiday."""
    if dt.weekday() >= 5:  # Saturday or Sunday
        return False
    date_str = dt.strftime("%Y-%m-%d")
    return date_str not in holidays


def get_trading_days(start_date: str, num_days: int) -> List[datetime]:
    """Generates consecutive valid NSE trading day dates starting from start_date."""
    days = []
    curr = pd.to_datetime(start_date)
    while len(days) < num_days:
        if is_trading_day(curr):
            days.append(curr.to_pydatetime())
        curr += timedelta(days=1)
    return days


def generate_session_timestamps(date: datetime) -> List[pd.Timestamp]:
    """
    Generates exactly 75 five-minute bar opening timestamps for an NSE session:
    09:15, 09:20, ..., 15:25 IST.
    """
    timestamps = []
    base_date = pd.Timestamp(date).tz_localize(None) if pd.Timestamp(date).tzinfo is None else pd.Timestamp(date).tz_convert(None)
    start_dt = base_date.replace(hour=9, minute=15, second=0, microsecond=0)
    for i in range(BARS_PER_SESSION):
        ts = start_dt + timedelta(minutes=i * BAR_DURATION_MINUTES)
        ts_tz = pd.Timestamp(ts).tz_localize(TIMEZONE)
        timestamps.append(ts_tz)
    return timestamps


def compute_closure_minutes(prev_close_ts: pd.Timestamp, curr_open_ts: pd.Timestamp) -> float:
    """
    Computes elapsed physical minutes between the previous session close (15:30)
    and current session open (09:15).
    """
    diff = (curr_open_ts - prev_close_ts).total_seconds() / 60.0
    return max(float(BAR_DURATION_MINUTES), diff)


def compute_delta_log(dt_minutes: float) -> float:
    """
    delta_log = ln(dt_minutes / 5), clipped at 8.0.
    For normal 5m bar: dt_minutes=5 -> ln(1) = 0.0.
    For overnight: dt_minutes=1065 -> ln(213) ~ 5.361.
    For weekend: dt_minutes=3945 -> ln(789) ~ 6.671.
    """
    val = math.log(max(1.0, dt_minutes / float(BAR_DURATION_MINUTES)))
    return min(float(MAX_DELTA_LOG), max(0.0, val))
