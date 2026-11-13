"""Domain constants and frozen schemas for NSE cash market trading."""
from typing import List

DISCLAIMER: str = (
    "SYNTHETIC - NOT REAL MARKET RESULTS. "
    "Risk overlay research only; not investment advice."
)

TIMEZONE: str = "Asia/Kolkata"

# NSE Cash Market Timing
BARS_PER_SESSION: int = 75
BAR_DURATION_MINUTES: int = 5
SESSION_OPEN_TIME: str = "09:15"
SESSION_CLOSE_TIME: str = "15:30"
TOTAL_SESSION_MINUTES: int = 375  # 75 * 5

# Closures (minutes)
OVERNIGHT_MINUTES: int = 1065     # 17.75 hours (15:30 to 09:15 next day)
WEEKEND_MINUTES: int = 3945       # 65.75 hours (Fri 15:30 to Mon 09:15)

# Clipped log duration max
MAX_DELTA_LOG: float = 8.0

# Target horizons (bar count K)
K_5M: int = 1
K_1H: int = 12
K_24H: int = 75

# Frozen Market Data Schema
SCHEMA_COLUMNS: List[str] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "gap_ret",
    "dt_minutes",
    "gap_flag",
    "session_id",
    "bar_idx"
]

# Signals CSV Schema
SIGNALS_COLUMNS: List[str] = [
    "timestamp",
    "horizon",
    "sigma_hat",
    "gap_share",
    "risk_regime",
    "target_fraction",
    "prev_fraction",
    "signal",
    "reason_code",
    "gap_adjacent"
]
