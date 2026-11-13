"""Deterministic seeded synthetic data generator for multi-symbol multi-session 5-min bars."""
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.common.constants import (
    BAR_DURATION_MINUTES,
    BARS_PER_SESSION,
    DISCLAIMER,
    SCHEMA_COLUMNS,
    TIMEZONE,
)
from src.data.calendar import (
    compute_closure_minutes,
    compute_delta_log,
    generate_session_timestamps,
    get_trading_days,
)

# 8 Symbols with sector allocations
SYMBOL_METADATA = {
    "NIFTY50": {"sector": "Market", "beta": 1.0, "base_price": 21500.0, "annual_vol": 0.14},
    "RELIANCE": {"sector": "Energy", "beta": 1.15, "base_price": 2700.0, "annual_vol": 0.22},
    "TCS": {"sector": "IT", "beta": 0.85, "base_price": 3800.0, "annual_vol": 0.18},
    "HDFCBANK": {"sector": "Banking", "beta": 1.10, "base_price": 1650.0, "annual_vol": 0.20},
    "INFY": {"sector": "IT", "beta": 0.95, "base_price": 1550.0, "annual_vol": 0.23},
    "ICICIBANK": {"sector": "Banking", "beta": 1.20, "base_price": 1050.0, "annual_vol": 0.24},
    "ITC": {"sector": "FMCG", "beta": 0.65, "base_price": 450.0, "annual_vol": 0.16},
    "LT": {"sector": "Infrastructure", "beta": 1.05, "base_price": 3500.0, "annual_vol": 0.21},
}

SECTOR_MAP = {
    "Energy": 0,
    "IT": 1,
    "Banking": 2,
    "FMCG": 3,
    "Infrastructure": 3,
    "Market": -1,
}


def intraday_vol_multiplier(bar_idx: int) -> float:
    """
    Empirical U-shaped intraday volatility curve for NSE (75 bars, 0 to 74):
    Higher at open (09:15-10:00) and close (14:45-15:30), calm midday.
    """
    x = bar_idx / 74.0
    # Quadratic U-shape centered at ~0.45 (around 12:30 IST)
    return float(1.8 - 2.8 * x + 3.1 * (x ** 2))


def generate_synthetic_market_data(
    num_sessions: int = 60,
    start_date: str = "2024-01-01",
    seed: int = 42,
    symbols: Optional[List[str]] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Generates multi-session 5-minute bars for multiple symbols using a factor model:
      r_{i,t} = beta_{m,i} * F_{m,t} + beta_{s,i} * F_{s,t} + epsilon_{i,t}
    Also generates synthetic overnight proxy returns.
    
    Returns:
      symbols_dfs: Dict[symbol, DataFrame with SCHEMA_COLUMNS]
      proxies_df: DataFrame of proxy returns aligned with sessions
    """
    np.random.seed(seed)
    if symbols is None:
        symbols = list(SYMBOL_METADATA.keys())

    trading_days = get_trading_days(start_date, num_sessions)
    total_bars = num_sessions * BARS_PER_SESSION
    num_symbols = len(symbols)

    # Pre-generate session timestamps
    session_timestamps: List[List[pd.Timestamp]] = [
        generate_session_timestamps(day) for day in trading_days
    ]

    # Pre-calculate dt_minutes and gap_flag for each bar
    all_timestamps: List[pd.Timestamp] = []
    all_session_ids: List[int] = []
    all_bar_indices: List[int] = []
    all_dt_minutes: List[float] = []
    all_gap_flags: List[int] = []

    for s_idx, day_ts in enumerate(session_timestamps):
        for b_idx, ts in enumerate(day_ts):
            all_timestamps.append(ts)
            all_session_ids.append(s_idx)
            all_bar_indices.append(b_idx)
            if s_idx == 0 and b_idx == 0:
                all_dt_minutes.append(float(BAR_DURATION_MINUTES))
                all_gap_flags.append(0)
            elif b_idx == 0:
                # Opening bar of day s_idx: previous close was 15:30 of day s_idx-1
                prev_close_ts = session_timestamps[s_idx - 1][-1] + pd.Timedelta(minutes=BAR_DURATION_MINUTES)
                closure_m = compute_closure_minutes(prev_close_ts, ts)
                all_dt_minutes.append(closure_m)
                all_gap_flags.append(1)
            else:
                all_dt_minutes.append(float(BAR_DURATION_MINUTES))
                all_gap_flags.append(0)

    # Factor model innovation generation
    # Market factor (annual vol ~15%), Sector factors (4 sectors), Idiosyncratic noise
    dt_annual = (5.0 / (375.0 * 250.0))  # 5-min fraction of annual trading year
    base_sigma_bar = math.sqrt(dt_annual)

    # 1 Market factor + 4 Sector factors
    F_market = np.random.normal(0, 1, size=total_bars)
    F_sectors = np.random.normal(0, 1, size=(4, total_bars))

    # Overnight gap shocks for each session (s_idx > 0)
    gap_shocks_market = np.zeros(num_sessions)
    gap_shocks_sectors = np.zeros((4, num_sessions))
    for s in range(1, num_sessions):
        closure_dt = all_dt_minutes[s * BARS_PER_SESSION]
        closure_scale = math.sqrt(closure_dt / (375.0 * 250.0))
        gap_shocks_market[s] = np.random.normal(0, 0.18 * closure_scale)
        for sec in range(4):
            gap_shocks_sectors[sec, s] = np.random.normal(0, 0.12 * closure_scale)

    symbols_dfs: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        meta = SYMBOL_METADATA.get(sym, {"sector": "Market", "beta": 1.0, "base_price": 1000.0, "annual_vol": 0.20})
        sec_idx = SECTOR_MAP.get(meta["sector"], -1)
        beta_m = meta["beta"]
        ann_vol = meta["annual_vol"]
        base_price = meta["base_price"]

        prices_open = np.zeros(total_bars)
        prices_high = np.zeros(total_bars)
        prices_low = np.zeros(total_bars)
        prices_close = np.zeros(total_bars)
        volumes = np.zeros(total_bars)
        gap_returns = np.zeros(total_bars)

        curr_price = base_price

        for i in range(total_bars):
            s_idx = all_session_ids[i]
            b_idx = all_bar_indices[i]
            is_gap = (all_gap_flags[i] == 1)

            u_mult = intraday_vol_multiplier(b_idx)
            bar_vol = ann_vol * base_sigma_bar * u_mult

            # Idiosyncratic return
            idio = np.random.normal(0, bar_vol * 0.6)
            m_comp = beta_m * F_market[i] * bar_vol * 0.7
            s_comp = (F_sectors[sec_idx, i] * bar_vol * 0.4) if sec_idx >= 0 else 0.0

            intraday_ret = m_comp + s_comp + idio

            if is_gap:
                # Overnight jump
                closure_dt = all_dt_minutes[i]
                gap_scale = math.sqrt(closure_dt / (375.0 * 250.0))
                g_idio = np.random.normal(0, ann_vol * gap_scale * 0.5)
                g_m = beta_m * gap_shocks_market[s_idx]
                g_s = gap_shocks_sectors[sec_idx, s_idx] if sec_idx >= 0 else 0.0
                g_total = g_m + g_s + g_idio

                bar_open = curr_price * math.exp(g_total)
                gap_returns[i] = g_total
            else:
                bar_open = curr_price
                gap_returns[i] = 0.0

            # Intraday progression from open to close
            bar_close = bar_open * math.exp(intraday_ret)
            high_jump = abs(np.random.normal(0, bar_vol * 0.5))
            low_jump = abs(np.random.normal(0, bar_vol * 0.5))
            bar_high = max(bar_open, bar_close) * (1.0 + high_jump)
            bar_low = min(bar_open, bar_close) * (1.0 - low_jump)

            # Volume model: higher at open/close and higher on high-volatility bars
            base_vol = 50000.0 * (1000.0 / max(50.0, curr_price))
            vol_mult = (u_mult ** 1.2) * (1.0 + 5.0 * abs(intraday_ret) / max(0.001, bar_vol))
            volume = float(np.random.poisson(max(100, int(base_vol * vol_mult))))

            prices_open[i] = bar_open
            prices_high[i] = bar_high
            prices_low[i] = bar_low
            prices_close[i] = bar_close
            volumes[i] = volume

            curr_price = bar_close

        df = pd.DataFrame({
            "timestamp": all_timestamps,
            "open": prices_open,
            "high": prices_high,
            "low": prices_low,
            "close": prices_close,
            "volume": volumes,
            "gap_ret": gap_returns,
            "dt_minutes": all_dt_minutes,
            "gap_flag": all_gap_flags,
            "session_id": all_session_ids,
            "bar_idx": all_bar_indices,
        })
        df.attrs["disclaimer"] = DISCLAIMER
        symbols_dfs[sym] = df

    # Generate synthetic overnight proxy series for each session
    # Proxies: us_fut, brent, gold, usdinr, india_vix
    proxy_records = []
    for s in range(num_sessions):
        m_jump = gap_shocks_market[s] if s > 0 else 0.0
        # US futures correlated positively with Indian market
        us_fut = 0.70 * m_jump + np.random.normal(0, 0.006)
        # Brent crude negatively correlated with Indian market
        brent = -0.45 * m_jump + np.random.normal(0, 0.010)
        # Gold safe haven
        gold = -0.25 * m_jump + np.random.normal(0, 0.005)
        # USD/INR increases when Nifty gaps down
        usdinr = -0.35 * m_jump + np.random.normal(0, 0.002)
        # India VIX rises sharply when market gaps down
        vix_chg = -2.5 * m_jump + np.random.normal(0, 0.03)

        proxy_records.append({
            "session_id": s,
            "date": trading_days[s].strftime("%Y-%m-%d"),
            "us_fut_ret": us_fut,
            "brent_ret": brent,
            "gold_ret": gold,
            "usdinr_ret": usdinr,
            "india_vix_chg": vix_chg,
        })

    proxies_df = pd.DataFrame(proxy_records)
    proxies_df.attrs["disclaimer"] = DISCLAIMER
    return symbols_dfs, proxies_df
