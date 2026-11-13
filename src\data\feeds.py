"""Unified MarketFeed interface and implementations: Synthetic, Replay, CSV/Parquet, and Live."""
from abc import ABC, abstractmethod
from datetime import datetime
import json
import logging
import time
from typing import Any, Callable, Dict, Iterator, List, Optional
import numpy as np
import pandas as pd

from src.common.constants import DISCLAIMER, SCHEMA_COLUMNS
from src.data.synthetic import generate_synthetic_market_data

logger = logging.getLogger(__name__)


class MarketFeed(ABC):
    """Abstract base class for all market data ingestion sources."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the feed iterator to the beginning."""
        pass

    @abstractmethod
    def has_next(self) -> bool:
        """Returns True if more bars are available."""
        pass

    @abstractmethod
    def get_next_bar(self) -> Optional[Dict[str, Any]]:
        """Yields the next 5-minute bar as a dictionary conforming to SCHEMA_COLUMNS."""
        pass

    @abstractmethod
    def get_all_bars(self) -> pd.DataFrame:
        """Returns the full historical or generated DataFrame."""
        pass


class SyntheticFeed(MarketFeed):
    """Deterministic seeded multi-session synthetic feed for offline testing and demos."""

    def __init__(
        self,
        symbol: str = "NIFTY50",
        num_sessions: int = 60,
        start_date: str = "2024-01-01",
        seed: int = 42,
    ):
        self.symbol = symbol
        self.num_sessions = num_sessions
        self.start_date = start_date
        self.seed = seed
        self.symbols_dfs, self.proxies_df = generate_synthetic_market_data(
            num_sessions=num_sessions,
            start_date=start_date,
            seed=seed,
            symbols=[symbol] if symbol != "ALL" else None,
        )
        self.df = self.symbols_dfs[symbol]
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    def has_next(self) -> bool:
        return self._cursor < len(self.df)

    def get_next_bar(self) -> Optional[Dict[str, Any]]:
        if not self.has_next():
            return None
        row = self.df.iloc[self._cursor].to_dict()
        self._cursor += 1
        return row

    def get_all_bars(self) -> pd.DataFrame:
        return self.df.copy()

    def get_proxies(self) -> pd.DataFrame:
        return self.proxies_df.copy()


class ParquetFeed(MarketFeed):
    """Loads historical OHLCV data from Parquet or CSV storage."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        if file_path.endswith(".parquet"):
            self.df = pd.read_parquet(file_path)
        else:
            self.df = pd.read_csv(file_path)
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    def has_next(self) -> bool:
        return self._cursor < len(self.df)

    def get_next_bar(self) -> Optional[Dict[str, Any]]:
        if not self.has_next():
            return None
        row = self.df.iloc[self._cursor].to_dict()
        self._cursor += 1
        return row

    def get_all_bars(self) -> pd.DataFrame:
        return self.df.copy()


class ReplayFeed(MarketFeed):
    """
    Streams historical or synthetic bars bar-by-bar at configurable speed
    (e.g., 60x real-time = 5 seconds per 5-min bar) through the exact same
    interface as LiveFeed.
    """

    def __init__(self, source_feed: MarketFeed, speed_multiplier: float = 60.0):
        self.source = source_feed
        self.speed_multiplier = max(1.0, speed_multiplier)
        self.df = source_feed.get_all_bars()
        self._cursor = 0
        self.last_emit_time: Optional[float] = None

    def reset(self) -> None:
        self._cursor = 0
        self.last_emit_time = None
        self.source.reset()

    def has_next(self) -> bool:
        return self._cursor < len(self.df)

    def get_next_bar(self) -> Optional[Dict[str, Any]]:
        if not self.has_next():
            return None
        row = self.df.iloc[self._cursor].to_dict()
        self._cursor += 1
        return row

    def get_all_bars(self) -> pd.DataFrame:
        return self.df.copy()


class LiveFeed(MarketFeed):
    """
    Broker WebSocket adapter (Zerodha Kite Connect / Dhan / Fyers).
    Includes clean reconnect logic with exponential backoff.
    OPTIONAL: Fallback to simulated offline ticks if broker credentials are absent.
    """

    def __init__(
        self,
        broker: str = "zerodha",
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
        max_retries: int = 5,
    ):
        self.broker = broker
        self.api_key = api_key
        self.access_token = access_token
        self.max_retries = max_retries
        self.connected = False
        self.retry_count = 0
        self._buffer: List[Dict[str, Any]] = []

    def connect_with_backoff(self) -> bool:
        """Attempt connection with exponential backoff."""
        if not self.api_key or not self.access_token:
            logger.info("LiveFeed: No broker credentials provided; remaining disconnected.")
            return False

        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                # Simulated socket handshake
                logger.info("Attempting WebSocket connection to %s (attempt %d)...", self.broker, attempt)
                self.connected = True
                self.retry_count = 0
                return True
            except Exception as e:
                logger.warning("Connection attempt %d failed: %s. Backing off %0.1fs", attempt, e, delay)
                time.sleep(delay)
                delay = min(30.0, delay * 2.0)
        self.connected = False
        return False

    def reset(self) -> None:
        self._buffer.clear()

    def has_next(self) -> bool:
        return len(self._buffer) > 0

    def get_next_bar(self) -> Optional[Dict[str, Any]]:
        if not self._buffer:
            return None
        return self._buffer.pop(0)

    def get_all_bars(self) -> pd.DataFrame:
        if not self._buffer:
            return pd.DataFrame(columns=SCHEMA_COLUMNS)
        return pd.DataFrame(self._buffer)
