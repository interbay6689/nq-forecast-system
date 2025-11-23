"""Utilities for loading raw OHLCV data files.

This module defines a minimal data-source abstraction alongside a
``fetch_raw_data`` helper so downstream stages can rely on a normalized
OHLCV schema. A lightweight placeholder source is provided for now; swap it
with a real vendor (e.g., Polygon, Interactive Brokers) without changing the
consuming code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, MutableMapping, Protocol

import pandas as pd


DEFAULT_COLUMN_MAP: Mapping[str, str] = {
    "timestamp": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


class MissingColumnsError(ValueError):
    """Raised when a loaded CSV is missing required OHLCV columns."""


class RawDataSource(Protocol):
    """Protocol for pluggable OHLCV data providers."""

    def fetch(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Return raw OHLCV data as a ``DataFrame`` with vendor column names."""


@dataclass(slots=True)
class PlaceholderDataSource:
    """Simple in-memory data generator used as a stand-in for real data.

    The placeholder returns a minute-level ``DataFrame`` between ``start`` and
    ``end`` (right-exclusive) with deterministic synthetic prices. Replace this
    class with a real data-provider that implements the :class:`RawDataSource`
    protocol once an API is selected.
    """

    frequency: str = "1min"

    def fetch(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        index = pd.date_range(start, end, freq=self.frequency, inclusive="left", tz="UTC")
        if index.empty:
            return pd.DataFrame(columns=DEFAULT_COLUMN_MAP.values())

        base = pd.Series(range(len(index)), index=index)
        data = pd.DataFrame(
            {
                "timestamp": index,
                "open": 15000 + base * 0.25,
                "high": 15000 + base * 0.3,
                "low": 15000 + base * 0.2,
                "close": 15000 + base * 0.28,
                "volume": 10_000 + base * 5,
            }
        )
        data["symbol"] = symbol
        return data


def _validate_and_normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame includes the canonical OHLCV columns."""

    missing = set(DEFAULT_COLUMN_MAP.values()) - set(df.columns)
    if missing:
        raise MissingColumnsError(
            f"Data source is missing required columns: {', '.join(sorted(missing))}"
        )
    ordered_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    extras = [col for col in df.columns if col not in ordered_columns]
    return df[ordered_columns + extras]


def load_ohlcv_csv(
    path: str | Path,
    *,
    column_map: MutableMapping[str, str] | None = None,
    timezone: str | None = "UTC",
) -> pd.DataFrame:
    """Load OHLCV data from a CSV file into a normalized ``DataFrame``.

    This helper is useful for backfilling historical data from exported files
    while maintaining the canonical column order.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if column_map:
        df = df.rename(columns=column_map)

    df = _validate_and_normalize_columns(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")

    if timezone:
        df = df.tz_convert(timezone)
    else:
        df.index = df.index.tz_localize(None)

    return df


def fetch_raw_data(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    source: RawDataSource | None = None,
    timezone: str | None = "UTC",
) -> pd.DataFrame:
    """Fetch normalized OHLCV data for ``symbol`` within a time window.

    Parameters
    ----------
    symbol:
        Instrument identifier (e.g., ``"NQ"``). Passed through to the data
        source.
    start, end:
        Datetime bounds interpreted by the underlying source. The placeholder
        assumes UTC and treats ``end`` as exclusive.
    source:
        Concrete data provider. Defaults to :class:`PlaceholderDataSource` until
        a vendor API is wired in.
    timezone:
        Target timezone for the resulting ``DatetimeIndex``. Use ``None`` to
        keep timestamps timezone-naive.
    """

    provider: RawDataSource = source or PlaceholderDataSource()
    raw_df = provider.fetch(symbol, start, end)
    raw_df = _validate_and_normalize_columns(raw_df)

    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True)
    df = raw_df.sort_values("timestamp").set_index("timestamp")

    if timezone:
        df = df.tz_convert(timezone)
    else:
        df.index = df.index.tz_localize(None)

    return df
