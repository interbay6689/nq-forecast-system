"""Utilities for loading raw OHLCV data files.

These helpers provide a thin wrapper around ``pandas`` CSV loading so that
all downstream stages receive data with a consistent schema and timezone
handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, MutableMapping

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


def _validate_and_normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(DEFAULT_COLUMN_MAP.values()) - set(df.columns)
    if missing:
        raise MissingColumnsError(
            f"CSV is missing required columns: {', '.join(sorted(missing))}"
        )
    ordered_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    return df[ordered_columns]


def load_ohlcv_csv(
    path: str | Path,
    *,
    column_map: MutableMapping[str, str] | None = None,
    timezone: str | None = "UTC",
) -> pd.DataFrame:
    """Load OHLCV data from a CSV file into a normalized ``DataFrame``.

    Parameters
    ----------
    path:
        Location of the CSV file. ``FileNotFoundError`` is raised when the path
        does not exist.
    column_map:
        Optional mapping from CSV column names to the canonical schema. This is
        useful when working with vendor-specific column names.
    timezone:
        Timezone to convert the timestamp index to. The CSV is assumed to be
        UTC; set to ``None`` to leave the timestamps timezone-naive.

    Returns
    -------
    pandas.DataFrame
        ``DataFrame`` indexed by timestamp with columns ``open, high, low,
        close, volume``.
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
