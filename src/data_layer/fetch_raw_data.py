from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import pandas as pd


Timeframe = Literal["1m", "5m", "15m", "1H", "4H", "1D", "1W"]


def _normalize_timeframe(tf: str) -> str:
    """
    Convert human-readable timeframe to pandas offset alias.

    Examples
    --------
    "1m" -> "1T"
    "5m" -> "5T"
    "15m" -> "15T"
    "1H" -> "1H"
    "4H" -> "4H"
    "1D" -> "1D"
    "1W" -> "1W"
    """
    tf = tf.strip()
    if tf.endswith("m"):
        return tf[:-1] + "T"
    return tf


def load_ohlcv_file(
    path: str | Path,
    *,
    time_column: str = "timestamp",
    tz: Optional[str] = "UTC",
) -> pd.DataFrame:
    """
    Load OHLCV data from a CSV or Parquet file into a normalized DataFrame.

    Assumptions
    -----------
    - File contains at least: timestamp, open, high, low, close, volume
    - `timestamp` is in ISO format or numeric epoch (ms/sec)
    - Data is for a single instrument (e.g., NQ futures)

    Parameters
    ----------
    path:
        Path to CSV/Parquet file.
    time_column:
        Name of the timestamp column.
    tz:
        Timezone to localize the DateTimeIndex to (default: UTC).
        If None, no tz-localization will be applied.

    Returns
    -------
    pd.DataFrame
        Index: DatetimeIndex (sorted ascending)
        Columns: ["open", "high", "low", "close", "volume"]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    if time_column not in df.columns:
        raise KeyError(f"Expected time column '{time_column}' in data file")

    # Convert timestamp to datetime
    df[time_column] = pd.to_datetime(df[time_column], utc=True, errors="coerce")
    df = df.dropna(subset=[time_column]).copy()
    df = df.sort_values(time_column)

    df = df.set_index(time_column)

    if tz is not None:
        # If already tz-aware, convert; otherwise localize
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(tz)
        else:
            df.index = df.index.tz_convert(tz)

    # Normalize columns
    rename_map = {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }
    df = df.rename(columns=rename_map)

    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required OHLCV columns: {missing}")

    return df[required_cols]


def load_and_resample(
    path: str | Path,
    *,
    timeframes: list[Timeframe] | None = None,
    tz: Optional[str] = "UTC",
) -> dict[str, pd.DataFrame]:
    """
    Convenience helper:
    Load raw file and return a dict of resampled DataFrames per timeframe.

    Parameters
    ----------
    path:
        Path to CSV/Parquet file with OHLCV data.
    timeframes:
        List of timeframes like ["1m", "5m", "15m", "1H", "4H", "1D", "1W"].
    tz:
        Timezone for the DateTimeIndex.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are timeframe strings, values are OHLCV DataFrames.
    """
    from .resample_timeframes import resample_ohlcv  # local import

    if timeframes is None:
        timeframes = ["1m", "5m", "15m", "1H", "4H", "1D", "1W"]

    base_df = load_ohlcv_file(path, tz=tz)
    return {
        tf: resample_ohlcv(base_df, _normalize_timeframe(tf))
        for tf in timeframes
    }
