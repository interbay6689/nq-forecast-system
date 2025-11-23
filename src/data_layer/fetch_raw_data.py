from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import nasdaqdatalink
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


def get_nasdaq_api_key(env_var: str = "NASDAQ_API_KEY") -> str:
    """
    Read the Nasdaq Data Link API key from an environment variable.

    Parameters
    ----------
    env_var:
        Name of the environment variable holding the API key.

    Returns
    -------
    str
        API key string.

    Raises
    ------
    RuntimeError
        If the environment variable is not set.
    """
    key = os.getenv(env_var)
    if not key:
        raise RuntimeError(
            f"Nasdaq API key not found. Please set environment variable '{env_var}'."
        )
    return key


def fetch_nq_from_nasdaq(
    dataset_code: str,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    tz: Optional[str] = "UTC",
) -> pd.DataFrame:
    """
    Fetch NQ OHLCV data from Nasdaq Data Link.

    Examples
    --------
    dataset_code could be something like:
    - "CHRIS/CME_NQ1"   (continuous NQ future)
    - Or another symbol supported by Nasdaq Data Link.

    Parameters
    ----------
    dataset_code:
        Nasdaq Data Link dataset code (e.g. "CHRIS/CME_NQ1").
    start_date:
        Optional start date (YYYY-MM-DD).
    end_date:
        Optional end date (YYYY-MM-DD).
    tz:
        Target timezone for the index.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, columns: ["open", "high", "low", "close", "volume"]
    """
    api_key = get_nasdaq_api_key()
    nasdaqdatalink.ApiConfig.api_key = api_key

    raw = nasdaqdatalink.get(
        dataset_code,
        start_date=start_date,
        end_date=end_date,
    )

    df = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Last": "close",
            "Settle": "close",
            "Volume": "volume",
        }
    )

    if "Date" in df.columns:
        df.index = pd.to_datetime(df["Date"], utc=True)
    else:
        df.index = pd.to_datetime(df.index, utc=True)

    if tz is not None:
        df.index = df.index.tz_convert(tz)

    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required OHLCV columns from Nasdaq data: {missing}")

    df = df[required_cols].sort_index()

    return df
