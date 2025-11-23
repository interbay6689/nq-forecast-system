from __future__ import annotations

import calendar
from typing import Optional

import pandas as pd


def _get_session(hour_utc: int) -> str:
    """
    Simple session mapping based on UTC hour.

    This can be adjusted later to be more precise.
    """
    # Approximate sessions, adjustable later
    if 0 <= hour_utc < 7:
        return "Asia"
    if 7 <= hour_utc < 13:
        return "London"
    if 13 <= hour_utc < 22:
        return "NewYork"
    return "AfterHours"


def enrich_with_time_features(
    df: pd.DataFrame,
    *,
    tz: Optional[str] = "UTC",
) -> pd.DataFrame:
    """
    Add time-based features to an OHLCV DataFrame.

    Features:
    - session: Asia/London/NewYork/AfterHours
    - day_of_week: 0-6 (Mon=0)
    - day_name: string name
    - week_of_month: 1-5
    - month: 1-12
    - month_name: string name

    Parameters
    ----------
    df:
        DataFrame with DatetimeIndex.
    tz:
        Optional timezone to convert index for feature calculation.
        Only affects derived features, not the underlying index.

    Returns
    -------
    pd.DataFrame
        Original columns + new feature columns.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")

    idx = df.index
    if tz is not None:
        # Work on a converted copy of the index for feature extraction
        idx = idx.tz_convert(tz)

    # Basic date components
    day_of_week = idx.dayofweek
    day_name = idx.day_name()
    month = idx.month
    month_name = idx.month_name()

    # Week of month: 1..5
    # e.g. 1–7 -> 1, 8–14 -> 2, etc.
    week_of_month = ((idx.day - 1) // 7) + 1

    # Session by UTC hour (using original index)
    hour_utc = df.index.tz_convert("UTC").hour
    session = [ _get_session(h) for h in hour_utc ]

    enriched = df.copy()
    enriched["session"] = session
    enriched["day_of_week"] = day_of_week
    enriched["day_name"] = day_name
    enriched["week_of_month"] = week_of_month
    enriched["month"] = month
    enriched["month_name"] = month_name

    return enriched
