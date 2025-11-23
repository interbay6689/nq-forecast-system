"""Add session and event annotations to intraday data."""

from __future__ import annotations

from datetime import date, time
from typing import Iterable

import pandas as pd

SESSION_OPEN = time(hour=9, minute=30)
SESSION_CLOSE = time(hour=16, minute=0)
DEFAULT_SESSION_TZ = "America/New_York"


def _normalize_index_to_tz(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    if index.tz is None:
        localized = index.tz_localize("UTC")
    else:
        localized = index
    return localized.tz_convert(tz)


def add_session_features(
    df: pd.DataFrame,
    *,
    session_tz: str = DEFAULT_SESSION_TZ,
    macro_event_dates: Iterable[date] | None = None,
) -> pd.DataFrame:
    """Annotate OHLCV data with session labels and calendar context.

    The function leaves the input columns untouched and appends:

    - ``session``: ``RTH`` for the regular trading hours window, ``ETH`` otherwise.
    - ``weekday``: Integer day of week (Monday=0).
    - ``weekday_name``: Friendly weekday name.
    - ``is_macro_event_day``: ``True`` when the candle's calendar date matches a
      provided macro event date (e.g., CPI/FOMC).
    """

    if df.empty:
        return df.copy()

    localized_index = _normalize_index_to_tz(df.index, session_tz)
    session_mask = (localized_index.time >= SESSION_OPEN) & (localized_index.time < SESSION_CLOSE)

    annotated = df.copy()
    annotated["session"] = pd.Series(session_mask, index=annotated.index).map({True: "RTH", False: "ETH"})
    annotated["weekday"] = localized_index.weekday
    annotated["weekday_name"] = localized_index.day_name()

    if macro_event_dates:
        macro_dates = {pd.Timestamp(d).date() for d in macro_event_dates}
        date_index = pd.Index(localized_index.date)
        annotated["is_macro_event_day"] = pd.Series(
            date_index.isin(macro_dates), index=annotated.index
        )
    else:
        annotated["is_macro_event_day"] = False

    return annotated
