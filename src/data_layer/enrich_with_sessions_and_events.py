"""Add session and event annotations to intraday data."""

from __future__ import annotations

from datetime import date, time
from typing import Iterable, NamedTuple, Sequence

import pandas as pd

SESSION_OPEN = time(hour=9, minute=30)
SESSION_CLOSE = time(hour=16, minute=0)
DEFAULT_SESSION_TZ = "America/New_York"


class SessionWindow(NamedTuple):
    """Simple definition of an intraday session window."""

    label: str
    start: time
    end: time


DEFAULT_SESSION_WINDOWS: Sequence[SessionWindow] = (
    SessionWindow("Asia", time(hour=20, minute=0), time(hour=8, minute=0)),
    SessionWindow("London", time(hour=3, minute=0), time(hour=11, minute=30)),
    SessionWindow("NY", SESSION_OPEN, SESSION_CLOSE),
)


def _normalize_index_to_tz(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    if index.tz is None:
        localized = index.tz_localize("UTC")
    else:
        localized = index
    return localized.tz_convert(tz)


def _label_sessions(local_times: pd.DatetimeIndex, windows: Sequence[SessionWindow]) -> pd.Series:
    def label_for(ts: pd.Timestamp) -> str:
        for window in windows:
            if window.start <= window.end:
                in_window = window.start <= ts.time() < window.end
            else:
                in_window = ts.time() >= window.start or ts.time() < window.end
            if in_window:
                return window.label
        return "Off"

    return pd.Series([label_for(ts) for ts in local_times], index=local_times)


def _week_of_month(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(((index.day - 1) // 7) + 1, index=index)


def enrich_with_time_features(
    df: pd.DataFrame,
    *,
    session_tz: str = DEFAULT_SESSION_TZ,
    session_windows: Sequence[SessionWindow] = DEFAULT_SESSION_WINDOWS,
    macro_event_dates: Iterable[date] | None = None,
) -> pd.DataFrame:
    """Annotate OHLCV data with session labels and calendar context.

    The function leaves the input columns untouched and appends:

    - ``session``: Asia/London/NY labeling based on configurable local windows.
    - ``session_type``: ``RTH`` for NY regular hours, ``ETH`` otherwise.
    - ``day_of_week`` / ``weekday_name``
    - ``week_of_month`` / ``month``
    - ``is_macro_event_day``: ``True`` when the candle's calendar date matches a
      provided macro event date (e.g., CPI/FOMC).
    """

    if df.empty:
        return df.copy()

    localized_index = _normalize_index_to_tz(df.index, session_tz)
    session_labels = _label_sessions(localized_index, session_windows)
    rth_mask = (localized_index.time >= SESSION_OPEN) & (localized_index.time < SESSION_CLOSE)

    annotated = df.copy()
    annotated["session"] = session_labels.values
    annotated["session_type"] = pd.Series(rth_mask, index=annotated.index).map({True: "RTH", False: "ETH"})
    annotated["day_of_week"] = localized_index.weekday
    annotated["weekday_name"] = localized_index.day_name()
    annotated["week_of_month"] = _week_of_month(localized_index)
    annotated["month"] = localized_index.month

    if macro_event_dates:
        macro_dates = {pd.Timestamp(d).date() for d in macro_event_dates}
        date_index = pd.Index(localized_index.date)
        annotated["is_macro_event_day"] = pd.Series(
            date_index.isin(macro_dates), index=annotated.index
        )
    else:
        annotated["is_macro_event_day"] = False

    return annotated
