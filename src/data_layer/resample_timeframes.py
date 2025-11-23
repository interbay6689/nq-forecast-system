"""Resample OHLCV data across timeframes.

The helpers here standardize aggregation rules for OHLCV bars and expose a
friendly wrapper for generating multiple timeframes at once.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def _ensure_datetime_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex for resampling")


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing)}")


def resample_ohlcv(
    df: pd.DataFrame, timeframe: str, *, include_non_price_columns: bool = True
) -> pd.DataFrame:
    """Resample OHLCV data to a new timeframe.

    Parameters
    ----------
    df:
        OHLCV DataFrame indexed by ``DatetimeIndex``.
    timeframe:
        Pandas offset alias (e.g., ``'5T'`` for 5 minutes, ``'1H'`` for hourly,
        ``'1D'`` for daily).
    include_non_price_columns:
        When ``True`` (default) additional columns are forward-filled to match
        the resampled index. Disable if you prefer the output to only contain
        OHLCV columns.
    """

    _ensure_datetime_index(df)
    _validate_columns(df)

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    ohlcv = df[REQUIRED_COLUMNS].resample(timeframe, label="right", closed="right").agg(agg)
    ohlcv = ohlcv.dropna(how="any")

    if include_non_price_columns:
        extra_columns: Iterable[str] = [col for col in df.columns if col not in REQUIRED_COLUMNS]
        if extra_columns:
            extras = (
                df[extra_columns]
                .resample(timeframe, label="right", closed="right")
                .ffill()
                .reindex(ohlcv.index)
            )
            ohlcv = ohlcv.join(extras)

    return ohlcv


def resample_timeframes(
    df: pd.DataFrame,
    timeframes: Iterable[str],
    *,
    include_non_price_columns: bool = True,
) -> dict[str, pd.DataFrame]:
    """Generate multiple resampled DataFrames from a base timeframe.

    Parameters
    ----------
    df:
        Base OHLCV data at its native granularity (expected to be minute bars in
        this project). Must be indexed by ``DatetimeIndex``.
    timeframes:
        Iterable of pandas offset aliases to produce (e.g., ``["5T", "15T",
        "1H", "1D"]``). The function returns a dictionary keyed by the alias.
    include_non_price_columns:
        Forward-fill non-price columns so derived timeframes remain enriched
        with any metadata (sessions, calendar fields, events).
    """

    return {
        timeframe: resample_ohlcv(
            df, timeframe, include_non_price_columns=include_non_price_columns
        )
        for timeframe in timeframes
    }
