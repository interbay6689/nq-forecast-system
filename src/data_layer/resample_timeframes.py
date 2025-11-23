"""Resample OHLCV data across timeframes."""

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
