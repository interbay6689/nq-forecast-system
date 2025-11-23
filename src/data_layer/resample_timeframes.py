from __future__ import annotations

import pandas as pd


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a new timeframe using pandas' resample.

    Parameters
    ----------
    df:
        DataFrame with DatetimeIndex and columns:
        ["open", "high", "low", "close", "volume"].
    rule:
        Pandas offset alias, e.g. "1T", "5T", "15T", "1H", "4H", "1D", "1W".

    Returns
    -------
    pd.DataFrame
        Resampled OHLCV data, with same columns and DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")

    ohlc_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    resampled = df.resample(rule, label="right", closed="right").agg(ohlc_dict)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])

    return resampled
