from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence

import pandas as pd

from src.data_layer.resample_timeframes import resample_ohlcv


Direction = Literal["bull", "bear"]


@dataclass
class FVG:
    """
    Represents a single Fair Value Gap (FVG) detected on a price series.

    Definition used (wick-based, 3-candle logic):
    - Bullish FVG:
        low[i]   > high[i-2]
        Zone: [high[i-2], low[i]]
    - Bearish FVG:
        high[i]  < low[i-2]
        Zone: [high[i], low[i-2]]
    """

    id: int
    direction: Direction
    tf: str

    created_at: pd.Timestamp
    start_index: int
    end_index: int

    top: float
    bottom: float
    width: float

    is_filled: bool = False
    filled_at: Optional[pd.Timestamp] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "direction": self.direction,
            "tf": self.tf,
            "created_at": self.created_at,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "top": self.top,
            "bottom": self.bottom,
            "width": self.width,
            "is_filled": self.is_filled,
            "filled_at": self.filled_at,
        }


def _validate_ohlcv(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required OHLC columns: {missing}")


def detect_fvg(
    df: pd.DataFrame,
    *,
    tf: str = "",
    min_width: float = 0.0,
) -> List[FVG]:
    """
    Detect Fair Value Gaps (FVG) on a single timeframe.

    Parameters
    ----------
    df:
        OHLCV DataFrame with DatetimeIndex. Must contain:
        ['open', 'high', 'low', 'close'].
    tf:
        Timeframe label (e.g. "1m", "5m", "15m", "1H").
        Stored on each FVG for later aggregation.
    min_width:
        Minimum price width of the gap (top - bottom) to keep.
        Use 0.0 to keep all.

    Returns
    -------
    List[FVG]
        List of detected gaps in chronological order.
    """
    _validate_ohlcv(df)

    highs = df["high"].values
    lows = df["low"].values
    idx = df.index

    fvgs: List[FVG] = []
    next_id = 1

    # 3-candle logic: use i-2 and i
    # we start from i=2 so that i-2 >= 0
    for i in range(2, len(df)):
        # bullish FVG: low[i] > high[i-2]
        if lows[i] > highs[i - 2]:
            bottom = highs[i - 2]
            top = lows[i]
            width = float(top - bottom)
            if width >= min_width:
                fvgs.append(
                    FVG(
                        id=next_id,
                        direction="bull",
                        tf=tf,
                        created_at=idx[i],
                        start_index=i - 2,
                        end_index=i,
                        top=float(top),
                        bottom=float(bottom),
                        width=width,
                    )
                )
                next_id += 1

        # bearish FVG: high[i] < low[i-2]
        if highs[i] < lows[i - 2]:
            top = lows[i - 2]
            bottom = highs[i]
            width = float(top - bottom)
            if width >= min_width:
                fvgs.append(
                    FVG(
                        id=next_id,
                        direction="bear",
                        tf=tf,
                        created_at=idx[i],
                        start_index=i - 2,
                        end_index=i,
                        top=float(top),
                        bottom=float(bottom),
                        width=width,
                    )
                )
                next_id += 1

    return fvgs


def fvgs_to_frame(fvgs: List[FVG]) -> pd.DataFrame:
    """
    Convert a list of FVG objects into a pandas DataFrame.

    Index will be the FVG id.
    """
    if not fvgs:
        return pd.DataFrame(
            columns=[
                "id",
                "direction",
                "tf",
                "created_at",
                "start_index",
                "end_index",
                "top",
                "bottom",
                "width",
                "is_filled",
                "filled_at",
            ]
        ).set_index("id")

    records = [f.to_dict() for f in fvgs]
    df = pd.DataFrame.from_records(records)
    df = df.set_index("id").sort_index()
    return df


# ===========================
# Multi-timeframe utilities
# ===========================


def _normalize_tf_to_rule(tf: str) -> str:
    """
    Convert a human timeframe like '1m', '5m', '15m', '1H', '4H', '1D', '1W'
    to a pandas resample rule.

    We prefer 'min' instead of deprecated 'T'.
    """
    tf = tf.strip()
    if tf.endswith("m"):
        # '1m' -> '1min'
        return tf[:-1] + "min"
    return tf


def detect_fvg_for_timeframes(
    base_df: pd.DataFrame,
    timeframes: Sequence[str],
    *,
    min_width: float = 0.0,
) -> pd.DataFrame:
    """
    Detect FVGs for multiple timeframes from a base OHLCV DataFrame.

    Parameters
    ----------
    base_df:
        OHLCV DataFrame at a relatively low timeframe (e.g. 1m).
    timeframes:
        List of timeframe labels like ["1m", "5m", "15m", "1H"].
    min_width:
        Minimum width filter passed to `detect_fvg`.

    Returns
    -------
    pd.DataFrame
        Unified DataFrame of all detected FVGs across timeframes.
    """
    all_fvgs: list[FVG] = []

    for tf in timeframes:
        rule = _normalize_tf_to_rule(tf)

        # Resample if needed
        if rule != "1min":
            df_tf = resample_ohlcv(base_df, rule)
        else:
            df_tf = base_df

        fvgs_tf = detect_fvg(df_tf, tf=tf, min_width=min_width)
        all_fvgs.extend(fvgs_tf)

    return fvgs_to_frame(all_fvgs)


def save_fvgs_to_parquet(df_fvgs: pd.DataFrame, path: str | Path) -> None:
    """
    Save FVG DataFrame to Parquet file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_fvgs.to_parquet(path)


def load_fvgs_from_parquet(path: str | Path) -> pd.DataFrame:
    """
    Load FVG DataFrame from Parquet file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FVG file not found: {path}")
    return pd.read_parquet(path)
