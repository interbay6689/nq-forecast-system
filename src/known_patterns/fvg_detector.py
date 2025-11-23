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
    Fair Value Gap object in a standardized "pattern" format.
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

    # שדות עתידיים/סטנדרטיים לטבלת דפוסים
    is_filled: bool = False
    filled_at: Optional[pd.Timestamp] = None

    # נוכל למלא בהמשך בפונקציית העשרה
    session: Optional[str] = None
    part_of_day: Optional[str] = None  # בוקר/צהריים/ערב
    distance_to_price: Optional[float] = None  # המרחק מהמחיר הנוכחי

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pattern_type": "FVG",
            "direction": self.direction,
            "tf": self.tf,
            "created_at": self.created_at,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "top": self.top,
            "bottom": self.bottom,
            "width": self.width,
            "price_range": (self.bottom, self.top),
            "is_filled": self.is_filled,
            "filled_at": self.filled_at,
            "session": self.session,
            "part_of_day": self.part_of_day,
            "distance_to_price": self.distance_to_price,
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
    Convert a list of FVG objects into a standardized patterns DataFrame.

    Index will be the FVG id.
    """
    if not fvgs:
        cols = [
            "id",
            "pattern_type",
            "direction",
            "tf",
            "created_at",
            "start_index",
            "end_index",
            "top",
            "bottom",
            "width",
            "price_range",
            "is_filled",
            "filled_at",
            "session",
            "part_of_day",
            "distance_to_price",
        ]
        return pd.DataFrame(columns=cols).set_index("id")

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


from src.data_layer.enrich_with_sessions_and_events import (
    enrich_with_time_features,
)


def _classify_part_of_day(hour_utc: int) -> str:
    """
    מחלק את היום לבוקר/צהריים/ערב על בסיס שעה ב-UTC.
    תוכל לכוונן לפי אזור זמן/סשנים בהמשך.
    """
    if 6 <= hour_utc < 12:
        return "morning"
    if 12 <= hour_utc < 18:
        return "noon"
    return "evening"


def enrich_fvgs_with_time_and_price(
    df_fvgs: pd.DataFrame,
    base_df: pd.DataFrame,
    *,
    tz: str = "UTC",
    last_price: Optional[float] = None,
) -> pd.DataFrame:
    """
    Enrich an FVG patterns DataFrame with time-based metadata and distance to current price.

    Parameters
    ----------
    df_fvgs:
        DataFrame as returned by `fvgs_to_frame`.
    base_df:
        OHLCV DataFrame (same instrument) used as reference for time features.
    tz:
        Timezone used for time feature extraction.
    last_price:
        Optional current price. If None, will use last close of base_df.

    Returns
    -------
    pd.DataFrame
        FVG DataFrame with added columns:
        ['session', 'part_of_day', 'distance_to_price']
    """
    if df_fvgs.empty or base_df.empty:
        return df_fvgs

    # מעשירים את ה-DF הבסיסי במאפייני זמן
    base_enriched = enrich_with_time_features(base_df, tz=tz)

    # מייצרים טבלה עם session וחלקי יום לפי created_at
    meta = base_enriched.copy()
    meta["hour_utc"] = meta.index.tz_convert("UTC").hour
    meta["part_of_day"] = meta["hour_utc"].apply(_classify_part_of_day)

    # נצריך רק עמודות רלוונטיות
    meta_cols = meta[["session", "part_of_day"]]

    # מצמידים לפי created_at
    df = df_fvgs.copy()
    df = df.join(meta_cols, on="created_at", how="left")

    # קביעת מחיר אחרון
    if last_price is None and not base_df.empty:
        last_price = float(base_df["close"].iloc[-1])

    if last_price is not None:
        def _distance(row) -> float:
            # אפשר להגדיר כמינימום מרחק מהטופ/בוטם
            if row["direction"] == "bull":
                # אם המחיר מעל ה-FVG, נמדוד מהטופ; אחרת מהבוטם
                ref = row["top"] if last_price >= row["top"] else row["bottom"]
                return float(last_price - ref)
            else:
                ref = row["bottom"] if last_price <= row["bottom"] else row["top"]
                return float(ref - last_price)

        df["distance_to_price"] = df.apply(_distance, axis=1)

    return df
