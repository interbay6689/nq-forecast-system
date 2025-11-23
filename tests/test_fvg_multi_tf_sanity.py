import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.known_patterns.fvg_detector import (  # noqa: E402
    detect_fvg_for_timeframes,
)


def create_dummy_ohlcv_1m() -> pd.DataFrame:
    times = pd.date_range("2024-01-01 09:30:00", periods=30, freq="min", tz="UTC")
    data = {
        "open":   [100 + i * 0.5 for i in range(len(times))],
        "high":   [101 + i * 0.5 for i in range(len(times))],
        "low":    [ 99 + i * 0.5 for i in range(len(times))],
        "close":  [100 + i * 0.5 for i in range(len(times))],
        "volume": [100 + i * 10 for i in range(len(times))],
    }
    return pd.DataFrame(data, index=times)


def main() -> None:
    base_df = create_dummy_ohlcv_1m()
    print("[*] Base 1m OHLCV shape:", base_df.shape)

    timeframes = ["1m", "5m", "15m"]
    df_fvgs = detect_fvg_for_timeframes(base_df, timeframes, min_width=0.0)

    print("[*] Detected FVGs across timeframes:")
    print(df_fvgs.head())
    print("Total FVGs:", len(df_fvgs))


if __name__ == "__main__":
    main()
