import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.known_patterns.fvg_detector import detect_fvg, fvgs_to_frame  # noqa: E402


def create_dummy_ohlcv() -> pd.DataFrame:
    """
    Create a small OHLCV DataFrame with a known bullish and bearish FVG.

    We use 1-minute candles:
    - Bars 0..6

    Bullish FVG example (wick-based):
    - high[0] = 100
    - low[2]  = 106  > high[0] (100) -> bullish FVG

    Bearish FVG example:
    - low[3]  = 200
    - high[5] = 194  < low[3] (200)  -> bearish FVG
    """
    times = pd.date_range("2024-01-01 09:30:00", periods=7, freq="T", tz="UTC")

    data = {
        # indices: 0    1    2    3    4    5    6
        "open":  [100, 101, 105, 200, 201, 198, 197],
        "high":  [100, 103, 107, 205, 203, 194, 198],
        "low":   [ 99, 100, 106, 200, 199, 193, 195],
        "close": [101, 102, 106, 202, 200, 195, 196],
        "volume": [10, 12, 14, 20, 18, 16, 15],
    }

    df = pd.DataFrame(data, index=times)
    return df


def main() -> None:
    df = create_dummy_ohlcv()
    print("[*] Dummy OHLCV:")
    print(df)

    fvgs = detect_fvg(df, tf="1m", min_width=0.5)
    print(f"\n[*] Detected {len(fvgs)} FVG(s).")

    if fvgs:
        df_fvgs = fvgs_to_frame(fvgs)
        print("\n[*] FVG table:")
        print(df_fvgs)
    else:
        print("[!] No FVG detected – check logic.")


if __name__ == "__main__":
    main()
