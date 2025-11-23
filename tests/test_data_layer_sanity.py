from pathlib import Path

import pandas as pd

from src.data_layer.fetch_raw_data import load_ohlcv_file
from src.data_layer.resample_timeframes import resample_ohlcv
from src.data_layer.enrich_with_sessions_and_events import enrich_with_time_features


def _create_dummy_csv(path: Path) -> None:
    """
    Create a small dummy OHLCV CSV file for sanity checks.
    """
    data = {
        "timestamp": [
            "2024-01-01 09:30:00",
            "2024-01-01 09:31:00",
            "2024-01-01 09:32:00",
            "2024-01-01 09:33:00",
            "2024-01-01 09:34:00",
            "2024-01-01 09:35:00",
        ],
        "open":   [16000, 16001, 16002, 16005, 16003, 16004],
        "high":   [16002, 16003, 16005, 16006, 16005, 16007],
        "low":    [15998, 15999, 16000, 16002, 16001, 16003],
        "close":  [16001, 16002, 16004, 16004, 16004, 16006],
        "volume": [100,   120,   90,    150,   110,   130],
    }
    df = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    base_path = Path("data/raw")
    csv_path = base_path / "dummy_nq.csv"

    if not csv_path.exists():
        print(f"[*] Creating dummy file at {csv_path}")
        _create_dummy_csv(csv_path)

    print("[*] Loading OHLCV from CSV...")
    df = load_ohlcv_file(csv_path, time_column="timestamp", tz="UTC")
    print(df.head())
    print("Index type:", type(df.index))
    print("Columns:", df.columns.tolist())

    # Resample ל־2 דקות כדי לראות שהריסמפל עובד
    print("\n[*] Resampling to 2 minutes...")
    df_2m = resample_ohlcv(df, "2T")
    print(df_2m.head())

    # העשרת מאפייני זמן
    print("\n[*] Enriching with time features...")
    df_enriched = enrich_with_time_features(df_2m, tz="UTC")
    print(df_enriched.head())
    print("Extra cols:", [c for c in df_enriched.columns if c not in ["open","high","low","close","volume"]])


if __name__ == "__main__":
    main()
