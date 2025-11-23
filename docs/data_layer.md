# Data Layer

The data layer focuses on preparing consistent OHLCV time series that can feed
pattern detection, temporal analysis, and the forecast engine. The initial
building blocks cover three needs:

## CSV ingestion

``load_ohlcv_csv`` in ``src/data_layer/fetch_raw_data.py`` reads vendor CSV
exports, normalizes column names, and enforces the required OHLCV schema. The
loader ensures timestamps are timezone-aware (UTC by default) so downstream
modules can combine multiple feeds without ambiguity.

## Timeframe resampling

``resample_ohlcv`` in ``src/data_layer/resample_timeframes.py`` converts a
``DatetimeIndex`` data set to any pandas offset alias while preserving OHLCV
semantics (first/last/high/low/sum). Optional forward-filling keeps auxiliary
columns aligned with the resampled bars.

## Session and event enrichment

``add_session_features`` in ``src/data_layer/enrich_with_sessions_and_events.py``
adds contextual columns that will be used by temporal and forecast engines:
regular-trading-hours vs. extended-hours labels, weekday metadata, and a simple
flag for macro event days.
