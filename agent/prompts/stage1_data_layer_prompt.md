# Stage 1 – Data Layer Prompt

## 🎯 Goal

לבנות שכבת נתונים אחידה (Data Layer) ל-NQ שתספק:

- נתונים היסטוריים נקיים ומסודרים
- טיימפריימים: 1m, 5m, 15m, 1H, 4H, Daily, Weekly
- שדות זמן ומטא־דאטה: session, day_of_week, week_of_month, month, וכו’

הכול בפורמט נוח לעיבוד (DataFrame / Parquet) לשימוש בשלבים הבאים.

---

## 📂 Target modules (Stage 1)

- `src/data_layer/fetch_raw_data.py`
- `src/data_layer/resample_timeframes.py`
- `src/data_layer/enrich_with_sessions_and_events.py`

---

## 🧠 Tasks for the Agent

1. **Design API**

   - להציע פונקציות ברורות לדאטה:
     - `fetch_raw_data(...) -> DataFrame`
     - `resample_timeframes(df, timeframes: list[str]) -> dict[str, DataFrame]`
     - `enrich_with_time_features(df) -> DataFrame`

2. **Implement basic versions**

   - לכתוב קוד לדוגמה שמניח שיש מקור נתונים (placeholder) שניתן להחליף בהמשך.
   - להקפיד על:
     - שימוש ב־pandas
     - docstrings ברורים
     - טיפול נכון ב-DateTimeIndex

3. **Document assumptions**

   - לציין אילו הנחות נעשו (למשל: timezone, מקור נתונים, שמות עמודות price/volume).

---

## 📝 Response format (expected from the agent)

בתשובה של הסוכן:

1. סקירה קצרה של העיצוב המוצע (בנקודות).
2. קטעי קוד עבור כל אחד מהקבצים:
   - `fetch_raw_data.py`
   - `resample_timeframes.py`
   - `enrich_with_sessions_and_events.py`
3. הערות על נקודות פתוחות/שאלות (אם יש).
