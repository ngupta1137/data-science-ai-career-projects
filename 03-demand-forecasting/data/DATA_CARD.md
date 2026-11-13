# Data card: Bike Sharing (daily)

## Source
The Bike Sharing dataset from the UCI Machine Learning Repository, daily file (`day.csv`), covering two years of a bike rental system (Capital Bikeshare, Washington D.C., 2011-2012). Hourly data exists in the same archive and is not used.
- Source page: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Reference: Fanaee-T, H., and Gama, J. "Event labeling combining ensemble detectors and background knowledge." Progress in Artificial Intelligence, 2(2), 2014, pp. 113-127.

## Terms
Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0) on the UCI repository. Verify the exact terms on the source page before redistributing anything derived from this data. The data is **not** committed: `python -m demand_forecast.download_data` fetches it and the code rejects the file if the committed checksum does not match.

## Shape
Documented by the source: 731 rows, one per day. `demand_forecast/config.py` enforces the row count and `demand_forecast/data.py` enforces the schema, non-negative counts, consecutive dates, and the identity below.

| Column | Meaning |
|---|---|
| dteday | Date |
| season, mnth, weekday | Calendar fields. weekday 0 is Sunday |
| holiday, workingday | Calendar flags |
| casual, registered | Rentals by user type |
| cnt (target) | Total rentals, equal to casual plus registered |
| weathersit, temp, atemp, hum, windspeed, yr, instant | Present in the file, not used as features |

## Known quirks
- **`cnt` equals `casual + registered`.** Using either component for the same day leaks the answer. The code checks the identity and uses it for the leakage trap.
- **Two years, one system.** Growth between the two years is strong, so a model that ignores level will drift. There is no way to test a third year.
- **Weather is observed, not forecast.** The weather columns are same-day observations. Using them would need a forecast 7 days ahead, which does not exist in this data, so they are excluded.
- **Holidays and workingday are calendar facts.** They are known in advance and are allowed.
- **Normalised weather fields.** The weather values in the file are scaled by the source. They are not used here.

## Not checked
- How rentals were counted, or whether system outages show as low counts.
- Whether the two years are comparable in fleet size.
- Any difference between user groups beyond the casual/registered split.
