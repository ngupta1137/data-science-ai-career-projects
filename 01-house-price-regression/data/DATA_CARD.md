# Data card: California Housing

## Source
The California Housing dataset distributed with scikit-learn (`sklearn.datasets.fetch_california_housing`). It is built from the 1990 U.S. census, one row per census block group.

Original reference (verify before release): Pace, R. Kelley, and Ronald Barry. "Sparse Spatial Autoregressions." Statistics and Probability Letters, 33(3), 1997, pp. 291-297.

## Terms
Not stated in this repository. The data is **not** committed here. `python -m house_price.download_data` fetches it through scikit-learn, which verifies the archive against its own checksum. Check the scikit-learn documentation for the current terms before redistributing anything derived from it.

## Shape
Documented by scikit-learn: 20,640 rows, 8 numeric features, 1 target. `house_price/config.py` enforces the row count, and `house_price/data.py` enforces ranges and rejects missing values.

| Column | Meaning |
|---|---|
| MedInc | Median income in the block group (tens of thousands of 1990 dollars) |
| HouseAge | Median house age in the block group |
| AveRooms | Average rooms per household |
| AveBedrms | Average bedrooms per household |
| Population | Block group population |
| AveOccup | Average household members |
| Latitude, Longitude | Block group location |
| MedHouseVal (target) | Median house value, units of $100,000 |

## Known quirks
- **The target is censored.** Values are capped near 5.0 (about $500,000). The run writes the exact count of capped rows to `results/metrics.json`. No model can learn values above the cap.
- **Outliers in AveRooms and AveOccup.** Some block groups have extreme averages, usually because of few households. Tree models tolerate this. The ridge baseline does not, and its error shows it.
- **Block groups, not houses.** The target is a median over many houses. Do not read a prediction as the price of one house.
- **1990 prices.** Nothing here transfers to current markets.
- **Spatial correlation.** Nearby rows are similar. This is why the split is spatial. See `house_price/splits.py`.

## Not checked
- Whether the capped value was set consistently across all source regions.
- Whether features were derived from overlapping census tables.
- Any fairness or redlining effect embedded in 1990 housing values. Do not use this data for decisions about people.
