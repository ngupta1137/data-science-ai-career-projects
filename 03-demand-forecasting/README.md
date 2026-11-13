# Project 03: Demand forecasting

**Status: built on Bike Sharing (UCI). The decision gate FAILED, and that is reported as is.** Every number in the reports comes from `results/metrics.json`.

## Decision context
A weekly plan for how many bikes to position and how many staff to schedule. The person acting on it is an operations planner who needs each day of the next week forecast one week in advance. Over-forecasting wastes bikes and staff. Under-forecasting loses rentals. The decision gate is fixed in `demand_forecast/config.py` before any result is seen.

## What you learn
- Why a forecast is judged by a rolling-origin backtest, not a random split.
- What "known at forecast time" means, and how to build features that respect it.
- MAE, RMSE, sMAPE, and MASE, and what each hides.
- Why a block bootstrap is the right interval when days within a week are correlated.
- How target leakage hides inside columns that look innocent.

## Data
Daily bike sharing counts for two years, via the UCI repository. Not committed. See `data/DATA_CARD.md` for terms, quirks, and what was not checked.

## Backtest design
Forecast 7 days ahead. At each origin the model is fitted on all earlier days and forecasts the next 7. The first half of the origins tunes hyperparameters and picks the headline model and the best baseline. The second half is the test, scored once. Test origins are all later than tuning origins.

## The traps
`demand_forecast/train.py` measures three and writes them to `results/metrics.json` under `traps`:
1. **Same-day components.** `casual + registered` equals the target, so using them reads the answer.
2. **Short lags.** Lag 1 and lag 2 are not known 7 days ahead.
3. **Shuffled folds.** A random K-fold trains on days after the day being scored.

## Models
Three baselines (repeat the last value, repeat the same weekday last week, mean of the last 28 days), ridge regression, and histogram gradient boosting. Features are calendar facts and past demand at least 7 days old. A test checks that changing the last 6 days of demand does not change any feature.

## Process note (read this)
- The decision gate failed on the real dataset: headline ridge regression achieved a test MAE of 984.63, which did not reliably beat the 28-day rolling mean baseline (test MAE 961.02), yielding a 95% bootstrap interval on the MAE gap spanning zero (-95.9 to 41.6). In accordance with repository rules, this result is reported as is without post hoc tuning.

## Not done
- No weather inputs. Weather is not known 7 days ahead, and a real system would need a weather forecast.
- No prediction intervals. Point forecasts only.
- No hierarchical or multi-location forecasting.
- No cost model for over- and under-forecasting.
- No deep learning or classical ARIMA/ETS models. The aim is evaluation design, not model breadth.

## Run

```bash
pip install -r ../requirements.txt
python -m demand_forecast.download_data --write-checksum   # first time only, then commit data/CHECKSUMS.txt
python -m demand_forecast.train                             # writes results/
python -m demand_forecast.report                            # renders reports/ from results/
python -m pytest                                            # tests; no network needed
```

`python -m demand_forecast.train --fast` runs one configuration per model for smoke tests. Its output is marked `fast_mode: true` and the release gate rejects it.

## Files

| Path | Purpose |
|---|---|
| `demand_forecast/data.py` | Download, checksum, validation including date continuity |
| `demand_forecast/features.py` | Horizon-safe features, plus two deliberately leaky variants for the traps |
| `demand_forecast/metrics.py` | MAE, RMSE, sMAPE, MASE, rolling origins, block bootstrap |
| `demand_forecast/models.py` | Baselines, ridge, gradient boosting, search grids |
| `demand_forecast/train.py` | Backtest, tuning, test scoring, traps, error analysis, plot |
| `demand_forecast/report.py` | Renders the model card and decision memo from `results/` |
| `tests/` | Hand-checked metrics, a no-leakage perturbation test, validation, end-to-end smoke test |
| `results/` | `metrics.json`, `error_analysis.json`, plot |
| `reports/` | `model_card.md`, `decision_memo.md` |
