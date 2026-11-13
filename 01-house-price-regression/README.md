# Project 01: House price regression

**Status: built on California Housing. The decision gate PASSED.** Every number in the reports comes from `results/metrics.json`.

## Decision context
A first-pass screen that estimates median house value for a block group. The person acting on it is an analyst deciding which areas need a closer look. A wrong estimate wastes review time or hides an area that needed attention. The decision gate is fixed in `house_price/config.py` before any result is seen.

## What you learn
- Why a baseline comes first, and how to tell whether a model beats it.
- Why a random split overstates performance on spatial data, and by how much.
- RMSE, MAE, and R2, and what each hides.
- How to put an honest interval on a test metric when rows are correlated.
- Where a model fails, by price band, income, and location.

## Data
California Housing, 1990 census block groups, via scikit-learn. Not committed. See `data/DATA_CARD.md` for terms, quirks, and what was not checked.

## Split design
Rows in the same 0.5 degree cell stay together in train or test. Cross-validation on the training rows uses the same grouping. The test rows are scored once, after the headline model is chosen by cross-validation.

## The trap
`house_price/train.py` refits the same model and hyperparameters on a random split and reports both test errors side by side. The measured gap is written to `results/metrics.json` under `leakage_trap`. The two test sets contain different rows, and the file says so.

## Models
Constant baselines (mean, median), ridge regression, and histogram gradient boosting. Ridge and boosting are tuned with a small grid under grouped cross-validation. Nothing is tuned on the test rows.

## Process note (read this)
- Removed generation of `test_predictions.csv` in `results/`. Repository rules and `check_release.py` block committing CSV data artifacts.

## Not done
- No time-based validation. The data is a single snapshot.
- No deep learning, no stacking, no heavy tuning. The point is evaluation, not leaderboard rank.
- No cost-weighted metric. No real decision costs exist for this teaching dataset.
- No fairness analysis. Do not use this data for decisions about people.

## Run

```bash
pip install -r ../requirements.txt
python -m house_price.download_data --write-checksum   # first time only, then commit data/CHECKSUMS.txt
python -m house_price.train                             # writes results/
python -m house_price.report                            # renders reports/ from results/
python -m pytest                                # tests; no network needed
```

`python -m house_price.train --fast` runs a tiny search for smoke tests. Its output is marked `fast_mode: true` and the release gate rejects it.

## Files

| Path | Purpose |
|---|---|
| `house_price/data.py` | Download, checksum, validation |
| `house_price/splits.py` | Spatial and random splits, overlap checks |
| `house_price/features.py` | Feature construction (no fitting, so no leakage) |
| `house_price/metrics.py` | RMSE, MAE, R2, cluster bootstrap, segment tables |
| `house_price/train.py` | Search, test scoring, trap, error analysis, plots |
| `house_price/report.py` | Renders the model card and decision memo from `results/` |
| `tests/` | Metric hand-checks, leakage, validation, end-to-end smoke test |
| `results/` | `metrics.json`, `error_analysis.json`, plots |
| `reports/` | `model_card.md`, `decision_memo.md` |
