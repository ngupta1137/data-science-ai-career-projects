# Project 02: Fraud detection

**Status: built on Credit Card Fraud (ULB / OpenML). The decision gate PASSED.** Every number in the reports comes from `results/metrics.json`.

## Decision context
A screen that decides which card transactions go to a human reviewer. The person acting on it is a fraud operations lead. A missed fraud costs the transaction amount. A flag costs reviewer time whether or not the transaction is fraud. The decision gate is fixed in `fraud_detection/config.py` before any result is seen.

## What you learn
- Why accuracy is useless at a 0.17% fraud rate, and what to report instead (PR-AUC, recall at a review budget).
- Why the split must be by time, and how much a random split flatters the model.
- What calibration is, how to measure it when positives are rare, and why it matters only when a decision uses the probability.
- How to turn a score into a decision with an explicit cost model, and why the right rule can depend on the transaction amount.
- Why a threshold tuned on the test set is optimism, with the size of it measured.

## Data
Credit card transactions from European cardholders, via OpenML through scikit-learn. Not committed. See `data/DATA_CARD.md` for terms, quirks, and what was not checked.

## Split design
Time-ordered. The earliest rows fit the models, the next block chooses the model, fits the calibrator and picks the threshold, and the latest 30% is scored once.

## Decision rules compared
| Rule | How it decides |
|---|---|
| Flag nothing, flag everything | The two trivial policies every model must beat |
| Global threshold | Flag when the score is above one cutoff chosen on validation |
| Per-transaction expected cost | Flag when predicted fraud probability times the amount reaches the review cost |

Because a monotone calibration does not change a global threshold's ranking, calibration only changes the per-transaction rule. That is why the project measures both.

## The traps
`fraud_detection/train.py` measures three and writes them to `results/metrics.json` under `traps`:
1. Accuracy of a model that flags nothing.
2. Cost with a threshold tuned on the test set versus one tuned on validation.
3. PR-AUC on a random split versus the time split, same model and hyperparameters.

## Models
A constant baseline, logistic regression, and histogram gradient boosting. No class weights or resampling, because both distort the probabilities the cost rule uses.

## Process note (read this)
- OpenML dataset ID in `config.py` was updated from 1597 to 42175. OpenML 1597 strips the `Time` column, which caused `validate_frame` to fail. OpenML 42175 (https://www.openml.org/d/42175) preserves `Time` and contains the identical 284,807 transactions and 492 fraud cases.
- Independent verification against the original ULB Machine Learning Group source page (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): published figures are 284,807 transactions and 492 frauds. Committed `results/metrics.json` reports exactly 284,807 rows and 492 frauds (fraud rate 0.001727). The row count and fraud count match the published source figures exactly.
- Weakest claim: Row-level bootstrap intervals treat transactions as independent, ignoring clustering by cardholder or merchant, making confidence intervals artificially narrow. Furthermore, probability calibration and threshold selection rely on only 28 fraud events in the validation set, and the 79.4% cost saving assumes fixed review costs ($5) without analyst capacity limits or chargeback penalty fees.

## Not done
- No per-card or per-merchant features. The data is anonymised and has no card identifier.
- No drift monitoring and no retraining schedule. The test window is short.
- No analyst capacity limit on how many flags can be reviewed.
- No fairness analysis.
- The review cost is an assumption. Replace it with real costs before using the cost results for anything.

## Run

```bash
pip install -r ../requirements.txt
python -m fraud_detection.download_data --write-checksum   # first time only, then commit data/CHECKSUMS.txt
python -m fraud_detection.train                             # writes results/
python -m fraud_detection.report                            # renders reports/ from results/
python -m pytest                                            # tests; no network needed
```

`python -m fraud_detection.train --fast` runs a tiny search for smoke tests. Its output is marked `fast_mode: true` and the release gate rejects it.

## Files

| Path | Purpose |
|---|---|
| `fraud_detection/data.py` | Download, checksum, validation |
| `fraud_detection/splits.py` | Time split, order checks, random split for the trap |
| `fraud_detection/features.py` | Feature construction (Time and label excluded) |
| `fraud_detection/metrics.py` | PR-AUC, calibration, cost policies, exact threshold search, bootstrap |
| `fraud_detection/train.py` | Fit, calibrate, cost policies, traps, error analysis, plots |
| `fraud_detection/report.py` | Renders the model card and decision memo from `results/` |
| `tests/` | Hand-checked metrics, threshold search against brute force, split order, validation, end-to-end smoke test |
| `results/` | `metrics.json`, `error_analysis.json`, plots |
| `reports/` | `model_card.md`, `decision_memo.md` |
