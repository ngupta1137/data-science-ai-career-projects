# Data card: simulated model monitoring streams

## Source
None. Every stream is drawn by `drift_monitor/data.py` from a model whose drift is set by the code. No download, no personal data, no third-party content.

## Why simulated
A monitor can only be scored if you know when the world changed. Production data never says. A simulation gives the true onset, so the project can count false alarms, detections, and delays. The cost is realism, listed below. This is simulated data with a known truth, not test fixtures: tests use small draws, while `results/` come from the full run. The data source is named `simulated_known_truth`.

## Terms
CC-BY-4.0, same as the rest of this repository's data and documentation.

## Model
- Five numeric features, standard normal, independent. One categorical feature with three levels, shares 0.5, 0.3, 0.2.
- Binary outcome from a logistic model with fixed weights (see `config.py`). A logistic regression is trained on 20,000 reference rows and monitored. Its reference AUC comes from separate rows.
- One batch of 500 rows per week for 52 weeks. Drift starts at week 26 (0 indexed).

## Drift scenarios
`null`, `covariate_abrupt` (one feature shifts by 1.0 sd), `covariate_small` (0.3 sd), `covariate_gradual` (1.0 sd reached over 12 weeks), `variance_shift` (one feature's sd times 1.8), `category_shift` (category shares 0.2, 0.3, 0.5), `concept_drift` (the weight of one feature moves from 1.0 to -0.5 while its distribution is unchanged). All sizes are assumptions chosen to be easy to read, not estimates of real drift.

## Reproducibility
`data/CHECKSUMS.txt` holds a SHA-256 of small fixed draws of every scenario. `python -m drift_monitor.train` refuses to run if the generator no longer produces it. It does not hash the full simulation.

## Known quirks
- Features are independent and stationary before drift. Real features are correlated, seasonal, and noisy.
- Thresholds are calibrated on null streams drawn from the generator. A real team cannot do that and must resample its own reference history.
- Labels arrive on a fixed lag with no missing labels.

## Not checked
- Behaviour on correlated, high-dimensional, text, or image inputs.
- Any real production data.
