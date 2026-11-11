# Project 14: Out-of-stock detection

**Status: built on a simulated chain where some shelves go empty while the system shows stock. The decision gate FAILED, and that is reported as is.** Results are reproducible offline. They describe this simulated setup, not any retailer.

Domain: [retail and consumer goods](../domains/retail-cpg.md).

## Decision context
A field team can check a fixed number of shelves a day. Which ones? The common rule, three days without a sale, is simple. The question is whether a value-ranked list does better, and whether it is good enough to go live. The decision gate is fixed in `oos_detect/config.py` before results are seen.

## What you learn
- Why one zero-day rule floods reps with slow-selling shelves and finds fast-selling empty shelves late.
- Turning a gap in sales into the chance the shelf is empty, with Bayes' rule.
- Ranking by expected lost margin against a visit cost, within a daily budget.
- Why an alarm that is usually right can still lose money.
- How to check whether a predicted chance is calibrated, and what overconfidence costs.

## Data
Simulated. See `data/DATA_CARD.md`.

## The traps
Measured in `results/metrics.json` under `traps`: zero sales read as zero demand, one rule for every shelf, and an alarm without value.

## Process note (read this)
- The gate failed on precision. The value-ranked list beats no visits and beats the zero-day rule, but most of its visits find a full shelf.
- I then added a post hoc calibration table, labelled as such in `results/error_analysis.json`. It shows the predicted chance of an empty shelf is far too high. It explains the failure and does not change it.
- The trap "zero sales read as zero demand" turned out small in this setup, and my correction for it overshot. The naive rate did slightly better. That is reported, not hidden.
- A fair criticism of the gate: a precision bar is not an economic test, since a low-precision visit can still pay. I kept it because precision is what a field team experiences, and changing it after seeing results would not be honest.

## Not done
- Real data or shelf audits.
- Weekday and promotion patterns, back-room stock, rep misses.
- A calibrated version of the headline policy. That is the next step named in the decision memo.

## Run

```bash
pip install -r ../requirements.txt
python -m oos_detect.train --write-checksum   # first time only
python -m oos_detect.train                    # writes results/ (a few seconds)
python -m oos_detect.report                   # renders reports/ from results/
python -m pytest                              # tests; no network needed
```

`--fast` uses fewer stores and resamples. Its output is marked `fast_mode: true` and the release gate rejects it. Run these commands from this folder.

## Files

| Path | Purpose |
|---|---|
| `oos_detect/data.py` | Simulated shelves with all random draws made up front, validation, fingerprint |
| `oos_detect/detect.py` | Zero runs, sales-rate estimates, chance a shelf is empty, policies, day-by-day simulation |
| `oos_detect/metrics.py` | Net value, precision, recall, store bootstrap |
| `oos_detect/train.py` | Policies, sales-rate bands, visit budget, post hoc calibration, plot |
| `oos_detect/report.py` | Renders the model card and decision memo from `results/` |
| `tests/` | Hand-computed runs, rates and posteriors, capacity, determinism, reproduction |
| `results/` | `metrics.json`, `error_analysis.json`, `net_value.png` |
| `reports/` | `model_card.md`, `decision_memo.md` |
