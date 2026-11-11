# Project 07: Model monitoring and drift

**Status: built on simulated streams with a known truth. The decision gate FAILED, and that is reported as is.** Results are reproducible offline. They describe how monitors behave under the stated assumptions, not any real system.

## Decision context
An ML platform owner has a deployed model and must choose a weekly monitor. A false alarm trains people to ignore alerts. A miss lets a model degrade unseen. The decision gate is fixed in `drift_monitor/config.py` before results are seen.

## What you learn
- PSI and KS drift statistics, and what each one can and cannot see.
- Calibrating a threshold to a false alarm budget over a whole year, not per batch.
- Why fixed PSI cutoffs (0.1, 0.2) depend on batch size.
- Measuring detection rate and delay against a known onset.
- Concept drift: why monitoring inputs alone cannot see it, and what delayed labels buy you.

## Data
Simulated. See `data/DATA_CARD.md`.

## The traps
Measured and written to `results/metrics.json` under `traps`: fixed PSI cutoffs on small batches, a threshold moved across batch sizes, and a label-free monitor facing concept drift.

## Process note (read this)
- The first full run used 200 calibration streams. The headline monitor's false alarm rate on the test null streams was 0.073 (interval 0.049 to 0.109), so the pre-set limit of 0.10 on the upper bound failed.
- I raised the calibration set to 1,000 streams because calibration noise is a known contributor, and committed to keep whatever came out. It gave 0.070 (0.046 to 0.105): still a fail. The rule was not changed and the gate result stands.
- After seeing the failure I added a post hoc follow-up with 2,000 fresh null streams, labelled as such in the results. It is not part of the gate and cannot rescue it. It shows the achieved false alarm rate is above the 0.05 target for several monitors, which is the useful finding: calibration needs many null streams, and real teams do not have them.

## Not done
- Calibration by resampling a real reference history. Designed, not built.
- Retraining, adaptive references, seasonality.
- Text, image, or high-dimensional inputs.
- Any real production data.

## Run

```bash
pip install -r ../requirements.txt
python -m drift_monitor.train --write-checksum   # first time only
python -m drift_monitor.train                    # writes results/ (about 2 minutes)
python -m drift_monitor.report                   # renders reports/ from results/
python -m pytest                                 # tests; the reproduction test reruns the full pipeline
```

`--fast` uses a few streams for smoke tests. Its output is marked `fast_mode: true` and the release gate rejects it. Run these commands from this folder.

## Files

| Path | Purpose |
|---|---|
| `drift_monitor/data.py` | Population, drifting streams, monitored model, generator fingerprint |
| `drift_monitor/metrics.py` | PSI, KS, delayed-label AUC drop, Wilson interval |
| `drift_monitor/train.py` | Calibration, headline selection, test, sweep, traps, plot |
| `drift_monitor/report.py` | Renders the model card and decision memo from `results/` |
| `tests/` | Hand-computed statistics, scenario behaviour, evaluation logic, determinism, reproduction |
| `results/` | `metrics.json`, `error_analysis.json`, `detection_by_shift_size.png` |
| `reports/` | `model_card.md`, `decision_memo.md` |
