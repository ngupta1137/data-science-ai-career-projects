# Project 13: Promotion lift

**Status: built on a simulated grocery chain with a known effect of the promotion. The decision gate PASSED, and the pass needs the caveats below.** Results are reproducible offline. They describe this simulated setup, not any retailer.

Domain: [retail and consumer goods](../domains/retail-cpg.md).

## Decision context
A category manager asks: the promotion sold more units, so do we run it again? The question is profit, not units. The decision gate is fixed in `promo_lift/config.py` before results are seen.

## What you learn
- Why a before-and-after view credits the season to the promotion.
- Why counting only the promoted item in promotion weeks misses cannibalisation and pull-forward.
- Difference-in-differences on total category profit with randomised control stores.
- Why the discount on units that would have sold anyway is often the biggest cost.
- How many stores a test needs before it gives a definite answer.
- How picking promotion stores by hand breaks the right method.

## Data
Simulated. See `data/DATA_CARD.md`.

## The traps
Measured in `results/metrics.json` under `traps`: before and after without control, counting only the promoted item, manager-picked stores, and units up while profit is down.

## Process note (read this)
- The headline method and the gate were fixed before the first run and not changed.
- The gate passed. But at 200 stores, the right method gave a definite answer in only about three in four simulated campaigns. This campaign happened to be one of them. Read the store-count table before planning a real test.
- The effect sizes were chosen so that units rise while profit falls. A setup with a profitable promotion would teach the same methods with the opposite answer.

## Not done
- Real data.
- Supplier funding, halo, competitor response, stock-outs, repeated promotions.
- Synthetic control or matched-store methods for when randomisation is not possible.

## Run

```bash
pip install -r ../requirements.txt
python -m promo_lift.train --write-checksum   # first time only
python -m promo_lift.train                    # writes results/ (a few seconds)
python -m promo_lift.report                   # renders reports/ from results/
python -m pytest                              # tests; no network needed
```

`--fast` runs fewer simulated campaigns. Its output is marked `fast_mode: true` and the release gate rejects it. Run these commands from this folder.

## Files

| Path | Purpose |
|---|---|
| `promo_lift/data.py` | Simulated stores and weeks, true profit and its parts, validation, fingerprint |
| `promo_lift/metrics.py` | Store profit, three estimators, Welch intervals, the call |
| `promo_lift/train.py` | Headline campaign, method study, manager-picked variant, store counts, plot |
| `promo_lift/report.py` | Renders the model card and decision memo from `results/` |
| `tests/` | Hand-computed estimators, simulator mechanisms, determinism, reproduction |
| `results/` | `metrics.json`, `error_analysis.json`, `weekly_sales.png` |
| `reports/` | `model_card.md`, `decision_memo.md` |
