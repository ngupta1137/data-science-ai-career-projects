# Project 12: Uplift modelling for an offer

**Status: built on simulated customers with a known effect of the offer. The decision gate PASSED, and the passing needs the caveats below.** Results are reproducible offline. They describe this simulated setup, not any real retailer.

## Decision context
Send a discount to everyone, to nobody, or to a chosen group. A model of who will buy is the wrong tool, because some people buy anyway and some buy less when offered. The decision gate is fixed in `uplift_eval/config.py` before results are seen.

## What you learn
- Uplift learners (single model, two models, class transformation) trained on a randomised trial.
- The uplift (Qini) curve and its area, and profit by inverse-probability weighting.
- Why a response model loses to an uplift model at the same budget.
- Why training on past targeted offers misleads, and why trial data matters.
- How much trial data each learner needs.
- Why absolute profit estimates from one sample run above the truth, and why to compare policies by paired differences.

## Data
Simulated. See `data/DATA_CARD.md`.

## The traps
Measured and written to `results/metrics.json` under `traps`: response model instead of uplift, optimistic profit estimates, offers that followed buying signals, how much data is needed, profit by share offered, and sensitivity to the cost of an offer.

## Process note (read this)
- The gate passed in a well-powered, strong-effect simulation. Do not read the pass as "uplift modelling works for you". The data-needed table shows that with 2,000 training customers the simple linear learner beat the headline learner, and that the four learners are close with 20,000.
- The headline is picked on dev among four near-equal learners. Their Qini intervals overlap, so the pick is partly luck.
- Every absolute profit estimate sat above its truth, because all estimates share one sample's outcomes. The true value on test rows is close to the population value, so this is not a bias in the test rows. I did not intend this when I set the gate, and I did not change the gate. The gate compares policies by paired difference, which cancels most of it.
- Training on observational offers lowered true profit for most learners. Use trial data.

## Not done
- Real data.
- Multiple offers, dose, timing, fatigue, long-run effects.
- A later randomised holdout, which is how true profit is checked in practice.
- Tuning of the learners.

## Run

```bash
pip install -r ../requirements.txt
python -m uplift_eval.train --write-checksum   # first time only
python -m uplift_eval.train                    # writes results/
python -m uplift_eval.report                   # renders reports/ from results/
python -m pytest                               # tests; no network needed
```

`--fast` uses smaller settings for smoke tests. Its output is marked `fast_mode: true` and the release gate rejects it. Run these commands from this folder.

## Files

| Path | Purpose |
|---|---|
| `uplift_eval/data.py` | Simulated customers, trial and observational variants, splits, validation, fingerprint |
| `uplift_eval/models.py` | Single-model, two-model and class-transformation learners, response model |
| `uplift_eval/metrics.py` | Qini and AUUC, IPW and true profit, bootstrap |
| `uplift_eval/train.py` | Candidates, dev selection, test audit, traps, plot |
| `uplift_eval/report.py` | Renders the model card and decision memo from `results/` |
| `tests/` | Hand-computed curves, simulator mechanisms, learners, determinism, reproduction |
| `results/` | `metrics.json`, `error_analysis.json`, `qini.png` |
| `reports/` | `model_card.md`, `decision_memo.md` |
