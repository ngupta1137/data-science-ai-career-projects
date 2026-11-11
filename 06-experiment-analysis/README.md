# Project 06: Experiment analysis

**Status: built on simulated experiments with a known truth.** Results are reproducible offline. They describe how analysis methods behave under the stated assumptions. They do not describe any real product.

## Decision context
A product analytics lead is choosing the default way the team analyses A/B tests. A wrong default ships changes that do nothing (false positives) or misses changes that work (low power). The decision gate is fixed in `experiment_analysis/config.py` before results are seen.

## What you learn
- What a false positive rate is, and how to measure it with A/A simulations.
- CUPED: why a pre-period metric shrinks variance by about `1 - rho^2`, and when it is invalid.
- Power and sample size: formula versus simulation.
- Peeking: why stopping at the first significant look inflates false positives, and what a correction costs.
- Multiple metrics: uncorrected, Bonferroni, and Holm, for both errors and power.
- Why heavy tails and small samples break a normal approximation.

## Data
Simulated, with a known true effect. See `data/DATA_CARD.md`.

## The traps
Measured and written to `results/metrics.json` under `traps`: peeking, ten metrics with no correction, skewed outcomes with few users, and a covariate that moves with treatment (CUPED then removes part of the real effect).

## Process note (read this)
The first full run failed gate check 1: the CUPED A/A false positive rate at the headline cell was 0.064, with an interval that excluded 0.05. With 2,000 simulated experiments the interval is about plus or minus one point, so the cell could not separate 0.05 from 0.06. I ran two separate diagnostic runs of 20,000 A/A experiments with different seeds. They gave 0.0495 and 0.0486, so the first number was simulation noise. I then raised the headline A/A cell to 20,000 experiments (`HEADLINE_AA_REPS`) and changed no rule. The failed draw is not in the committed results. The grid still flags a CUPED false positive rate above nominal at 200 users per arm, which looks real.

## Not done
- No sequential designs with alpha spending or always-valid intervals. Designed, not built.
- No sample ratio mismatch check.
- No interference, novelty effects, repeated measures, or ratio metrics.
- No real experiment data.

## Run

```bash
pip install -r ../requirements.txt
python -m experiment_analysis.train --write-checksum   # first time only
python -m experiment_analysis.train                    # writes results/ (about 15 seconds)
python -m experiment_analysis.report                   # renders reports/ from results/
python -m pytest                                       # tests; no network needed
```

`--fast` runs 200 experiments per cell for smoke tests. Its output is marked `fast_mode: true` and the release gate rejects it. Run these commands from this folder.

## Files

| Path | Purpose |
|---|---|
| `experiment_analysis/data.py` | Simulation of experiments and metrics, generator fingerprint |
| `experiment_analysis/analysis.py` | Welch difference, CUPED, interim looks, Bonferroni, Holm |
| `experiment_analysis/metrics.py` | Wilson and bootstrap intervals, sample size formula |
| `experiment_analysis/train.py` | All simulations, gate, traps, error analysis, plot |
| `experiment_analysis/report.py` | Renders the model card and decision memo from `results/` |
| `tests/` | Hand-computed values, estimator properties, determinism, reproduction of committed results |
| `results/` | `metrics.json`, `error_analysis.json`, `peeking_false_positives.png` |
| `reports/` | `model_card.md`, `decision_memo.md` |
