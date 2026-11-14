# Project 18: Multi-touch attribution

**Status: built on simulated customer journeys. The decision gate FAILED, and that is reported as is.** Every number in the reports comes from `results/metrics.json`.

## Decision context
A growth marketing team needs to decide how to allocate a $50,000 monthly acquisition budget across four advertising channels: Display, Social, Paid Search, and Email. The person acting on the attribution model is the Head of Growth or Chief Marketing Officer. A wrong allocation wastes ad budget on non-incremental touchpoints, starves top-of-funnel customer acquisition, or misattributes organic baseline conversions to paid ad campaigns. The decision gate is fixed in `touch_attribution/config.py` before any result is seen.

## What you learn
- Why heuristic last-touch attribution starves top-of-funnel introducer channels (Display) and over-credits bottom-of-funnel closers (Search, Email).
- How to implement a first-order Markov chain attribution model with counterfactual channel removal effects.
- How to measure true causal credit using counterfactual removal lifts in a simulated environment with known ground truth.
- Why a substantial fraction of conversions are organic baseline conversions that occur without any ad exposure.
- How to evaluate whether an algorithmic attribution model actually delivers higher conversion yield upon budget reallocation.

## Data
Simulated multi-touch customer journeys with known counterfactual incrementality.

**Data storage best practice:** Following industry best practices for causal and decision-evaluation systems, no synthetic CSV customer files are stored in git. Instead, the reproducible data generator in `touch_attribution/data.py` synthesizes customer journeys and counterfactual removal lifts deterministically in memory from fixed random seeds. This guarantees zero external download dependencies, prevents repository bloat, and provides exact known ground truth. See `data/DATA_CARD.md` for generation details and parameter specifications.

## Models compared
| Model | Attribution Logic |
|---|---|
| Last Touch | 100% of credit to the final touchpoint before conversion |
| First Touch | 100% of credit to the first touchpoint that initiated the path |
| Linear | Equal credit (1/K) divided across all touchpoints |
| Position Based | 40% to first, 40% to last, 20% divided across middle touches |
| Time Decay | Exponential decay prioritizing touches closer to conversion |
| Markov (Removal) | First-order absorbing Markov chain measuring channel removal effects (Headline) |
| True Causal | Counterfactual Shapley incrementality from simulator ground truth |

## The traps, measured
1. **Last-touch attribution bias:** Last touch starves top-of-funnel Display and over-credits bottom-of-funnel Email and Search, missing that Display touches initiate customer consideration.
2. **Ignoring organic baseline conversion:** Heuristic attribution assigns 100% of conversions to ad channels, ignoring that roughly 36% of converting customers would have converted organically with zero marketing touches.
3. **Linear attribution dilution:** Linear attribution spreads credit equally across all touchpoints (1/K), diluting decisive discovery and closing touches by over-crediting passive incidental touches.

## Process note (read this)
- The decision gate rule was set in `touch_attribution/config.py` before results were generated: (1) holdout channel share MAE below 0.05 (5 percentage points); (2) headline share MAE strictly beats naive last touch; (3) budget reallocation achieves at least 5% gain in true incremental conversions.
- The decision gate FAILED and is reported honestly as failed. The Markov model achieved a share MAE of 0.0571 (exceeding the 0.0500 threshold) and delivered a +3.50% reallocation gain (below the 5.0% requirement). While the Markov model correctly shifts budget toward top-of-funnel Display compared to last touch, its first-order memoryless assumption prevents it from fully matching true multi-channel synergistic counterfactuals.
- Weakest claims:
  1. First-order Markov chain assumption: Real customer journeys exhibit memory, lag delays, and path-dependent state transitions that violation of first-order Markov properties can distort.
  2. Static touch costs: The budget simulation assumes constant linear costs per touch, whereas live digital ad auctions experience rising marginal CPMs and CPCs as budget scales.
  3. Closed-world tracking: The simulation assumes perfect user tracking across all touches; real-world cross-device cookie loss and privacy controls disproportionately blind top-of-funnel attribution.

## Not done
- No cross-device identity stitching or probabilistic identity graphs.
- No real-time bidding auction curves.
- No multi-quarter brand equity buildup or creative wear-out modeling.

## Run

```bash
pip install -r ../requirements.txt                     # Windows PowerShell: pip install -r ..\requirements.txt
python -m touch_attribution.train --write-checksum     # writes data/CHECKSUMS.txt and results/
python -m touch_attribution.report                     # renders reports/ from results/
python -m pytest                                       # tests; no network needed
```

`python -m touch_attribution.train --fast` runs a rapid smoke test. Its output is marked `fast_mode: true` and the release gate rejects it.

## Files

| Path | Purpose |
|---|---|
| `touch_attribution/config.py` | Constants, channel costs, pre-set decision gate rule |
| `touch_attribution/data.py` | Customer journey simulator, transition probabilities, counterfactual truth |
| `touch_attribution/models.py` | Last touch, first touch, linear, position-based, time decay, Markov removal |
| `touch_attribution/metrics.py` | Share MAE, Spearman rank correlation, budget reallocation evaluation |
| `touch_attribution/train.py` | Pipeline execution, gate checks, trap measurements, plots |
| `touch_attribution/report.py` | Renders model card and decision memo from results |
| `tests/` | Unit tests for data generation, metrics, traps, and clean reproduction |
| `results/` | `metrics.json`, `error_analysis.json`, `attribution_comparison.png` |
| `reports/` | `model_card.md`, `decision_memo.md` |

