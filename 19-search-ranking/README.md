# Project 19: Search ranking and position debiasing

**Status: built on simulated search impressions and click logs. The decision gate PASSED, and that is reported as is.** Every number in the reports comes from `results/metrics.json`.

Part of the [E-commerce, media and consumer apps](../domains/ecommerce-media.md) domain guide.

## Decision context
An e-commerce search engineering team must decide whether to ship a new search ranking model to replace the existing production baseline (a BM25 lexical match and quality heuristic). The person acting on the decision is the Head of Search or VP of Engineering. A wrong decision risks deploying a model that overfits to position-biased historical clicks, which appears to improve offline click metrics on head queries while degrading true customer search relevance and cannibalizing long-tail catalog items. The decision gate is fixed in `search_ranking/config.py` before any result is seen.

## What you learn
- Why logged search clicks are severely distorted by examination position bias (users click top positions regardless of relevance).
- Why a naive ranking model trained on raw clicks appears to win offline on click metrics, but its gain completely vanishes when evaluated on true unbiased relevance.
- How Inverse Propensity Scoring (IPS) uncouples position examination from document attraction to train debiased rankers.
- Why popularity bias causes naive models to degrade tail and niche queries by promoting irrelevant generic items.
- How to evaluate candidate rankers using counterfactual offline IPS metrics and paired-query bootstrap confidence intervals.

## Data
Simulated search queries, candidate item impressions, and position-biased clicks with known true relevance ground truth.

**Data storage best practice:** Following industry best practices for algorithmic evaluation, no synthetic CSV impression or click files are stored in git. Instead, the reproducible data generator in `search_ranking/data.py` synthesizes search sessions, candidate items, position examination curves, and clicks deterministically in memory from fixed random seeds. This ensures zero external download dependencies, prevents repository bloat, and provides exact known ground truth. See `data/DATA_CARD.md` for generation details and parameter specifications.

## Models compared
| Model | Ranking Strategy |
|---|---|
| Production Baseline | BM25 lexical match (65%) and merchant quality rating (35%) |
| Naive Click Ranker | Point-wise logistic ranker trained directly on raw unweighted logged clicks |
| IPS Debiased Ranker | Point-wise logistic ranker trained with Inverse Propensity Scoring (Headline) |

## The traps, measured
1. **Position bias decay in logged clicks:** Clicks are overwhelmingly concentrated at top positions due to human eye-tracking examination drop-off, with the top 3 positions capturing over 85% of total clicks while bottom positions receive under 6%.
2. **Offline gain that vanishes once position bias is removed:** Naive rankers trained on raw clicks achieve an apparent offline gain on click metrics, but this gain completely vanishes and reverses into negative territory on true relevance once position bias is removed.
3. **Tail query degradation masked by head queries:** High-volume head queries mask severe ranking degradation on tail and niche queries, where naive models boost irrelevant popular distractors.

## Process note (read this)
- The decision gate rule was set in `search_ranking/config.py` before results were generated: (1) True Unbiased NDCG@10 lift over production baseline is at least +0.030 (with 95% bootstrap CI lower bound strictly positive); (2) True NDCG lift on tail queries is non-negative (>= 0.000); (3) offline IPS-counterfactual NDCG lift directionally agrees with true unbiased NDCG lift.
- The decision gate PASSED. The candidate IPS debiased ranker achieved a True NDCG@10 lift of +0.0425 (95% bootstrap CI: [+0.0383, +0.0459]), strictly exceeded the +0.030 threshold, achieved +0.0255 lift on tail queries without degradation, and matched directional sign on offline IPS counterfactual evaluation (+0.0034).
- In contrast, the Naive Click Ranker failed: its true relevance lift vanished (-0.0024) and it severely degraded tail queries (-0.0223).
- Weakest claims:
  1. Known position propensity: The simulation assumes the true examination decay curve (eta = 1.35) is known; live production systems must estimate propensities via randomized position interleaving or dual learning algorithms.
  2. Single-click cascade assumption: Real search behavior exhibits multi-click sessions, query reformulation, and satisfaction abandonments that single-click models simplify.
  3. Linear scoring hypothesis: Real ranking systems employ deep cross-encoders or gradient boosted decision trees with high-order feature interactions, which may introduce additional propensity estimation variance.

## Not done
- No real-time contextual bandit exploration.
- No session-level multi-click query reformulation modeling.
- No cross-device user personalization graph.

## Run

```bash
pip install -r ../requirements.txt                     # Windows PowerShell: pip install -r ..\requirements.txt
python -m search_ranking.train --write-checksum        # writes data/CHECKSUMS.txt and results/
python -m search_ranking.report                        # renders reports/ from results/
python -m pytest                                       # tests; no network needed
```

`python -m search_ranking.train --fast` runs a rapid smoke test. Its output is marked `fast_mode: true` and the release gate rejects it.

## Files

| Path | Purpose |
|---|---|
| `search_ranking/config.py` | Constants, simulation parameters, pre-set decision gate rule |
| `search_ranking/data.py` | Search session simulator, candidate generator, click cascade, validation |
| `search_ranking/models.py` | Production baseline, Naive click ranker, IPS debiased ranker |
| `search_ranking/metrics.py` | DCG@K, NDCG@K, IPS counterfactual metric, bootstrap confidence intervals |
| `search_ranking/train.py` | Pipeline execution, model fitting, gate evaluation, trap measurement, plots |
| `search_ranking/report.py` | Renders model card and decision memo from results |
| `tests/` | Unit tests for data validation, metric edge cases, traps, and reproduction |
| `results/` | `metrics.json`, `error_analysis.json`, `ranking_comparison.png` |
| `reports/` | `model_card.md`, `decision_memo.md` |
