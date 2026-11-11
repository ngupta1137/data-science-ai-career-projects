# Project 11: Recommender evaluation

**Status: built on a simulated catalogue, users with drifting taste, and clicks logged under a popularity-biased policy. The decision gate FAILED, and that is reported as is.** Results are reproducible offline. They describe this simulation, not any real service.

## Decision context
A personalised recommender is proposed to replace a popularity list. The team has a click log and wants to pick the model offline. The decision is whether offline NDCG can be trusted to choose which model ships. The gate is fixed in `recsys_eval/config.py` before results are seen.

## What you learn
- A temporal split: train on the past, validate on the next window, test on the one after.
- Ranking metrics (NDCG, recall, hit rate), per-user bootstrap intervals and paired comparisons.
- Why the logged clicks are not a neutral test set: they were chosen by an earlier policy.
- How to separate "offline metric" from "what users actually like", which only a simulation can do.
- Coverage and exposure inequality, not just accuracy.
- Why a random split and sampled metrics flatter models.

## Data
Simulated. See `data/DATA_CARD.md`.

## Headline results (full run, test window)
- Models ordered by observed NDCG@10: recent_popularity, item_knn, popularity, svd, svd_ips, oracle, random. By true like rate: oracle, svd_ips, svd, random, item_knn, recent_popularity, popularity. Spearman correlation between the two: -0.68.
- The oracle, which knows what users like, scores NDCG@10 0.035 against popularity's 0.107.
- With a logging policy that shows items evenly, the same correlation is +0.93.
- The model chosen on validation (item_knn) beats popularity on both observed NDCG@10 and true like rate, and is still below random on true like rate (0.211 against 0.237). The best candidate by truth is svd_ips (0.404), which validation ranked last.
- Popularity scores below random on true like rate (0.183 against 0.237). It is a selection effect: once a user's own clicks are removed, the popular items left are ones they were shown and passed on. A test in `tests/test_data.py` checks the mechanism.

Every number is in `results/metrics.json` and `results/error_analysis.json`; `reports/` renders them.

## The traps
Measured and written to `results/metrics.json` under `traps`: the observed metric rewarding the logging policy, a random split instead of a temporal one, sampled metrics, the cold-start ceiling, and an exposure sweep.

## Process note (read this)
- The gate has two checks that compare against popularity and one that asks whether the offline metric can be trusted. The first two pass and the third fails. Passing the first two says little, because popularity is a poor baseline here. I wrote the gate that way before seeing results and did not change it. A better gate would also require beating random on a true or randomised measure.
- My first draft of the sampled-metrics note and its test claimed sampling inflates every model's hit rate. The data showed item_knn at 0.473 sampled against 0.474 full. I corrected the claim to say poorly ranked models are inflated most, and the note now states whether the sampled ordering matched the full ordering (here it did). No model, window or gate rule was touched.
- The exposure sweep is a design feature, not a rescue. It is the only way to show that exposure, not model quality, drives the disagreement.
- The inverse-popularity exponent (0.5) was fixed before results and not tuned.
- Fast mode and the full run differ only in bootstrap draws and sweep points, so their model numbers are identical.

## Not done
- An online experiment or off-policy estimator. That is the real fix and it is designed, not built.
- Neural, sequential, or two-tower models; tuning of ranks and neighbourhoods.
- A cold-start model, a feedback loop, real data.
- A cost model for a like.

## Run

```bash
pip install -r ../requirements.txt
python -m recsys_eval.train --write-checksum   # first time only
python -m recsys_eval.train                    # writes results/ (a few seconds)
python -m recsys_eval.report                   # renders reports/ from results/
python -m pytest                               # tests; no network needed
```

`--fast` uses fewer bootstrap draws and sweep points. Its output is marked `fast_mode: true` and the release gate rejects it. Run these commands from this folder.

## Files

| Path | Purpose |
|---|---|
| `recsys_eval/data.py` | World, logging policy, click simulation, fingerprint |
| `recsys_eval/models.py` | Random, popularity, recent popularity, item-kNN, SVD, inverse-popularity SVD, oracle |
| `recsys_eval/metrics.py` | Top-k, NDCG, recall, hit rate, true like rate, sampled hit rate, coverage, Gini, bootstrap |
| `recsys_eval/train.py` | Windows, validation pick, test, gate, traps, sweep, error analysis, plot |
| `recsys_eval/report.py` | Renders the model card and decision memo from `results/` |
| `tests/` | Hand-computed metrics, model examples, simulator mechanisms, determinism, reproduction |
| `results/` | `metrics.json`, `error_analysis.json`, `observed_vs_true.png` |
| `reports/` | `model_card.md`, `decision_memo.md` |

## Source
Krichene and Rendle, 2020, "On Sampled Metrics for Item Recommendation", KDD 2020: sampled metrics can disagree with exact metrics about which model is better.
