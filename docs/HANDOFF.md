# Handoff: work left to do

Status as of 2026-11-12. Read [AGENTS.md](../AGENTS.md) first; its rules apply to every task below. Do the tasks in order. One branch and one pull request per task.

## Where things stand

| Item | State |
|---|---|
| Projects 01 to 04 | Built: code and tests. Not run: no metrics, because the build machine could not reach the dataset hosts |
| Projects 05 to 14 | Built, with committed results. Gates: 05, 06, 12, 13 passed; 07, 08, 09, 10, 11, 14 failed and say so |
| Project 15 credit scorecard | In progress on branch `wip/15-credit-scorecard`. Code, results, reports and tests exist. Gate failed. README, data card and wiring missing |
| Projects 16 to 26 | Planned only (table below) |
| LICENSE | Holds a marker, not the CC-BY-4.0 text. The only release-gate failure |
| Zenodo, release tag, DOI | Not done. Owner only |

## Task 1: run projects 01 to 04 on real data

On a machine with internet access, for each of `01-house-price-regression`, `02-fraud-detection`, `03-demand-forecasting` and `04-text-classification`:

1. Read its README and `data/DATA_CARD.md`. Download with the commands in the project README; the code checks the data against `data/CHECKSUMS.txt` once it exists. If a download, URL or checksum fails, stop and report it. Never edit an expected checksum to make a run pass.
2. Verify the dataset reference in the data card against the source and remove "Reference to verify before release" only if it checks out.
3. Run `train` then `report`. Commit `results/` and `reports/`, set `STATUS` to `built`, update the README status line and the root README row with the gate verdict.
4. Do not change any gate, model list or metric in `config.py`. If the code fails on the real data, fix the bug, say what you fixed in the process note, and rerun.

## Task 2: finish project 15 on its branch

```bash
git fetch origin && git checkout wip/15-credit-scorecard
```

1. Write `15-credit-scorecard/README.md` following `14-out-of-stock-detection/README.md`. Status line: simulated, gate FAILED, reported as is. Domain: banking-insurance. Process note must say: no model on recorded features beat the old policy because the old policy used an applicant trait the data never recorded; recalibration on the random sample beat the naive model; the test set knows every outcome, which no lender has.
2. Write `15-credit-scorecard/data/DATA_CARD.md` from `credit_score/config.py` and `credit_score/data.py`, following project 14's card.
3. In `credit_score/report.py`, the "What this means" paragraph says the recalibrated model "brings the predicted default rate within the band". Make that clause depend on `checks["calibrated_on_approved"]` so it cannot go stale.
4. Set `STATUS` to `built`, rerun `train` and `report`, wire it (pyproject, then `tools/wire_domain_project.py` with domain `banking-insurance`), run `make check`, open a pull request into `main`.

## Task 3: build projects 16 to 26

Same standard as 13 and 14: a simulation with a known truth, a headline method and gate fixed and committed before the first full run, three or more measured traps, error analysis by segment, model card and decision memo from results. Full run under about 30 seconds. One pull request per project, or per domain pair.

| # | Folder | Domain | Decision | Trap to measure |
|---|---|---|---|---|
| 16 | `16-claims-frequency-severity` | banking-insurance | Price a policy by segment | Ignoring exposure time; loss ratio by decile; one model for frequency and size |
| 17 | `17-marketing-mix-model` | marketing-growth | Where to move budget | Adstock and saturation ignored; correlated spend inflates one channel's return |
| 18 | `18-multi-touch-attribution` | marketing-growth | Which channel gets credit | Last touch against the true incremental effect from the simulator |
| 19 | `19-search-ranking` | ecommerce-media | Ship the new ranker? | Clicks biased by position; offline gain that vanishes once position bias is removed |
| 20 | `20-price-elasticity` | ecommerce-media | Set the price | Price changed in response to demand, so a naive fit gets elasticity wrong |
| 21 | `21-clinical-risk-score` | healthcare | Which patients to flag for follow-up | High AUC with poor calibration; net benefit by threshold (decision curve) |
| 22 | `22-survival-analysis` | healthcare | Time-to-event estimates | Treating censored patients as survivors or dropping them |
| 23 | `23-forecast-to-order` | operations-supply-chain | How much to stock | Ordering the mean forecast instead of the cost-optimal quantile |
| 24 | `24-predictive-maintenance` | operations-supply-chain | When to inspect | Features that leak the failure window; alert lead time against false alarms |
| 25 | `25-agent-tool-use-eval` | enterprise-genai | Can the agent act without a human check? | One-run success against reliability over repeated runs; unsafe actions per task |
| 26 | `26-llm-routing` | enterprise-genai | Which model answers which query | Average quality hides the failures that matter; cost against quality frontier |

Healthcare projects must state in the README and data card that no patient data is used and nothing is clinical or regulatory guidance. Projects 25 and 26 simulate agents and models; they must say no language model was called. A real-model run may be added later as a separate, labelled task.

## Task 4: LICENSE text

Replace the marker in `LICENSE` with the full, unmodified Creative Commons Attribution 4.0 International legal code from creativecommons.org. Then `python tools/check_release.py` (without `--ci`) should pass.

## Owner only

Switch the repository on in Zenodo, tag `v0.1.0`, put the DOI in README and `CITATION.cff`, run `python tools/check_release.py --post`.
