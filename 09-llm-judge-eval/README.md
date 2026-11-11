# Project 09: Pairwise LLM judge audit

**Status: built on simulated answers, annotators and a simulated judge with planted biases. The decision gate FAILED, and that is reported as is.** Results are reproducible offline. They describe this simulated setup, not GPT-4, Claude, or any real judge. No model was called.

## Decision context
A team wants to use an LLM to pick the better of two answers, instead of paying humans, so it can rank candidate systems. The decision is whether to trust that judge, and how. The gate is fixed in `llm_judge_eval/config.py` before results are seen.

## What you learn
- How to measure agreement with humans, and why the human consensus is itself wrong about one pair in ten.
- How to estimate slot bias and length bias with a logistic fit, and how to put an interval on them.
- Why calling both orders and breaking splits with a coin does nothing, and what to do with the split instead.
- How a verbose system climbs the ranking under a length-biased judge.
- Why a fixed-slot or length-confounded evaluation set flatters a judge.
- How big a human-labelled set you need before an agreement number means anything.

## Data
Simulated. See `data/DATA_CARD.md`. To audit your own judge, see "Use it on real judge calls" below.

## Headline results (full run, test set)
- Single call in random order: agreement with the human consensus 0.761 (0.746 to 0.776). The gate asked for a lower bound of 0.80. A perfect judge scores 0.898 here, because the consensus errs.
- Triage (keep verdicts where both orders agree, send splits to humans): 0.873 agreement on the 72.7% of pairs kept.
- A single-call judge ranks the verbose system first of six. Its true rank is third.
- Planted slot bias 0.40 and length bias 0.50 are recovered as 0.43 and 0.50 with the true quality, and 0.35 and 0.55 using the human label as the quality stand-in.

Every number is in `results/metrics.json`; `reports/` renders them.

## The traps
Measured and written to `results/metrics.json` under `traps`: the human label is not truth, a fixed-slot evaluation, a length-confounded evaluation set, eval set size, and a verbose system over-ranked.

## Process note (read this)
- **Design change after a first look.** My first draft counted `swap` (both orders, coin on splits) as a headline candidate. A fast smoke run showed it matching a single call. That is an identity, not luck: a coin on splits keeps the same expected agreement as one call in a random order. I replaced it in the headline set with `swap_calibrated` (uses the split as a feature) and reported triage separately. I did not touch the planted biases, the 0.80 floor or the 5% limit. The fast run was not a release result.
- **Dev tie.** `single_order` and `swap_calibrated` tied on dev (0.7807). The rule breaks ties by list order, which chose `single_order`. On test `swap_calibrated` scored 0.782 against 0.761. The verdict is FAIL either way, because the lower bound for `swap_calibrated` is 0.767, below 0.80. I did not change the rule.
- **The calibrated judge is not a fix.** It passes the verbose-system check at the default settings, but the sensitivity sweep shows it over-rates the verbose system when the judge has no length bias (V beats S4 in 0.58 of prompts, true value below 0.5). The calibrator learns that longer answers are better on average, because length tracks quality here, and extrapolates to a system far outside that range. At the default length bias the two errors happen to cancel.
- The sweep and the triage result were planned, not added to rescue the gate. Neither changes the verdict.

## Not done
- A real model or real human labels. The real run is designed, not built: use the audit module.
- Self-preference, prompt sensitivity, multi-turn or rubric judging.
- Human length or style preference.
- Cost modelling for human labelling.

## Run

```bash
pip install -r ../requirements.txt
python -m llm_judge_eval.train --write-checksum   # first time only
python -m llm_judge_eval.train                    # writes results/ (a few seconds)
python -m llm_judge_eval.report                   # renders reports/ from results/
python -m pytest                                  # tests; no network needed
```

`--fast` uses fewer bootstrap draws for smoke tests. Its output is marked `fast_mode: true` and the release gate rejects it. Run these commands from this folder.

## Use it on real judge calls
1. For each pair of answers X and Y, call your judge twice: X first, then Y first.
2. Write one row per call with columns `pair_id, order, judge_prefers_first, human_prefers_x, len_x, len_y` (details in `llm_judge_eval/audit.py`).
3. Run `python -m llm_judge_eval.audit judgments.csv`.

You get slot bias, split rate, agreement with intervals, the triage coverage, and a length slope. Treat the length slope with care: it uses the human label as the quality stand-in, which overstated the planted value in this project's simulation. You need at least 30 pairs to run it, and far more to certify anything.

## Files

| Path | Purpose |
|---|---|
| `llm_judge_eval/data.py` | Simulated pairs, annotators, judge calls, fingerprint |
| `llm_judge_eval/judge.py` | Single call, coin swap, calibrated swap, triage, reference deciders |
| `llm_judge_eval/metrics.py` | Wilson, kappa, bootstrap intervals, bias regression |
| `llm_judge_eval/audit.py` | Command line audit for real judge calls |
| `llm_judge_eval/train.py` | Candidates, dev selection, test audit, traps, system ranking, sweep, plot |
| `llm_judge_eval/report.py` | Renders the model card and decision memo from `results/` |
| `tests/` | Hand-computed metrics, simulator mechanisms, judge logic, audit input checks, determinism, reproduction |
| `results/` | `metrics.json`, `error_analysis.json`, `agreement.png` |
| `reports/` | `model_card.md`, `decision_memo.md` |

## Source
Zheng et al., 2023, "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (arXiv 2306.05685): strong judges reach over 80% agreement with humans, comparable to human-human agreement, and show position, verbosity and self-enhancement biases. The 0.80 floor in the gate is that convention, adopted here as an assumption.
