# Data Science and AI Career Projects

End-to-end projects for people building a career in data science and AI. Each project ships with code, a data access script, tests, committed metrics, and error analysis, so you can reproduce every number and see where the model fails.

**Status: pre-release scaffold.** Nothing here has been released or archived yet. Projects 01 to 15 are built with committed results. Projects 01 to 04 run on real public datasets, while projects 05 to 15 are built on generated or simulated data. The table below says which is which.

<!-- DOI badge goes here after the first Zenodo release -->

## What this is

- Fifteen reference projects, each built to the standard in [docs/evidence-standard.md](docs/evidence-standard.md).
- A baseline first, a leakage trap shown and fixed, error analysis by segment, and a decision memo for each.
- Self-study material. Run it, break it, compare your numbers to the committed ones.

## What this is not

- Not a course or a tutorial series.
- Not a leaderboard. The goal is a defensible result, not a high score.
- Not production code. Models are small and the pipelines are teaching-sized.
- Not a data host. Third-party datasets are downloaded by script and never committed.

## Projects

Prefer to start from your field? See [domains/](domains/README.md) for guides by domain: retail and consumer goods, banking and insurance, marketing and growth, e-commerce and media, healthcare, operations, and enterprise generative AI.

| # | Project | Skill it teaches | Status |
|---|---|---|---|
| 01 | [House price regression](01-house-price-regression) | Baselines, spatial leakage, RMSE/MAE/R2, residual analysis | Built on California Housing. Gate passed |
| 02 | [Fraud detection](02-fraud-detection) | Imbalanced classification, PR-AUC, calibration, cost-based decisions, time-based splits | Built on credit card fraud data. Gate passed |
| 03 | [Demand forecasting](03-demand-forecasting) | Rolling-origin backtests, horizon-safe features, MASE | Built on Bike Sharing data. Gate failed, reported as is |
| 04 | [Text classification](04-text-classification) | TF-IDF baselines, macro-F1, metadata leakage, confusion analysis | Built on 20 Newsgroups. Gate passed. Transformer comparison designed, not built |
| 05 | [RAG with evals](05-rag-evals) | Retrieval recall@k, MRR, refusal, failure taxonomy | Built and run on a generated corpus. Results committed, describe that corpus only |
| 06 | [Experiment analysis](06-experiment-analysis) | False positive rate, CUPED, power, peeking, multiple metrics | Built on simulated experiments with a known truth. Gate passed |
| 07 | [Model monitoring and drift](07-model-monitoring) | PSI and KS monitors, calibrated thresholds, detection delay, concept drift | Built on simulated streams. Gate failed, reported as is |
| 08 | [Fairness and model risk audit](08-fairness-audit) | Group metrics, unawareness trap, equal opportunity thresholds, utility cost | Built on simulated applicants. Gate failed, reported as is |
| 09 | [LLM judge audit](09-llm-judge-eval) | Position and length bias, agreement with humans, triage, calibrated deciders | Built on simulated judges and annotators, no LLM was called. Gate failed, reported as is |
| 10 | [Structured extraction](10-structured-extraction) | Field accuracy, silent errors, unseen templates, OCR noise ceiling | Built on generated invoices and rule-based extractors, no LLM was called. Gate failed, reported as is |
| 11 | [Recommender evaluation](11-recommender-eval) | NDCG, popularity bias, temporal split, sampled metrics, exposure | Built on a simulated world with drift. Gate failed, reported as is |
| 12 | [Uplift modelling](12-uplift-modelling) | Qini and AUUC, IPW profit, response-model trap, data needed | Built on simulated customers with a known effect. Gate passed, with caveats in the README |
| 13 | [Promotion lift](13-promotion-lift) | Difference-in-differences on category profit, cannibalisation, pull-forward, stores needed | Built on a simulated grocery chain. Gate passed, with caveats in the README |
| 14 | [Out-of-stock detection](14-out-of-stock-detection) | Empty-shelf chance from sales gaps, value-ranked visits under a budget, calibration | Built on a simulated chain. Gate failed, reported as is |
| 15 | [Credit scorecard](15-credit-scorecard) | Reject bias, random approval sample, recalibration, cost of learning | Built on simulated applicants with reject bias. Gate failed, reported as is |

Projects 05 to 15 run offline on data the code generates, because a simulation with a known truth is the only way to count how often a method is wrong. Their results describe those simulations, not real systems. Projects 13 onward are built for one domain each; [domains/](domains/README.md) groups them. Every project that failed its own pre-set decision gate says so in the status column above and in its decision memo.

Roadmap, not part of the claim above: a recommender. It is only added if the projects above meet the bar.

## How to run

### POSIX (Linux, macOS)

```bash
git clone https://github.com/ngupta1137/data-science-ai-career-projects.git
cd data-science-ai-career-projects
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

make check                             # lint, layout gate, tests (no network needed)
```

Individual steps: `make lint`, `make gate`, `make test`.

### Windows (PowerShell)

On Windows PowerShell where `make` is typically not installed:

```powershell
git clone https://github.com/ngupta1137/data-science-ai-career-projects.git
cd data-science-ai-career-projects
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

ruff check .
ruff format --check .
python tools/check_release.py --ci
python -m pytest
```

Working with a coding agent? [AGENTS.md](AGENTS.md) holds the rules every agent must follow here, and [docs/HANDOFF.md](docs/HANDOFF.md) holds the open tasks.

Each project folder has its own README with the commands to download data, train, and evaluate. Library versions are pinned in `requirements.txt` to the versions the code was tested on.

## Repository layout

One folder per project, the same shape every time. See [docs/repo-layout.md](docs/repo-layout.md) for the full tree and the rules the release gate enforces.

## Responsible use

These projects use public, historical datasets. Models trained on them are for learning. Do not use them to make decisions about real people or property. Each model card states its limits.

## Related repositories by the same author

- [enterprise-ai-decision-toolkit](https://github.com/ngupta1137/enterprise-ai-decision-toolkit)
- [agent-incident-atlas](https://github.com/ngupta1137/agent-incident-atlas)

## Licenses

Code is MIT (`LICENSE-CODE`). Documentation, reports, and authored data are CC-BY-4.0 (`LICENSE`). See `LICENSE-NOTE.md`.

## Cite

Use `CITATION.cff`. A Zenodo concept DOI will be added after the first release.

Author: Nitin Gupta, ORCID 0009-0003-2501-7378.
