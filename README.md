# Data Science and AI Career Projects

End-to-end projects for people building a career in data science and AI. Each project ships with code, a data access script, tests, committed metrics, and error analysis, so you can reproduce every number and see where the model fails.

**Status: pre-release scaffold.** Nothing here has been released or archived yet. Projects 01 to 04 have code and tests but no generated metrics. Project 05 is built on a generated corpus with committed results. The table below says which is which.

<!-- DOI badge goes here after the first Zenodo release -->

## What this is

- Five reference projects, each built to the standard in [docs/evidence-standard.md](docs/evidence-standard.md).
- A baseline first, a leakage trap shown and fixed, error analysis by segment, and a decision memo for each.
- Self-study material. Run it, break it, compare your numbers to the committed ones.

## What this is not

- Not a course or a tutorial series.
- Not a leaderboard. The goal is a defensible result, not a high score.
- Not production code. Models are small and the pipelines are teaching-sized.
- Not a data host. Third-party datasets are downloaded by script and never committed.

## Projects

| # | Project | Skill it teaches | Status |
|---|---|---|---|
| 01 | [House price regression](01-house-price-regression) | Baselines, spatial leakage, RMSE/MAE/R2, residual analysis | Code and tests written. Metrics not yet generated |
| 02 | [Fraud detection](02-fraud-detection) | Imbalanced classification, PR-AUC, calibration, cost-based decisions, time-based splits | Code and tests written. Metrics not yet generated |
| 03 | [Demand forecasting](03-demand-forecasting) | Rolling-origin backtests, horizon-safe features, MASE | Code and tests written. Metrics not yet generated |
| 04 | [Text classification](04-text-classification) | TF-IDF baselines, macro-F1, metadata leakage, confusion analysis | Code and tests written. Metrics not yet generated. Transformer comparison designed, not built |
| 05 | [RAG with evals](05-rag-evals) | Retrieval recall@k, MRR, refusal, failure taxonomy | Built and run on a generated corpus. Results committed, describe that corpus only |

Roadmap, not part of the claim above: experiment analysis (power, CUPED, peeking) and a recommender. These are only added if projects 01 to 05 meet the bar.

## How to run

```bash
git clone https://github.com/ngupta1137/data-science-ai-career-projects.git
cd data-science-ai-career-projects
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

make check                             # lint, layout gate, tests (no network needed)
```

Individual steps: `make lint`, `make gate`, `make test`.

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
