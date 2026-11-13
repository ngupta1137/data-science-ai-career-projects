# Changelog

## Unreleased

- Repository scaffold: license layout, citation metadata, CI (lint, gate, test), release gate with layout enforcement, evidence standard, repo layout document, Makefile, issue and PR templates, Dependabot.
- One top-level folder per project, one uniquely named Python package per project, one `pytest` run from the root.
- Fourteen reference projects: 01 to 04 on real public datasets, 05 to 14 on generated or simulated data.
- Project 01 (house price regression): built on California Housing. Decision gate passed.
- Project 02 (fraud detection): built on Credit Card Fraud dataset (OpenML 42175). Decision gate passed.
- Project 03 (demand forecasting): built on Bike Sharing dataset. Pre-set decision gate failed on rolling-origin backtest and reported as is.
- Project 04 (text classification): built on 20 Newsgroups. Decision gate passed. Transformer comparison is designed, not built.
- Project 05 (RAG with evals): built on a generated teaching corpus. Gate passed. Results, error analysis, model card, and decision memo are committed and reproducible offline.
- Project 06 (experiment analysis): built on simulated experiments with a known truth. Gate passed. The first full run failed one check from simulation noise; the headline A/A cell was raised to 20,000 experiments and the process is disclosed in the project README.
- Project 07 (model monitoring and drift): built on simulated streams. Pre-set gate failed on false alarm limit and reported as is, with a labelled post hoc follow-up.
- Project 08 (fairness and model risk audit): built on simulated applicants. Pre-set gate failed on true positive rate gap and reported as is, with a labelled post hoc study of dev set size.
- Project 09 (LLM judge audit): built on simulated judges and annotators with planted slot and length bias; no LLM was called. Pre-set gate failed and reported as is. A real-judge run is designed, not built.
- Project 10 (structured extraction): built on generated invoices and rule-based extractors; no LLM was called. Pre-set gate failed and reported as is.
- Project 11 (recommender evaluation): built on a simulated world with drift and popularity-biased exposure. Pre-set gate failed and reported as is.
- Domain guides (`domains/`): seven pages that route to the existing projects by domain, with what does not transfer. Index only; no new code or results. The release gate checks that links resolve and every project is covered.
- Project 12 (uplift modelling): built on simulated customers with a known effect. Gate passed in a well-powered setup; README states the limits.
- Project 13 (promotion lift): built on a simulated grocery chain. Pre-set gate passed; at 200 stores the right method gives a definite answer in about three campaigns in four, which README states.
- Project 14 (out-of-stock detection): built on a simulated chain. Pre-set gate failed on visit precision and reported as is, with a labelled post hoc calibration table.
- Agent rules (`AGENTS.md`), task handoff (`docs/HANDOFF.md`) and `tools/wire_domain_project.py` for wiring new domain projects.
