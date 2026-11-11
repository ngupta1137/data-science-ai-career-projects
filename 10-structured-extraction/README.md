# Project 10: Structured extraction from invoices

**Status: built on generated invoices with known field values. The decision gate FAILED, and that is reported as is.** Results are reproducible offline. They describe this generated setup, not real invoices. No LLM was called.

## Decision context
A team wants to turn invoices into structured records (vendor, number, date, total, currency) and post them without review. The decision is whether to switch the extractor on for a customer whose layout was not in the training data. The gate is fixed in `extraction_eval/config.py` before results are seen.

## What you learn
- Scoring by canonical value, so `March 15, 2024` equals `2024-03-15`.
- Document-level accuracy against field-level scores, and why you report both.
- A failure taxonomy: missing, unparseable, a named distractor (subtotal, due date, customer), or another value.
- Why a wrong value that passes the schema is worse than no value.
- Why a random split overstates how a model will do on a new customer's format.
- How OCR noise sets a ceiling no extractor can beat.

## Data
Generated. See `data/DATA_CARD.md`.

## Headline results (full run, test sets)
- Learned span ranker, seen templates: all five fields right on 100% of documents (95% interval 0.997 to 1.000). Do not read this as solved: the documents are cleaner than real ones.
- The same extractor on two unseen templates: 31.3% (0.288 to 0.340). On unseen templates it returned 715 wrong values and every one of them passes a schema check.
- Hand-written rules: 52.5% on seen templates (they cover only the common label variants), and they abstain on unseen templates instead of guessing.
- Training on two seen templates and testing on the third: mean 0.553, against 1.000 for a random split.
- Post hoc: 10 labelled documents from a new template took it from 0.0 to 1.0 (template E) and 0.627 to 0.963 (template G). More documents from seen templates changed nothing for the unseen ones.

Every number is in `results/metrics.json` and `results/error_analysis.json`; `reports/` renders them.

## The traps
Measured and written to `results/metrics.json` under `traps`: field scores hiding document errors, unseen templates, random split versus a held-out template, silent errors, OCR noise against its ceiling, and labelled documents needed.

## Process note (read this)
- The gate failed on the unseen-template check and I did not change it. The seen-template check passes with a margin that is itself a warning about the generator.
- The first candidate generator split vendor names only on `: ` and ` | `, which made the vendor field unreachable for template E by construction. I noticed it when a test asserting that every true value is readable from the text failed. I added a split on runs of spaces. This changed which candidates exist, not any threshold or rule, and it was done before the full run. With it, template E's vendor failure is a learned error (the model takes the header line), not a missing candidate.
- The hand-written rules abstain on vendor for unrecognised templates. An earlier draft fell back to the first line, which guessed wrong. I changed that before any full run because a rule set that guesses is a worse baseline than one that abstains.
- The label-variant study and the new-template study are labelled post hoc where they appear. The new-template study was added after the learning curve showed more seen-template data did not help; it cannot rescue the gate.
- The learning curve is flat on unseen templates because the failures are structural (position and label words), not because the model is saturated. The 50-document and 2,000-document models produced identical outputs on the unseen test set.

## Not done
- An LLM extractor with a JSON schema. Designed, not built; the scoring code applies to its output unchanged.
- Layout or image models, real invoices, line-item extraction, calibrated confidence.
- A cost model for review versus error.

## Run

```bash
pip install -r ../requirements.txt
python -m extraction_eval.train --write-checksum   # first time only
python -m extraction_eval.train                    # writes results/ (about 30 seconds)
python -m extraction_eval.report                   # renders reports/ from results/
python -m pytest                                   # tests; no network needed
```

`--fast` uses smaller sets for smoke tests. Its output is marked `fast_mode: true` and the release gate rejects it. Run these commands from this folder.

## Files

| Path | Purpose |
|---|---|
| `extraction_eval/data.py` | Invoice generator, templates, distractors, OCR noise, fingerprint |
| `extraction_eval/normalize.py` | Canonical dates, amounts, currencies |
| `extraction_eval/extractors.py` | First-match baseline, hand-written rules, learned span ranker |
| `extraction_eval/metrics.py` | Scoring, precision and recall, error taxonomy, silent errors, ceiling |
| `extraction_eval/train.py` | Fit, score, gate, traps, post hoc study, plot |
| `extraction_eval/report.py` | Renders the model card and decision memo from `results/` |
| `tests/` | Hand-computed values, generator invariants, extractor behaviour, determinism, reproduction |
| `results/` | `metrics.json`, `error_analysis.json`, `doc_level.png` |
| `reports/` | `model_card.md`, `decision_memo.md` |
