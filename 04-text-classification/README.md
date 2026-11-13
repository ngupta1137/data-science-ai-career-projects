# Project 04: Text classification

**Status: built on 20 Newsgroups. The decision gate PASSED.** Every number in the reports comes from `results/metrics.json`.

## Decision context
A first-pass router that sends each post to the right topic area. The person acting on it is a community or support lead. A misrouted post reaches the wrong reviewers and waits. The decision gate is fixed in `text_classification/config.py` before any result is seen.

## What you learn
- Why TF-IDF with a linear model is the baseline any fancier text model must beat.
- Macro-F1 versus accuracy, and what per-class results and a confusion matrix show that a single score hides.
- How shortcut features inflate a score: headers and quoted replies identify the poster, not the topic.
- How fitting a vectorizer on test text leaks, and how little or how much it matters.
- Why an interval on a text metric treats documents as independent, and what that misses.

## Data
The 20 Newsgroups collection through scikit-learn: about 18,800 posts in 20 topics. Not committed. See `data/DATA_CARD.md` for terms, quirks, and what was not checked.

## Split design
The official split is by posting date, so the test set is later than the training set and is used as given. A stratified random share of the training set is held out to choose the model. The test set is scored once.

## The traps
`text_classification/train.py` measures two and writes them to `results/metrics.json` under `traps`:
1. **Metadata kept.** The same model on text with headers, footers, and quotes left in. The run also lists the model's top tokens per class and what share of them never appear in the cleaned text.
2. **Vectorizer fitted on test text.** Same classifier, same settings, vocabulary and IDF learned from train plus test.

## Models
A majority-class baseline, then TF-IDF with Naive Bayes, logistic regression, and a linear SVM. Models are chosen by validation macro-F1. The gate asks whether a heavier model beats Naive Bayes by an interval that excludes zero, so complexity has to pay for itself.

## Process note (read this)
- Weakest claim: The decision gate requires beating a 20-class majority baseline (macro-F1 0.0050), which is a very low bar. Naive Bayes was selected by validation macro-F1 (0.7647 vs 0.7568 for LinearSVC), but on the holdout test set LinearSVC attained a higher macro-F1 (0.6805 vs 0.6761). Additionally, semantically adjacent categories such as talk.religion.misc (F1 0.213) remain heavily confused with soc.religion.christian.

## Not done
- **No transformer comparison.** Fine-tuning a transformer needs a deep learning stack, a GPU for reasonable time, and model downloads. It is designed but not built here, and it is not claimed. It is the natural next step once this baseline has committed numbers.
- No probability calibration. LinearSVC does not give probabilities.
- No cost-weighted metric, since the cost of a misroute is not known.
- No test on text from another source or period.

## Run

```bash
pip install -r ../requirements.txt                     # Windows PowerShell: pip install -r ..\requirements.txt
python -m text_classification.download_data --write-checksum   # first time only, then commit data/CHECKSUMS.txt
python -m text_classification.train                             # writes results/
python -m text_classification.report                            # renders reports/ from results/
python -m pytest                                                # tests; no network needed
```

`python -m text_classification.train --fast` runs one configuration per model for smoke tests. Its output is marked `fast_mode: true` and the release gate rejects it.

## Files

| Path | Purpose |
|---|---|
| `text_classification/data.py` | Download both text versions, checksum, validation |
| `text_classification/splits.py` | Stratified fit and validation split, overlap check |
| `text_classification/models.py` | Majority baseline, three TF-IDF models, search grids |
| `text_classification/metrics.py` | Accuracy, macro-F1, per-class table, confusion pairs, bootstrap, top features |
| `text_classification/train.py` | Model choice, test scoring, gate, two traps, error analysis, plot |
| `text_classification/report.py` | Renders the model card and decision memo from `results/` |
| `tests/` | Hand-checked metrics, split and data checks, end-to-end smoke test |
| `results/` | `metrics.json`, `error_analysis.json`, confusion matrix plot |
| `reports/` | `model_card.md`, `decision_memo.md` |
