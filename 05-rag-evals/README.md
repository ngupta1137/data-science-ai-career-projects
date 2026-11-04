# Project 05: RAG with evals

**Status: built on a generated teaching corpus.** Numbers in `results/` and `reports/` are reproducible offline. They describe this corpus only. They do not say how a retrieval system behaves on real documents.

## Decision context
A support lead wants a bot that answers policy questions with a value (days, weeks, dollars) and says "I cannot answer that" when the policies do not cover the question. The risk is not a wrong retriever. It is a confident answer to a question that has no answer. The decision gate is fixed in `rag_eval/config.py` before any result is seen.

## What you learn
- Evaluate retrieval and answering separately, so you know which stage lost the point.
- Recall@k, MRR, and nDCG on answerable questions, with a random retriever as the floor.
- Why unanswerable questions must be in the test set, and why always refusing is a baseline.
- A failure taxonomy that names each loss: retrieval miss, stale source, wrong document, wrong span, false refusal, answered unanswerable.
- Why complexity must pay for itself: hybrid and LSA retrievers are compared to BM25 with a paired interval.

## Data
Generated, committed, and checksummed. See `data/DATA_CARD.md`. 41 documents and 110 questions across six question types, split evenly into dev and test.

## Pipeline
1. Dev questions choose BM25 parameters, LSA dimensions, the headline retriever, and the abstention threshold.
2. Retrievers: BM25 (own implementation), TF-IDF cosine, LSA (TF-IDF plus truncated SVD, a classical stand-in for dense retrieval), reciprocal rank fusion of BM25 and LSA, and a seeded random baseline.
3. Reader: a deliberately simple extractor. It picks the retrieved sentence with a value that best overlaps the question and returns the first value in it. Confidence is the share of question word weight found in that sentence.
4. Test questions are scored once. Correct means the right value for an answerable question, or a refusal for an unanswerable one.

## The traps
Measured on the test questions and written to `results/metrics.json` under `traps`: tuning the abstention threshold on test, never abstaining, and scoring answerable questions only.

## Read the result carefully
The gate compares the pipeline to always refusing. It is a low bar. The decision memo reports how many unanswerable questions the pipeline actually refused, and in the committed run that is almost none. The lesson of this project is that word overlap is a weak signal for "no answer", and the error analysis shows which questions fool it.

## Not done
- **No language model.** Generating answers and judging them with a model is designed, not built, and not claimed.
- No dense neural retriever. LSA is not a substitute for one on real text.
- No real documents, no latency or cost measurement, no chunking study.
- Question types with few test items (`numeric_detail`, `stale_version`) are anecdotes, not estimates.

## Run

```bash
pip install -r ../requirements.txt
python -m rag_eval.generate   # rewrites data/ (identical to what is committed)
python -m rag_eval.train      # writes results/
python -m rag_eval.report     # renders reports/ from results/
python -m pytest              # tests; no network needed
```

`python -m rag_eval.train --fast` runs a reduced search for smoke tests. Its output is marked `fast_mode: true` and the release gate rejects it. Run these commands from this folder.

## Files

| Path | Purpose |
|---|---|
| `rag_eval/topics.py` | The fictional policy topics and question templates |
| `rag_eval/generate.py` | Builds corpus, questions, and checksums deterministically |
| `rag_eval/data.py` | Loading, checksum check, validation |
| `rag_eval/retrieval.py` | BM25, TF-IDF, LSA, reciprocal rank fusion, random baseline |
| `rag_eval/reader.py` | Extractive reader and answer normalisation |
| `rag_eval/metrics.py` | Recall@k, reciprocal rank, nDCG, bootstrap interval |
| `rag_eval/train.py` | Selection on dev, test scoring, taxonomy, traps, gate, plot |
| `rag_eval/report.py` | Renders the model card and decision memo from `results/` |
| `tests/` | Hand-checked metrics, retrieval and reader behaviour, data checks, determinism |
| `results/` | `metrics.json`, `error_analysis.json`, `correctness_by_type.png` |
| `reports/` | `model_card.md`, `decision_memo.md` |
