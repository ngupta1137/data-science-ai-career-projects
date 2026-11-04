# Data card: Northwind Logistics policy corpus (generated)

## Source
Authored for this project by `rag_eval/generate.py`. Northwind Logistics is a fictional company. There is no external dataset, no download, and no real person or organisation in the text. The files are committed because they are small and fully reproducible: `python -m rag_eval.generate` rewrites them byte for byte, and a test fails if the committed files drift from the generator.

## Why generated
A real document set needs a download this project cannot guarantee stays available, and real questions need hand labelling. A generated corpus lets every number in `results/` be reproduced offline. The cost is realism, listed below.

## Terms
CC-BY-4.0, same as the rest of this repository's data and documentation. No third-party content.

## Files
| File | Content |
|---|---|
| `corpus.jsonl` | One document per line: `id`, `kind`, `topic`, `title`, `text` |
| `questions.jsonl` | One question per line: `id`, `type`, `question`, `gold_docs`, `answer`, `split` |
| `CHECKSUMS.txt` | SHA-256 of both files. `train.py` refuses to run on a mismatch |

## Documents
Three kinds. `policy`: the current policy, states the value. `faq`: names the topic, states no value, so it is a lexical decoy. `archive`: an older version with a different value, so it is a stale decoy. Some topics have two values.

## Question types
`lexical` (shares words with the policy), `paraphrase` (asks the same thing in other words), `numeric_detail` (asks for the second value), `stale_version` (asks for the current value while an archived page exists), `unanswerable_attribute` (names a real policy, asks for something it does not state), `unanswerable_offtopic` (no policy covers it). Dev and test alternate within each type.

## Known quirks
- Questions and documents come from the same templates, so word overlap is easier than on real text.
- The test set is small, and two types (`numeric_detail`, `stale_version`) have only a handful of test questions. Read those rows as examples, not estimates.
- `validate()` checks that every answer appears in its gold documents, never in a FAQ page, and that a stale answer is absent from the archive page.

## Not checked
- Whether any real company uses this wording. It was written to be generic and is not based on a real policy.
- Whether results here predict results on real documents. They do not, and the model card says so.
