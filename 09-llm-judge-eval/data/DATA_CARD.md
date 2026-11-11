# Data card: simulated answer pairs, annotators and judge calls

## Source
None. Every pair, annotator vote and judge call is drawn by `llm_judge_eval/data.py` from a model whose mechanisms are set by the code. No download, no personal data, no real model output, no real annotator.

## Why simulated
To audit a judge you need to know which answer is really better, and what bias the judge has. With real data you only have noisy human labels, so you cannot tell judge error from label error. With a simulation you can. The cost is realism, listed below. This is simulated data with a known truth, not test fixtures: tests use small draws, while `results/` come from the full run. The data source is named `simulated_known_truth`.

## Terms
CC-BY-4.0, same as the rest of this repository's data and documentation.

## Model
- Each answer has a true quality `q` (normal, mean set per system) and a standardised length `l` with correlation 0.3 to `q`. A system can have a length offset.
- A pair is two answers, X and Y. The true winner is the higher `q`.
- Three annotators each prefer X with probability `sigmoid(2.5 * (qx - qy))`. They have no slot or length preference. The consensus is the majority.
- The judge is called twice per pair, once with X first and once with Y first, as independent draws. It picks the first slot with probability `sigmoid(1.5 * quality_gap + 0.4 + 0.5 * length_gap)`, seen from the first slot. The 0.4 is a slot bias and the 0.5 a length bias.
- Dev and test are independent draws of 1,500 and 3,000 pairs. The six-system ranking uses 2,000 fresh prompts per system pair.

## What is assumed, not observed
Every number above. The slot bias, length bias, human sharpness, judge sharpness and the length-quality correlation are choices to make the mechanisms easy to read. They are not estimates of any real judge. The sensitivity table in the model card varies the two biases.

## Reproducibility
`data/CHECKSUMS.txt` holds a SHA-256 of a small fixed draw. `python -m llm_judge_eval.train` refuses to run if the generator no longer produces it. It does not hash the full simulation.

## Known quirks
- Judge calls are independent given the pair. A real judge at temperature zero repeats itself, so swapping orders helps less against a deterministic bias than it does here.
- Humans have no length preference. Real annotators often do.
- Quality is one number. Real answers differ in correctness, style and safety at once.
- No ties are modelled in the human labels.

## Not checked
- Any real model, real annotator or real benchmark.
- Self-preference, prompt sensitivity, or multi-turn judging.

## Using your own data
`python -m llm_judge_eval.audit judgments.csv` audits real judge calls. The file format is in the module docstring. `python -m llm_judge_eval.audit --example example.csv` writes a simulated file in that format. CSV files are not committed to this repository.
