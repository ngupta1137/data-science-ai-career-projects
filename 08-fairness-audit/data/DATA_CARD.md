# Data card: simulated applicants for a fairness audit

## Source
None. Every applicant is drawn by `fairness_audit/data.py` from a model whose mechanisms are set by the code. No download, no personal data, no real lender, employer, or person.

## Why simulated
An audit asks why a gap exists. With real data you can only guess. With a simulation you know which mechanism produced the gap, so you can check whether each fix does what it claims. The cost is realism, listed below. This is simulated data with a known truth, not test fixtures: tests use small draws, while `results/` come from the full run. The data source is named `simulated_known_truth`.

## Terms
CC-BY-4.0, same as the rest of this repository's data and documentation.

## Model
- Group A (75%) and group B (25%). The label `g` is 0 for A and 1 for B.
- A latent score `z`, normal, mean 0 for A and -0.4 for B, variance 1. The outcome `y` (repays) is Bernoulli with probability `sigmoid(1.5 * z)`.
- `x1 = z + noise`. `x2 = z + noise`, with noise sd 1.0 for A and 1.8 for B (a worse measurement for group B). `x3 = 0.8 * g + noise` (a proxy for group, no outcome information beyond group). `x4` is pure noise.
- Fit, dev, and test sets are independent draws of 6,000, 3,000, and 6,000 rows.

## What is assumed, not observed
The size of the base rate gap, the measurement noise gap, the proxy strength, the reward for a good approval, and the cost of a bad one. All are choices to make the mechanisms easy to read.

## Reproducibility
`data/CHECKSUMS.txt` holds a SHA-256 of a small fixed draw. `python -m fairness_audit.train` refuses to run if the generator no longer produces it. It does not hash the full simulation.

## Known quirks
- Two groups only, observed without error.
- The outcome label is treated as the truth. In real lending or hiring data it is shaped by past decisions.
- Group B has about 575 positives in the test set, which limits how tightly any group B rate can be measured. That is a feature of this audit, and the model card shows it.

## Not checked
- Intersectional groups, missing group labels, or label bias.
- Any real data.
