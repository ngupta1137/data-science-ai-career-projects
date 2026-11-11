# Data card: simulated A/B experiments

## Source
None. Every experiment is drawn by `experiment_analysis/data.py` from a model whose true effect is set by the code. No download, no personal data, no third-party content.

## Why simulated
A real experiment never tells you the true effect, so you cannot count how often an analysis is wrong. A simulation can. The project's results are about how methods behave under stated assumptions. They are not facts about any real product.

This data is simulated, not synthetic test fixtures: tests use small draws, while `results/` come from the full simulation described here. The release gate rejects data sources named "synthetic" for that reason, and this project uses `simulated_known_truth`.

## Terms
CC-BY-4.0, same as the rest of this repository's data and documentation.

## Model
- Pre-period metric `x` and noise `e`, each with mean 0 and variance 1, normal or lognormal (sigma 1, heavy tailed and skewed).
- Outcome `y = rho * x + sqrt(1 - rho^2) * e + effect * treated`. Control variance is 1, so effects are in standard deviations.
- Two arms of equal size, random assignment, independent users.
- For the multiple-metric study: independent normal metrics.

## Reproducibility
`data/CHECKSUMS.txt` holds a SHA-256 of a small fixed draw of the generator. `python -m experiment_analysis.train` refuses to run if the generator no longer produces it. This detects changes to the generator or its random number behaviour. It does not hash the full simulation.

## Known quirks
- Users are independent. Real traffic has repeat users, bots, and shared devices.
- Effects are constant across users. Real effects vary.
- The pre-period metric is always observed. Real ones are often missing.

## Not checked
- Behaviour with missing covariates, unequal arm sizes, or unequal variances.
- Any real experiment dataset.
