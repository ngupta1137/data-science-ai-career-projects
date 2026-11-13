# Project 15: Credit scorecard with reject bias

**Status: built on simulated loan applicants where the old policy saw an applicant trait the data never recorded. The decision gate FAILED, and that is reported as is.** Results are reproducible offline. They describe this simulated setup, not any real lender.

Domain: [banking and insurance](../domains/banking-insurance.md).

## Decision context
A lender wants to replace an existing approval policy with a machine learning scorecard built on past loan outcomes. But past outcomes are only observed for applicants who were approved (reject bias). The question is whether a new scorecard trained on past approvals and recalibrated on a small random approval sample can beat the old policy in net profit per applicant. The decision gate is fixed in `credit_score/config.py` before results are seen.

## What you learn
- Why fitting a model only on approved loans produces biased probabilities that underestimate default risk on rejected applicants.
- Why inverse probability weighting (IPW) cannot fix selection bias caused by unrecorded features.
- How a small random sample of approved applicants can be used to recalibrate scores from an accepted-only model.
- Measuring the cost of learning: the default losses incurred by approving high-risk applicants at random.
- Why evaluating a scorecard only on approved loans overstates performance and hides miscalibration.

## Data
Simulated. See `data/DATA_CARD.md`.

## The traps
Measured in `results/metrics.json` under `traps`:
- Evaluating on approved loans only: the model looks well calibrated on loans you can see, but severely under-predicts default on applicants it would newly approve.
- Weighting cannot fix what was not recorded: IPW corrects selection on recorded features, but cannot recover unrecorded traits used by the old policy.
- A small unbiased sample: training a model solely on the random sample avoids bias but suffers from high variance and poor ranking; recalibration combines the rank power of the larger biased sample with the calibration of the unbiased sample.
- The cost of learning: random approvals incur real credit losses, establishing the price of acquiring unbiased calibration data.

## Process note (read this)
- The decision gate FAILED, and that is reported as is.
- No model on recorded features beat the old policy because the old policy used an applicant trait the data never recorded. A statistical model fitted only on recorded features gives away that information.
- Recalibration on the random sample beat the naive model (the accepted-only model) and brought the predicted default rate within the calibration band, but could not overcome missing information.
- The test set knows every outcome, which no lender has. In real credit operations, evaluating rejected applicants directly is impossible without a randomised trial or simulation.

## Not done
- Real credit bureau or loan-level default data.
- Risk-based pricing or variable interest rates.
- Economic cycle or macroeconomic stress testing.
- Fair lending or disparate impact analysis.
- A hybrid policy combining model scores with underwriter discretion.

## Run

```bash
pip install -r ../requirements.txt
python -m credit_score.train --write-checksum   # first time only
python -m credit_score.train                    # writes results/ (a few seconds)
python -m credit_score.report                   # renders reports/ from results/
python -m pytest                                # tests; no network needed
```

`--fast` uses fewer applicants and resamples. Its output is marked `fast_mode: true` and the release gate rejects it. Run these commands from this folder.

## Files

| Path | Purpose |
|---|---|
| `credit_score/data.py` | Simulated loan applicants, unrecorded trait, approval policies, validation, fingerprint |
| `credit_score/models.py` | Accepted-only model, IPW weighting, random sample model, recalibration |
| `credit_score/metrics.py` | Profit per applicant, calibration ratio, Gini coefficient, bootstrap intervals |
| `credit_score/train.py` | Training, evaluation across policies, sample-size sensitivity, error analysis, plot |
| `credit_score/report.py` | Renders the model card and decision memo from `results/` |
| `tests/` | Data generation, schema validation, metrics checks, fast run, reproduction test |
| `results/` | `metrics.json`, `error_analysis.json`, `calibration.png` |
| `reports/` | `model_card.md`, `decision_memo.md` |

