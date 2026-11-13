# Data card: simulated loan applicants with reject bias

## Source
None. Every applicant, feature, and outcome is drawn by `credit_score/data.py`. No download, no lender data, no personal data.

## Why simulated
In real lending data, repayment outcomes are observed only for approved applicants (reject bias). A lender never observes whether a declined applicant would have defaulted or repaid. The simulation generates both the unrecorded applicant trait and the true counterfactual default outcomes for all applicants, allowing unbiased evaluation of models that no real lender can perform. The data source is named `simulated_known_truth`.

## Terms
CC-BY-4.0, same as the rest of this repository's data and documentation.

## Model
- 40,000 past applications (`N_HISTORY`) and 40,000 fresh test applications (`N_TEST`).
- 5 recorded numeric features (standard normal).
- One unrecorded applicant trait (standard normal) that increases default probability (`HIDDEN_ON_DEFAULT = 0.6`).
- True log-odds of default: `-2.0 + 0.9*x0 + 0.6*x1 - 0.5*x2 + 0.4*x0*x1 + 0.5*(x3 > 1) + 0.6*u`.
- Old approval policy uses a risk index using both recorded features and the unrecorded trait: `0.9*x0 + 0.6*x1 - 0.5*x2 + 0.7*u - 2.0`. It approves softly below cutoff -1.6 with sharpness 3.0.
- Random sample: a random 5% of all applicants are approved regardless of the old policy (`RANDOM_APPROVE_SHARE = 0.05`).
- History observes outcomes only for approved loans (old policy approvals plus random approvals). Test set knows every applicant's outcome.

## What is assumed, not observed
Every parameter above, plus loan economics: margin 0.15 on a repaid loan, loss 0.60 on a default, approving when predicted default probability is below 0.20.

## Reproducibility
`data/CHECKSUMS.txt` holds a SHA-256 of a small fixed generator probe (seed=0). `python -m credit_score.train` refuses to run if the generator no longer produces it.

## Known quirks
- Features are independent standard normals except for non-linear interactions in the true default function.
- The unrecorded trait has no direct proxy among recorded features.
- Test set contains counterfactual outcomes for rejected applicants, which no real lender has.

## Not checked
- Real credit bureau or loan performance data.
- Macroeconomic cycles, inflation, or changing interest rates.
- Fair lending regulations, protected attributes, or adverse action requirements.

