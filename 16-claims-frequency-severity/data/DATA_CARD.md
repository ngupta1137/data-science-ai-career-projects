# Data card: simulated auto insurance claims portfolio

## Source
None. Every policy, exposure duration, claim count, and claim amount is drawn by `claims_pricing/data.py`. No download, no insurer policyholder data, no personal data.

## Why simulated
In real insurance portfolios you observe realized claims, but you do not observe the counterfactual true frequency lambda and true severity mu. Risk segmentation models are difficult to benchmark without known data-generating parameters. The simulation records true parameters and realizations under common random numbers, so candidate models face identical risks. The data source is named `simulated_known_truth`.

## Terms
CC-BY-4.0, same as the rest of this repository's data and documentation.

## Model
- 30,000 training policies and 20,000 holdout test policies.
- Driver age (18 to 75), annual mileage (4,000 to 25,000 miles), vehicle age (0 to 15 years), vehicle value ($5,000 to $80,000), and urban indicator (40% urban).
- Exposure duration in years drawn uniformly from 0.10 to 1.0 year.
- Claim frequency: Poisson process scaled by exposure, with log frequency depending on driver age, mileage, and urban driving.
- Claim severity: Gamma distribution for claims, with log severity depending on vehicle value, vehicle age, and urban driving.
- True pure premium rate is the product of true frequency and true severity.

## What is assumed, not observed
All parameter coefficients, distributions, target loss ratio (0.70), and break-even thresholds are assumptions chosen by the author.

## Reproducibility
`data/CHECKSUMS.txt` holds a SHA-256 of a small fixed draw. `python -m claims_pricing.train` refuses to run if the generator no longer produces it.

## Known quirks
- Severity follows a Gamma distribution with shape 1.5, omitting heavy Pareto casualty tails.
- Policies and claims are independent; no multi-car or fleet clustering.
- Exposure time is uninformative about risk per unit time.

## Not checked
- Real insurance company policy or claims databases.

