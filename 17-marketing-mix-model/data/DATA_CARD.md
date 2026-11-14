# Data card: simulated multi-channel marketing mix time series

## Source
None. Every weekly spend, organic baseline sales, and revenue observation is drawn by `marketing_mix/data.py`. No third-party download, no commercial advertising account data, no customer or user data.

## Why simulated
In real advertising portfolios, ground-truth incremental sales lift and true channel saturation curves are unobservable without expensive, prolonged randomized holdout experiments. Time-series marketing mix models are notoriously difficult to validate because confounding seasonal cycles, trend, and correlated media flighting obscure actual causality. The simulation records true data-generating parameters (exact adstock decay rates, Hill half-saturation thresholds, and maximum incremental lift), providing an objective ground truth to evaluate budget allocation policies and diagnostic traps. The data source is named `simulated_known_truth`.

## Terms
CC-BY-4.0, same as the rest of this repository's code and documentation.

## Model
- 104 training weeks (2 years) and 52 holdout evaluation weeks (1 year).
- Baseline organic sales: base intercept $100,000/week, linear organic trend +$50/week, annual 52-week sinusoidal seasonality amplitude $15,000, and Q4 holiday surge peak $25,000.
- Three marketing channels:
  - TV: weekly bursts, mean spend ~$37.5k/week (range $10k to $70k). True carryover decay alpha 0.60, Hill half-saturation spend $40,000, max lift $50,000.
  - Search: performance capture, mean spend ~$19.5k/week (range $6k to $36k). Spend is correlated with TV flighting (r ~ 0.56) representing search query halo. True carryover decay alpha 0.15, Hill half-saturation spend $18,000, max lift $28,000.
  - Social: periodic flighting, mean spend ~$26.5k/week (range $8k to $48k). True carryover decay alpha 0.35, Hill half-saturation spend $25,000, max lift $35,000.
- Observation noise: additive Gaussian noise with standard deviation $4,000/week.

## What is assumed, not observed
All true parameters, channel adstock decay rates, Hill half-saturation constants, maximum lift potentials, budget allocation constraints, and decision thresholds are assumptions chosen by the author.

## Reproducibility
`data/CHECKSUMS.txt` holds a SHA-256 hash of a deterministic fixed data draw. `python -m marketing_mix.train` checks this hash and refuses to run if generator output changes.

## Known quirks
- Channel saturation is modeled via concave Hill-1 / Michaelis-Menten response; S-shaped inflection curves are not modeled.
- Adstock carryover uses geometric decay; delayed peak adstock (e.g. Weibull or gamma distributed lag) is omitted.
- Spend correlation between TV and Search is linear Gaussian without ad-auction price spikes.

## Not checked
- Real advertiser media buy records or transaction logs.
- Cross-channel interaction effects (e.g. TV exposure increasing social click-through rate).

