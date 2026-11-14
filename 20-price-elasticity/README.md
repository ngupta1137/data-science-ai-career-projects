# Project 20: Price elasticity and profit optimization

**Status: built on simulated retail transactions. The decision gate PASSED, and that is reported as is.** Every number in the reports comes from `results/metrics.json`.

Part of the [E-commerce, media and consumer apps](../domains/ecommerce-media.md) domain guide.

## Decision context
An e-commerce retail merchandiser and pricing team must decide whether to optimize retail prices using standard ordinary least squares (Naive OLS) or upgrade to a causal econometric Two-Stage Least Squares (2SLS) instrumental variable framework. The person acting on the decision is the Chief Commercial Officer or VP of Pricing and Merchandising. A wrong decision risks deploying pricing recommendations from a naive model that mistakes demand surges for price insensitivity, setting catastrophic markups that collapse sales volume and destroy gross profit margin. The decision gate is fixed in `price_elasticity/config.py` before any results are generated.

## What you learn
- Why observed retail price and sales volume data suffer from simultaneity bias (pricing managers and dynamic pricing algorithms raise prices when demand surges).
- Why naive regression severely attenuates price elasticity toward zero, falsely claiming that consumer demand is inelastic (|beta| < 1.0).
- How exogenous wholesale cost shocks (commodity input costs and freight shifts) serve as valid instrumental variables to identify causal elasticity.
- Why naive pricing destroys gross profit: an algorithm convinced demand is inelastic attempts to maximize revenue by setting exorbitant markups, crushing volume on truly elastic goods.
- How omitted macroeconomic seasonal demand shifters exacerbate price endogeneity.
- How to construct a pre-registered econometric decision gate with bootstrap confidence intervals and counterfactual profit simulations.

## Data
Simulated weekly e-commerce transactions across diverse merchandise categories with known ground truth price elasticity and simultaneous demand shocks.

**Data storage best practice:** Following industry best practices for algorithmic evaluation, no raw synthetic CSV transaction files are committed to git to prevent repository bloat and ensure clean version control. Instead, the reproducible data generator in `price_elasticity/data.py` synthesizes weekly store-level transactions deterministically in memory from fixed random seeds. This ensures zero external download dependencies, prevents repository bloat, and provides exact known counterfactual ground truth. See `data/DATA_CARD.md` for generation details and parameter specifications.

## Models compared
| Model | Identification Strategy | Pricing Recommendation |
|---|---|---|
| Oracle Benchmark | Uses true causal category elasticity parameters | Theoretical gross-profit maximizing price: P* = cost * (beta / (1 + beta)) |
| Two-Stage Least Squares (Headline) | Instruments endogenous price using log wholesale cost shocks | Evaluates causal elasticity and sets profit-maximizing markup |
| Naive OLS (Biased Baseline) | Regresses log quantity on log price and observable covariates | Mistakenly identifies inelastic demand and caps markup at ceiling |

## The traps, measured
1. **Price changed in response to demand (simultaneity bias):** Retail prices and unobserved demand shocks exhibit positive correlation (r = 0.48). Naive OLS attenuates elasticity from true -2.15 to -0.47 (error: 1.68), misdiagnosing elastic products as inelastic. 2SLS purges this endogeneity via wholesale cost instruments (F = 2601.7 >> 50.0), estimating elasticity at -2.12 (error: 0.027).
2. **Profit destruction from naive pricing:** Convinced demand is inelastic, naive pricing drives prices to ceiling markups (4.5x unit cost). Sales volume collapses by over 80%, destroying 27.7% of gross profit ($111,157 destroyed on holdout catalog items). In contrast, 2SLS pricing recovers 98.3% of theoretical oracle gross profit.
3. **Omitted variable bias without seasonality:** Omitting seasonal holiday demand shifters introduces additional positive bias (+0.046 units) into the price coefficient, further disguising true customer price sensitivity.

## Process note (read this)
- The decision gate rule was set in `price_elasticity/config.py` before results were generated: (1) absolute elasticity estimation error against true counterfactual elasticity on holdout products is below 0.25 units (with 95% bootstrap CI upper bound below 0.35); (2) 2SLS achieves strictly lower estimation error than naive OLS; (3) gross profit under 2SLS recovers at least 90% of oracle profit, whereas naive OLS recovers strictly less than 80%.
- The decision gate PASSED. The headline 2SLS model achieved an estimation error of 0.0266 (95% bootstrap CI: [-2.2499, -2.0211]), beat naive OLS (error: 1.6757), and recovered 98.25% of oracle profit while naive OLS recovered only 72.25%.
- Weakest claims:
  1. Instrumental exclusion restriction: Wholesale cost shocks are assumed completely orthogonal to local consumer demand shocks. If supplier price increases coincide with consumer awareness campaigns, the exclusion restriction is violated.
  2. Constant elasticity functional form: Demand is assumed log-log constant elasticity. In real catalogs, price elasticity often varies with price thresholds and competitor reference pricing.
  3. Static competitor reaction: The model holds competitor prices fixed without accounting for dynamic multi-firm pricing retaliation.

## Not done
- Dynamic cross-product cannibalization matrix.
- High-frequency intra-day algorithmic clearance markdown schedules.
- Multi-echelon inventory stockout penalty modeling.

## Run

```bash
pip install -r ../requirements.txt                     # Windows PowerShell: pip install -r ..\requirements.txt
python -m price_elasticity.train --write-checksum        # writes data/CHECKSUMS.txt and results/
python -m price_elasticity.report                        # renders reports/ from results/
python -m pytest                                       # tests; no network needed
```

`python -m price_elasticity.train --fast` runs a rapid smoke test. Its output is marked `fast_mode: true` and the release gate rejects it.

## Files

| Path | Purpose |
|---|---|
| `price_elasticity/config.py` | Constants, economic parameters, pre-set decision gate rule |
| `price_elasticity/data.py` | Retail transaction simulator, cost shocks, demand confounders, validation |
| `price_elasticity/models.py` | Naive OLS, Two-Stage Least Squares (2SLS), Oracle benchmark |
| `price_elasticity/metrics.py` | Elasticity estimation error, counterfactual profit simulator, bootstrap CI, traps |
| `price_elasticity/train.py` | End-to-end training pipeline, gate evaluation, diagnostics plot |
| `price_elasticity/report.py` | Renders model card and decision memo from results |
| `tests/` | Unit tests for data validation, econometric estimators, traps, and reproduction |
| `results/` | `metrics.json`, `error_analysis.json`, `elasticity_curve.png` |
| `reports/` | `model_card.md`, `decision_memo.md` |
