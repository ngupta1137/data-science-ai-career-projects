# Model Card: Retail Price Elasticity and Optimal Markup Pricing

**Project:** `20-price-elasticity`
**Domain:** E-commerce, Media and Consumer Apps
**Decision Gate Status:** **PASSED** (reported as is)

## Model Overview
- **Headline Architecture:** Two-Stage Least Squares (2SLS) Instrumental Variable regression correcting for price endogeneity caused by simultaneous demand shocks.
- **Instrument:** Log wholesale unit cost shocks (exogenous supplier commodity and freight price shifts).
- **First-Stage Strength:** Stock-Yogo instrument F-statistic = 2601.73 (threshold: > 50.0).
- **Target Parameter:** Price elasticity of demand (beta = d ln Q / d ln P).
- **Downstream Application:** Profit-maximizing retail markup pricing under constant-elasticity log-log demand.

## Headline Benchmark Results
| Specification | Estimated Elasticity | Estimation Error | Total Gross Profit | Profit Recovery |
| :--- | :--- | :--- | :--- | :--- |
| Oracle (True Causal Benchmark) | -2.1333 | 0.0000 | $400,568.48 | 100.00% |
| Two-Stage Least Squares (2SLS Headline) | -2.1234 | 0.0266 | $393,559.02 | 98.25% |
| Naive OLS (Biased Baseline) | -0.4743 | 1.6757 | $289,411.24 | 72.25% |

- **Headline Metric:** Absolute elasticity estimation error = 0.0266.
- **95% Bootstrap Confidence Interval:** [-2.2499, -2.0211] (empirical_bootstrap_percentile).
- **First-Stage F-Statistic:** 2601.73 (rejects weak instrument hypothesis).

## Category-Level Profit Recovery
| Category | Oracle Profit | 2SLS Recovered Profit | Naive OLS Recovered Profit |
| :--- | :--- | :--- | :--- |
| Electronics | $31,593.02 | $30,798.19 (97.48%) | $13,846.53 (43.83%) |
| Apparel | $125,666.26 | $125,651.70 (99.99%) | $79,922.49 (63.60%) |
| Home living | $243,309.21 | $237,109.12 (97.45%) | $195,642.22 (80.41%) |

## Econometric Traps Measured
1. **Simultaneity / Endogeneity Trap:**
   - Retail price and unobserved demand shock correlation: r = 0.4783.
   - Naive OLS attenuates elasticity from true -2.15 to -0.47 (error: 1.6757).
   - Naive OLS erroneously estimates inelastic demand (|beta| < 1), wrongly predicting that price increases will monotonically raise revenue.
2. **Profit Destruction Trap:**
   - Under naive pricing, setting inelastic markup ceilings destroys 27.75% of potential gross profit ($111,157.24 destroyed).
   - 2SLS pricing captures 98.25% of oracle profit, limiting profit loss to 1.75%.
3. **Omitted Variable Bias Trap:**
   - Omitting holiday seasonality exacerbates attenuation bias by an additional +0.0784 units (naive with seasonality: -0.5069 vs omitting: -0.4285).

## Visual Diagnostics
The demand curve and profit response curves are rendered to [results/elasticity_curve.png](../results/elasticity_curve.png).

## Data & Provenance
- **Training Cohort:** 60 products across 100 weekly transaction periods (6000 observations).
- **Holdout Test Cohort:** 20 unseen products (2000 observations).
- **Dataset Fingerprint (SHA-256):** `5d00a54c092dc2c5dea57b148b90dcecd70e87249f658d7f69961c78c94c8f75`.
- **Python / NumPy Versions:** Python 3.13.7 / NumPy 2.5.3.
