# Project 17: Marketing mix model and budget reallocation

**Status: built on simulated weekly marketing and sales time series. The decision gate FAILED, and that is reported as is.** Results are reproducible offline. They describe this simulated setup, not any commercial advertiser.

Domain: [marketing and growth](../domains/marketing-growth.md).

## Decision context
An omni-channel brand spends an average of over $80,000 per week across TV, Paid Search, and Paid Social. Leadership must decide where to reallocate budget across these channels to maximize incremental sales revenue. Under naive linear regression, unmodelled carryover and spend correlations between TV and Search cause estimated channel returns to distort, inverting Search's slope and recommending cutting Search to the minimum spend bound. The decision question is whether an econometric marketing mix model (MMM) with geometric adstock carryover and Hill saturation provides reliable guidance to reallocate the marketing budget. The decision gate is fixed in `marketing_mix/config.py` before results are seen.

## What you learn
- Why geometric adstock carryover is needed to capture the multi-week lag effects of brand broadcast advertising.
- Why Hill saturation curves prevent linear models from over-promising returns on channels that suffer diminishing marginal returns.
- Diagnosing omitted variable bias and multicollinearity when upper-funnel TV campaigns drive branded search query volume.
- Formulating constrained budget optimization (SLSQP) to find the portfolio allocation that balances marginal returns across media channels.
- Measuring the economic difference between predictive accuracy (holdout R^2) and decision value (incremental revenue gain under budget reallocation).

## Data
Simulated. See `data/DATA_CARD.md`.

## The traps
Measured in `results/metrics.json` under `traps`:
1. Ignoring adstock decay: naive linear regression assumes zero carryover, understating TV's true multi-week brand persistence.
2. Ignoring saturation: linear extrapolation assumes doubling spend yields double return, whereas true Hill saturation produces severe diminishing marginal returns.
3. Correlated spend bias: when TV and Search spend are synchronized, naive OLS suffers multicollinearity, inverting Search's regression slope and cutting Search budget.

## Process note (read this)
- The decision gate FAILED, and that is reported as is. The holdout weekly revenue R^2 bootstrap lower bound fell short of the pre-set 0.75 threshold, and the marginal ROI rank correlation across channels missed the 0.80 bar because TV and Social marginal returns are virtually tied in this simulated environment.
- Crucially, the economic decision check passed: allocating budget with the headline MMM achieves higher true incremental revenue than naive linear regression, preventing the linear model from cutting Search to the minimum bound due to multicollinearity.
- Weakest claim: in production media buying, marketing channels exhibit dynamic auction CPM volatility, creative fatigue, and cross-channel synergy (e.g. TV exposure boosting social click-through rates) that are not modeled here; in the presence of strong cross-channel synergies, independent channel saturation curves may misstate the optimal allocation.

## Not done
- Real advertiser transaction, media buy, or attribution clickstream logs.
- Multi-cell randomized geo-lift experiment calibration.
- Weibull or delayed peak adstock lag distributions.
- Creative wear-in and wear-out fatigue curves.
- Macroeconomic indicators, competitor media pressure, or supply chain stock-outs.

## Run

```bash
python -m marketing_mix.train --write-checksum   # first time only
python -m marketing_mix.train                    # writes results/ (a few seconds)
python -m marketing_mix.report                   # renders reports/ from results/
python -m pytest                                 # tests; no network needed
```

`--fast` uses smaller sample sizes and fewer bootstrap iterations. Its output is marked `fast_mode: true` and the release gate rejects it. Run these commands from this folder.

## Files

| Path | Purpose |
|---|---|
| `marketing_mix/config.py` | Constants, simulator parameters, candidate models, and decision gate |
| `marketing_mix/data.py` | Simulated multi-channel marketing time series, validation, fingerprint |
| `marketing_mix/models.py` | Naive linear, adstock-only, saturation-only, headline MMM, and oracle |
| `marketing_mix/metrics.py` | Econometric metrics, budget optimization, bootstrap intervals |
| `marketing_mix/train.py` | Pipeline execution, trap evaluation, error analysis, artifact outputs |
| `marketing_mix/report.py` | Renders the model card and decision memo from `results/` |
| `tests/` | Unit tests for data generation, metrics, trap mechanisms, and pipeline reproduction |
| `results/` | `metrics.json`, `error_analysis.json`, `response_curves.png` |
| `reports/` | `model_card.md`, `decision_memo.md` |

