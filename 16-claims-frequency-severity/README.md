# Project 16: Claims frequency and severity pricing

**Status: built on simulated policies with claims frequency and severity. The decision gate PASSED.** Results are reproducible offline. They describe this simulated setup, not any insurer.

Domain: [banking, fintech and insurance](../domains/banking-insurance.md).

## Decision context
An insurer needs to price personal auto insurance policies by segment to achieve a target loss ratio (claims / charged premium) of 0.70. Under flat or unsegmented pricing, low-risk drivers cross-subsidize high-risk drivers, inducing adverse selection where the riskiest segment runs at an unsustainable underwriting deficit. The decision question is whether an actuarial two-part frequency-severity model (Poisson GLM with log-exposure offset times Gamma severity GLM) provides sufficient discrimination and decile stability to go live. The decision gate is fixed in `claims_pricing/config.py` before results are seen.

## What you learn
- Why exposure duration must weight frequency estimation (ignoring exposure treats short-term policies as low-risk).
- Why frequency and severity should be modelled separately when feature effects diverge (mileage affects frequency; vehicle value affects severity).
- Evaluating risk segmentation using the actuarial pure premium Gini index (concentration curve of cumulative exposure versus cumulative loss).
- Diagnosing adverse selection by computing realized loss ratios across risk deciles.
- Measuring paired bootstrap improvements over a single compound Tweedie regression.

## Data
Simulated. See `data/DATA_CARD.md`.

## The traps
Measured in `results/metrics.json` under `traps`:
1. Ignoring exposure time: treating unweighted claim count as annual frequency underprices short-term policies.
2. Loss ratio by decile: flat rating produces an unprofitable top-decile loss ratio, while segmented pricing keeps all deciles stable.
3. One model for frequency and size: fitting a single compound Tweedie regression conflates frequency drivers with severity drivers, reducing discrimination and increasing pure premium error.

## Process note (read this)
- The gate passed: pure premium Gini index 95% bootstrap lower bound exceeded the 0.25 threshold, paired difference over single Tweedie was positive, and the top-decile loss ratio remained below the 0.95 cap.
- The single compound Tweedie GLM achieves comparable aggregate Gini discrimination but suffers higher root mean squared error on pure premium because it cannot decouple frequency factors from severity factors.
- Exposure duration was simulated uniformly between 0.10 and 1.0 year without cancellation bias. In real books, early cancellations often correlate with high claim hazard.

## Not done
- Real insurance company policy or claims databases.
- Heavy-tailed Pareto severity distributions for catastrophic casualty losses.
- Vehicle make/model symboling tables and telematics sensor streams.
- Geographic territory boundary smoothing and credibility weighting.

## Run

```bash
python -m claims_pricing.train --write-checksum   # first time only
python -m claims_pricing.train                    # writes results/ (a few seconds)
python -m claims_pricing.report                   # renders reports/ from results/
python -m pytest                                  # tests; no network needed
```

`--fast` uses smaller sample sizes and fewer bootstrap iterations. Its output is marked `fast_mode: true` and the release gate rejects it. Run these commands from this folder.

## Files

| Path | Purpose |
|---|---|
| `claims_pricing/config.py` | Constants, economics assumptions, candidate models, and decision gate |
| `claims_pricing/data.py` | Simulated insurance portfolio with exposure durations, validation, fingerprint |
| `claims_pricing/models.py` | Flat baseline, no-exposure naive model, single Tweedie GLM, two-part headline model |
| `claims_pricing/metrics.py` | Actuarial Gini index, decile loss ratios, paired bootstrap intervals |
| `claims_pricing/train.py` | Pipeline execution, trap evaluation, error analysis, artifact outputs |
| `claims_pricing/report.py` | Renders the model card and decision memo from `results/` |
| `tests/` | Unit tests for data generation, metrics, trap mechanisms, and pipeline reproduction |
| `results/` | `metrics.json`, `error_analysis.json`, `loss_ratio_by_decile.png` |
| `reports/` | `model_card.md`, `decision_memo.md` |

