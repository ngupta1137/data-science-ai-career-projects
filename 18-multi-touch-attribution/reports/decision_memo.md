# Decision Memo: Multi-Touch Attribution Channel Reallocation

**To:** Chief Marketing Officer / Head of Growth
**From:** Marketing Science and Analytics
**Date:** 2026-11-14
**Recommendation Status:** **FAILED**

## The Question
Which marketing channels truly drive incremental conversions, and should we abandon heuristic last-touch attribution in favor of a first-order Markov chain model to reallocate our acquisition budget?

## The Rule (Pre-registered Before Results)
Pass only if all three hold for the first-order Markov attribution model, fixed before results as the headline: (1) the mean absolute error (MAE) across channel credit shares against true counterfactual Shapley credit is below 0.05 (5 percentage points); (2) the headline model achieves strictly lower share MAE than the naive last-touch baseline; (3) budget reallocation using headline credit shares yields at least a 5% increase in true incremental conversions over last-touch budget allocation on holdout test journeys.

## The Call
The headline Markov attribution model **FAILED** the pre-registered decision gate.
- **channel_credit_mae_below_threshold**: FAIL
- **beats_last_touch_mae**: FAIL
- **reallocation_gain_at_least_5pct**: FAIL

- **Headline Model Share MAE:** 0.0571 (95% CI: [0.0554, 0.0590]).
- **Last-Touch Share MAE:** 0.0522.
- **Budget Reallocation Yield:** 4,430.5 conversions vs 4,280.5 under last touch (+3.50% gain).

## What This Means
1. **First-order Markov attribution improves top-of-funnel visibility:** Last touch severely under-rewards Display (12.2% vs 19.2% true causal contribution) because Display acts as an early awareness touchpoint.
2. **Bottom-of-funnel channels absorb unearned credit:** Last-touch awards Search and Email outsized shares because they sit immediately prior to checkout.
3. **Organic conversions distort overall ROI:** Roughly 36.5% of conversions represent organic customer demand that would occur even with zero ad exposure. Rule-based models that assign 100% of revenue to paid media overstate true marketing incrementality.

## Where This Is Fragile
- **Memoryless assumption:** First-order Markov chains treat transition probabilities as dependent only on the immediate prior state, ignoring journey duration and cumulative multi-touch exposure effects.
- **Static auction pricing:** The budget simulation assumes fixed unit costs per touch ($0.50 for Display, $2.50 for Search), ignoring programmatic bidding competition that drives up marginal CPMs as spend scales.

## What a Wrong Call Costs
- If we stay on naive Last Touch, we will continue starving top-of-funnel customer acquisition, resulting in shrinking top-funnel audience pools over time.
- If we prematurely deploy an uncalibrated algorithmic model without randomized geo-lift or holdout validation, we risk misallocating media budget based on observational correlation rather than causal incrementality.

## Not Done
- No cookie-less identity resolution or cross-device identity graphs.
- No dynamic bidding response curves.
- No long-term brand equity carryover modeled past the multi-week purchase cycle.
