# Retail and consumer goods

**Who this is for:** Analysts, data scientists and product owners working on stores, promotions, pricing, supply and field execution.

This page routes you to projects in this repository. Every project describes a simulation or a generated dataset, not this domain's real systems.

## Projects built for this domain

These projects use a simulation shaped around this domain's own decision.

- [Promotion lift](../13-promotion-lift): was the promotion profitable once the discount on normal sales, cannibalisation and pull-forward are counted?
- [Out-of-stock detection](../14-out-of-stock-detection): which shelves should reps check today, with a fixed number of visits?

## Projects that carry over from the core set

1. [Demand forecasting](../03-demand-forecasting): Forecast sales by item and week, and learn why a rolling-origin backtest is the only honest test.
2. [Uplift modelling](../12-uplift-modelling): Decide which customers or stores get a promotion, and why a model of who buys is not a model of who the promotion changes.
3. [Experiment analysis](../06-experiment-analysis): Judge a price or promotion test: false positives, peeking, variance reduction, power.

## Also useful

- [Model monitoring and drift](../07-model-monitoring): Catch a forecast or recommendation model that has drifted after a range or season change.
- [Recommender evaluation](../11-recommender-eval): Recommendation and next-best-action evaluation when logged actions were chosen by an old policy.
- [Structured extraction](../10-structured-extraction): Pull fields out of invoices, delivery notes and trade documents.

## Suggested order

03, then 06, then 12, then 07.

## What does not transfer

The projects use generated data. They have no real sales calendars, stock-outs, promotion overlaps, cannibalisation, store clustering or retailer data sharing limits. 03 forecasts daily bike rentals (UCI Bike Sharing), not retail point-of-sale data. In 12 the effect is strong and smooth; real promotion lift is usually small against noise.

## Questions to ask in this domain

- What decision does the number feed, and what does a wrong call cost per store or per case?
- Was the promotion or action assigned at random, or by a rule people can explain?
- Do sales include stock-outs, so that zero demand and zero stock look the same?
- Who refits the model and how would you know it has gone stale?

## Status of the projects

Each project's status and decision-gate verdict are in the table in the [root README](../README.md). Failed gates are reported as failed.
