# Operations and supply chain

**Who this is for:** Teams working on planning, logistics, back-office processing and service quality.

This page routes you to projects in this repository. Every project describes a simulation or a generated dataset, not this domain's real systems.

## Start here

1. [Demand forecasting](../03-demand-forecasting): Demand forecasting with rolling-origin backtests and MASE.
2. [Model monitoring and drift](../07-model-monitoring): Monitoring of forecasts and inputs, with detection delay and false alarm rates.
3. [Structured extraction](../10-structured-extraction): Extraction of fields from invoices and delivery documents, with silent error rates.

## Also useful

- [Experiment analysis](../06-experiment-analysis): Test an operational change before rollout.
- [Fraud detection](../02-fraud-detection): Anomaly-style rare-event classification for claims, returns or leakage.

## Suggested order

03, then 07, then 10.

## What does not transfer

No real inventory, lead times, capacity limits or hierarchies. A forecast is only a step toward an order or a plan, and this repository stops at the forecast. 10 uses rule-based extractors on generated documents, not an OCR or LLM pipeline.

## Questions to ask in this domain

- What does a one-day error cost versus a one-week error?
- Who acts on the forecast and what do they do when it is wrong?
- Which documents arrive in a format you have not seen, and how do you notice?

## Status of the projects

See the table in the [root README](../README.md). Projects 01 to 04 have no generated metrics yet. Projects 07 to 11 failed their own decision gates, and say so.
