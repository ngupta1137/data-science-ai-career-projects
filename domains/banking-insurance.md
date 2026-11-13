# Banking, fintech and insurance

**Who this is for:** People building risk, fraud, collections, underwriting and document-processing models.

This page routes you to projects in this repository. Every project describes a simulation or a generated dataset, not this domain's real systems.

## Projects built for this domain

These projects use a simulation shaped around this domain's own decision.

- [Credit scorecard](../15-credit-scorecard): Score a credit application when outcomes are only seen for loans you approved, and measure the price of learning from a random sample.

## Projects that carry over from the core set

1. [Fraud detection](../02-fraud-detection): Rare-event classification: PR-AUC, calibration, cost-based thresholds, time-based splits.
2. [Fairness and model risk audit](../08-fairness-audit): Audit an approval policy by group, and see what a fix costs in utility.
3. [Model monitoring and drift](../07-model-monitoring): Monitor a deployed score for input drift and for the harder case where the label relationship changes.

## Also useful

- [House price regression](../01-house-price-regression): Valuation-style regression with a spatial leakage trap, relevant to collateral and property models.
- [Structured extraction](../10-structured-extraction): Extract fields from statements, invoices and forms, and measure silent errors.
- [Experiment analysis](../06-experiment-analysis): Evaluate a change to a decision flow with a proper experiment.

## Suggested order

02, then 08, then 07.

## What does not transfer

No real transactions, credit files or claims. Fraud in 02 uses a public dataset that has not yet been run here. The fairness audit in 08 has two groups and one definition (equal opportunity). It is a learning aid, not a fair-lending or compliance method. Take regulatory questions to counsel and your model risk function.

## Questions to ask in this domain

- Which error is more expensive, and is that cost the same across customers?
- Are labels delayed, and do they reflect past decisions rather than outcomes?
- Which groups can you measure at all, and are there enough positives to say anything?
- What evidence would a model risk reviewer ask for before sign-off?

## Status of the projects

See the table in the [root README](../README.md). Projects 01 to 04 have no generated metrics yet. Projects 07 to 11 failed their own decision gates, and say so.
