# Marketing and growth

**Who this is for:** Teams running campaigns, offers, lifecycle messages and experiments.

This page routes you to projects in this repository. Every project describes a simulation or a generated dataset, not this domain's real systems.

## Start here

1. [Uplift modelling](../12-uplift-modelling): Uplift modelling: whom to target, Qini curves, profit by inverse-probability weighting.
2. [Experiment analysis](../06-experiment-analysis): Experiment analysis: false positives, CUPED, power, peeking, many metrics.
3. [Recommender evaluation](../11-recommender-eval): Offline recommender evaluation and exposure bias.

## Also useful

- [Text classification](../04-text-classification): Classify customer text such as tickets, reviews and survey answers.
- [Model monitoring and drift](../07-model-monitoring): Monitor audience and score drift between campaigns.

## Suggested order

06, then 12, then 11.

## What does not transfer

Everything is simulated. Real campaigns have fatigue, timing, multiple offers, long-run effects and spill-over between customers, none of which are modelled. 12 shows a one-off binary offer.

## Questions to ask in this domain

- Was the offer randomised? If not, can you say how it was assigned?
- What is the cost of an offer and the value of a conversion, and how sensitive is the decision to both?
- How many customers do you have per arm, and does the data-needed table in 12 suggest a simple learner is enough?
- Are you comparing policies by paired difference or by separate noisy totals?

## Status of the projects

See the table in the [root README](../README.md). Projects 01 to 04 have no generated metrics yet. Projects 07 to 11 failed their own decision gates, and say so.
