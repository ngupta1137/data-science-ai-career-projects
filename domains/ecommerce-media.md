# E-commerce, media and consumer apps

**Who this is for:** Teams working on search, recommendations, ranking, support and content.

This page routes you to projects in this repository. Every project describes a simulation or a generated dataset, not this domain's real systems.

## Start here

1. [Recommender evaluation](../11-recommender-eval): Recommender evaluation under popularity-biased logs, drift and sampled metrics.
2. [Experiment analysis](../06-experiment-analysis): Online experiments and metric hygiene.
3. [RAG with evals](../05-rag-evals): Retrieval and answer quality for search and support assistants.

## Also useful

- [Text classification](../04-text-classification): Text classification for moderation, routing and tagging.
- [Uplift modelling](../12-uplift-modelling): Choose whom to show an incentive.
- [Model monitoring and drift](../07-model-monitoring): Monitor ranking and feature drift.

## Suggested order

11, then 06, then 05.

## What does not transfer

No real clickstream, catalogue or users. 11 has no online test and no off-policy estimator, so it cannot tell you what a recommender would do live. 05 is run on a generated corpus.

## Questions to ask in this domain

- Were the logged clicks produced by an earlier recommender, and how does that bias evaluation?
- Does the offline metric agree with an online result for any model you have shipped?
- Which segment loses when the average improves?

## Status of the projects

See the table in the [root README](../README.md). Projects 01 to 04 have no generated metrics yet. Projects 07 to 11 failed their own decision gates, and say so.
