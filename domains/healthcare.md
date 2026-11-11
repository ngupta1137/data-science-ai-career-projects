# Healthcare and life sciences

**Who this is for:** Teams working on clinical operations, patient communication, documentation and product information.

This page routes you to projects in this repository. Every project describes a simulation or a generated dataset, not this domain's real systems.

## Start here

1. [Text classification](../04-text-classification): Classify clinical or product text with a baseline, macro-F1 and a leakage check.
2. [RAG with evals](../05-rag-evals): Retrieval and refusal for question answering over a document set.
3. [Fairness and model risk audit](../08-fairness-audit): Audit a decision policy by group before it is used.

## Also useful

- [Model monitoring and drift](../07-model-monitoring): Monitor a deployed model for drift.
- [Structured extraction](../10-structured-extraction): Extract fields from forms and reports.
- [LLM judge audit](../09-llm-judge-eval): Check an automated judge before relying on it to grade answers.

## Suggested order

04, then 05, then 08.

## What does not transfer

No clinical or patient data is used anywhere in this repository, and nothing here is a medical device, a clinical validation method or privacy guidance. Clinical use needs real data governance, clinical review and regulatory advice. The projects teach evaluation habits only.

## Questions to ask in this domain

- Who is harmed by a wrong answer, and how is that checked by a person?
- Is the label a clinical truth or a record of a past decision?
- Which patient groups have too few examples to audit?
- What must the system refuse, and how do you measure that it does?

## Status of the projects

See the table in the [root README](../README.md). Projects 01 to 04 have no generated metrics yet. Projects 07 to 11 failed their own decision gates, and say so.
