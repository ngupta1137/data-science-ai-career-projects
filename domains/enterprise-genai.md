# Enterprise generative AI

**Who this is for:** Teams putting language models into workflows and needing to prove that they work before launch.

This page routes you to projects in this repository. Every project describes a simulation or a generated dataset, not this domain's real systems.

## Start here

1. [RAG with evals](../05-rag-evals): Retrieval-augmented answers: recall, MRR, refusal and a failure taxonomy.
2. [LLM judge audit](../09-llm-judge-eval): Audit an LLM judge for position and length bias before using it as a grader.
3. [Structured extraction](../10-structured-extraction): Structured extraction with silent-error accounting and unseen templates.

## Also useful

- [Text classification](../04-text-classification): A strong non-LLM baseline for text tasks.
- [Model monitoring and drift](../07-model-monitoring): Monitoring after launch.
- [Fairness and model risk audit](../08-fairness-audit): Group-wise audit of a decision the system supports.

## Suggested order

05, then 09, then 10.

## What does not transfer

No LLM was called anywhere. 09 uses a simulated judge with planted biases and 10 uses rule-based extractors on generated documents, so they test the evaluation method, not any model. A real-model run is designed, not built. Treat the code as an evaluation harness to point at your own system.

## Questions to ask in this domain

- What does the system do when it should not answer?
- Has the judge you use been checked against people, and for which biases?
- How much does a wrong answer cost compared with a refusal?
- What happens when the document format changes?

## Status of the projects

See the table in the [root README](../README.md). Projects 01 to 04 have no generated metrics yet. Projects 07 to 11 failed their own decision gates, and say so.
