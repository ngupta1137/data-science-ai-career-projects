# Contributing

Contributions are welcome if they raise the bar. A change is accepted when it meets the review bar below.

## Review bar

- Every number in a report comes from a committed file under `results/`, produced by code in the repo.
- New or changed code has tests. Metric code has a test against a hand-computed value.
- No third-party data is committed. Add a download script and a checksum.
- The data card states the source, the terms, and what was not checked.
- A leakage risk is named and tested for.
- No em dashes. Plain, direct language.
- `make check` passes (ruff lint and format, layout gate, tests).

## Reporting a wrong result

Open an issue with the project name, the command you ran, your library versions, and the number you got versus the committed number. A result that does not reproduce is treated as a bug.

## Proposing a new project

Open an issue first. A new project must teach one skill the existing projects do not, use a dataset whose terms allow this use, and meet the evidence standard in `docs/evidence-standard.md`.
