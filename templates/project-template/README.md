# Project NN: <name>

**Status:** planned | in_progress | built | released (also set in the `STATUS` file)

Copy this folder to `NN-name/` at the repo root and follow `docs/repo-layout.md`.

## Decision context
Who acts on this prediction, and what does a wrong prediction cost?

## What you learn
The skills this project teaches, in plain terms.

## Data
Source, terms, rows, target. Link to `data/DATA_CARD.md`.

## Split design
How the model will be used, how the split mirrors that, and the leakage risk.

## The traps
Beginner failures, each shown with its measured damage in `results/metrics.json`.

## Not done
What this project deliberately leaves out.

## Run
```bash
python -m <package>.download_data --write-checksum
python -m <package>.train
python -m <package>.report
python -m pytest
```

## Files
A table of package modules and what each does.
