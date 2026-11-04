# Project NN: <name>

**Status:** planned | built | released (also set in the `STATUS` file)

## Decision context
Who acts on this prediction, and what does a wrong prediction cost?

## Data
Source, terms, rows, target. Link to `data/DATA_CARD.md`.

## Split design
How the model will be used, how the split mirrors that, and the leakage risk.

## Results
Rendered from `results/metrics.json`. Do not type numbers here.

## The trap
One beginner failure, shown with its measured damage.

## Error analysis
Where the model fails, by segment.

## Not done
What this project deliberately leaves out.

## Run
```bash
python -m src.download_data
python -m src.train
python -m src.report
python -m pytest
```
