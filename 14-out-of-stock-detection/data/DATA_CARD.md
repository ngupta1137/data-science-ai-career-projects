# Data card: simulated shelves that go empty

## Source
None. Every shelf, day and empty-shelf episode is drawn by `oos_detect/data.py`. No download, no retailer data, no personal data.

## Why simulated
In real data you rarely know which zero-sales days were empty shelves, so you cannot score a detector. The simulation records every episode, and it draws all randomness up front so every policy faces the same demand and the same episodes. The data source is named `simulated_known_truth`.

## Terms
CC-BY-4.0, same as the rest of this repository's data and documentation.

## Model
- 50 stores with 200 shelves each. True sales rate per shelf is log-normal (log mean -0.5, sd 1.2) units a day, so many shelves sell less than one unit a day.
- Daily demand is Poisson at that rate. No weekday or season pattern.
- An in-stock shelf goes empty with chance 0.004 a day while the system shows stock, and stays empty for a geometric number of days with mean 14, unless a visit fixes it.
- 60 history days with no visits, then 60 evaluation days where a policy picks up to 150 shelves a day to visit.
- A visit to an empty shelf always fixes it from the next day.

## What is assumed, not observed
Every number above, plus margin 1.0 a unit and visit cost 3.0.

## Reproducibility
`data/CHECKSUMS.txt` holds a SHA-256 of a small fixed draw. `python -m oos_detect.train` refuses to run if the generator no longer produces it.

## Known quirks
- Empty shelves are independent across shelves and stores.
- Demand lost while empty is lost for good; no substitution to other items.

## Not checked
- Real point-of-sale or shelf audit data.
