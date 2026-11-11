# Data card: simulated catalogue, users and click log

## Source
None. Every user, item and click is drawn by `recsys_eval/data.py`. No download, no personal data, no real service.

## Why simulated
To say whether an offline metric is right, you need to know what users actually like, which a click log never says. Clicks only record what was shown and then chosen. With a simulation you know every user's true affinity for every item, so you can compare the offline metric with the truth, and change one cause (how strongly popular items are shown) while holding the rest fixed. The cost is realism, listed below. This is simulated data with a known truth, not test fixtures: tests use small draws, while `results/` come from the full run. The data source is named `simulated_known_truth`.

## Terms
CC-BY-4.0, same as the rest of this repository's data and documentation.

## Model
- 1,500 users and 600 items, each with an 8-dimensional latent vector. Affinity is their dot product over the square root of 8.
- A user's taste drifts linearly over 100 days along a random direction (drift 0.5).
- 40% of items are released after day 0, uniformly over the first 95 days. Nothing is shown before its release.
- Each item has a base popularity and a trend, both independent of how much anyone likes it.
- The logging policy shows items with probability proportional to `exp(1.5 * (popularity + trend * time))`. 250,000 impressions are drawn over 50 time bins, to users in proportion to a lognormal activity level.
- A shown item is clicked with probability `sigmoid(1.2 * affinity - 1.5)`. Only clicks are kept, about 59,000 of them.
- Validation window: days 70 to 85. Test window: days 85 to 100.

## What is assumed, not observed
Every number above. The popularity-biased exposure is the key assumption: real services also show popular things more, but how much is unknown. The sweep in the model card varies it.

## Reproducibility
`data/CHECKSUMS.txt` holds a SHA-256 of a small fixed draw. `python -m recsys_eval.train` refuses to run if the generator no longer produces it. It does not hash the full simulation.

## Known quirks
- Preference is exactly low-rank plus drift, which suits matrix factorisation. Real preference is not so tidy.
- Exposure does not depend on the user. Real systems personalise exposure, which makes the bias harder to model.
- One click type. No dwell time, ratings, purchases, sessions or context.
- No feedback loop: recommendations do not change later logs.
- Items the user already clicked are excluded from recommendations, which is why popularity can score below random on true like rate (the remaining popular items are ones the user was shown and passed on).

## Not checked
- Any real log, public benchmark or live traffic.
