# Data card: simulated grocery chain for a promotion

## Source
None. Every store and week is drawn by `promo_lift/data.py`. No download, no retailer data, no personal data.

## Why simulated
The true effect of a promotion on profit is never observed: a store either ran it or did not. A simulation knows both expected outcomes for every store, so each method can be scored against the truth and repeated over hundreds of campaigns. The data source is named `simulated_known_truth`.

## Terms
CC-BY-4.0, same as the rest of this repository's data and documentation.

## Model
- 200 stores, 52 weeks. Store size is log-normal (log mean 4.0, sd 0.5) in weekly units of the promoted item. Each store has its own weekly growth (mean 0.2%, sd 0.3%).
- A shared season of amplitude 15% that peaks in week 33, inside the promotion weeks 30 to 33.
- A sibling item sells about 0.8 of the promoted item, varying by store.
- Weekly units are Poisson with gamma noise (coefficient of variation 0.15).
- Promotion: lift of 60% (urban) or 35% (rural) on the promoted item, 10% of sibling sales lost during the promotion, 8% of promoted-item sales lost in each of the 4 weeks after.
- Economics: margin 2.0 on the promoted item, discount 0.6 on every promoted unit, margin 1.5 on the sibling.
- Headline campaign: half the stores promoted at random. Variant: the half with the best trend is picked.

## What is assumed, not observed
Every number above. They were chosen so that units rise while profit falls, a common pattern in real promotion reviews, but the sizes are not from any retailer.

## Reproducibility
`data/CHECKSUMS.txt` holds a SHA-256 of a small fixed draw. `python -m promo_lift.train` refuses to run if the generator no longer produces it.

## Known quirks
- Stores do not affect each other.
- No stock-outs, so sales equal demand.

## Not checked
- Real retailer data, supplier funding, competitor response.
