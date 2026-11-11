# Data card: simulated customers for an offer trial

## Source
None. Every customer is drawn by `uplift_eval/data.py` from a model whose mechanisms are set by the code. No download, no personal data, no real retailer or person.

## Why simulated
Uplift is the effect of an offer on one customer, and real data never shows it: each customer is seen either offered or not offered. A simulation records both outcomes' buying chances, so every policy can be scored on its true profit, and traps like the response-model mistake can be measured. The cost is realism, listed below. The data source is named `simulated_known_truth`. Tests use small draws; `results/` come from the full run.

## Terms
CC-BY-4.0, same as the rest of this repository's data and documentation.

## Model
- Eight standard normal features `x` (the first six matter, the last two are noise), and one unmeasured driver `u` of buying.
- Baseline buying chance: `sigmoid(-2 + 0.8*x4 + 0.6*x5 + 0.3*x6 + c*u)` (`c` = 0.8).
- The offer shifts the log odds by `0.4 + 0.9*x1 + 0.7*[x2>0]*x3 - 0.5*x1*x2`. It helps some customers and hurts others, so the true effect has both signs.
- Trial: offers are a fair coin flip. Observational variant: offers follow `x4`, `x5` and the hidden `u`, which are also drivers of buying. That variant is used only to measure what training on past targeted offers costs.
- Train, dev and test sets are independent draws of 20,000, 10,000 and 20,000 customers. A fresh population of 200,000 gives each policy's true profit.

## What is assumed, not observed
The effect formula, the strength of the hidden driver, the observational targeting rule, the value of a conversion (50) and the cost of an offer (2). All are choices to make the mechanisms easy to read.

## Reproducibility
`data/CHECKSUMS.txt` holds a SHA-256 of a small fixed draw. `python -m uplift_eval.train` refuses to run if the generator no longer produces it. It does not hash the full simulation.

## Known quirks
- The effect is strong and smooth, and the trial is large. Real uplift is usually small against noise and needs far more data.
- One binary offer, one binary outcome.
- The class-transformation learner relies on exactly half the trial being offered.

## Not checked
- Real data, multiple offers, dose, timing, fatigue or long-run effects.
