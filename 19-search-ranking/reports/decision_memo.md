# Decision Memo: Ship the New Search Ranker?

**To:** Head of Search and Discovery / VP of Engineering
**From:** Search Ranking and Information Retrieval Science
**Date:** 2026-11-14
**Recommendation Status:** **PASSED**
**Action:** Ship the IPS Debiased Ranker to production. Reject the Naive Click Ranker.

## The Question
Should we ship the new candidate search ranker to replace the baseline production BM25 system, or does its apparent offline performance gain vanish once position bias in historical click logs is eliminated?

## The Rule (Pre-registered Before Results)
Pass only if all three hold for the candidate Inverse Propensity Scoring ranker (ips_debiased), fixed before results as the headline: (1) the True Unbiased NDCG@10 lift over the baseline production ranker is at least +0.030 (with 95% bootstrap CI lower bound strictly greater than 0); (2) the True NDCG@10 lift on tail queries is non-negative (>= 0.000), ensuring head query popularity does not mask tail query degradation; (3) the offline IPS-counterfactual NDCG lift directionally agrees with true unbiased NDCG lift (sign matching), confirming that the offline metric does not vanish once position bias is removed.

## The Call
The candidate Inverse Propensity Scoring (IPS) debiased ranker **PASSED** the pre-registered decision gate.
- **unbiased_ndcg_lift_meets_threshold**: PASS
- **ci_lower_bound_positive**: PASS
- **no_tail_query_degradation**: PASS
- **debiased_agreement_ratio**: PASS

- **True Unbiased NDCG@10 Lift:** +0.0425 (95% CI: [+0.0383, +0.0459]).
- **Tail Query True NDCG Lift:** +0.0255 (non-negative, no tail cannibalization).
- **IPS Counterfactual Offline Lift:** +0.0034 (directionally agrees with true online gain).

## What This Means
1. **Never evaluate rankers on raw clicks:** Position 1 captures 85.6% of top-funnel clicks due to user gaze patterns rather than true document relevance. A model optimized on raw clicks simply memorizes past exposure.
2. **Apparent offline gains can be illusions:** The naive click model achieves positive lift on biased click metrics (+0.0001), but its true relevance lift vanishes (-0.0024). Shipping the naive model would degrade true user satisfaction.
3. **Head query success masks tail failure:** The naive model improves head queries (+0.0062) by boosting historically clicked popular brands, but severely damages tail queries (-0.0223). IPS weighting prevents this popularity distortion, lifting tail queries by +0.0255.

## Where This Is Fragile
- **Propensity model accuracy:** The IPS estimator relies on the power-law position decay assumption (eta = 1.35). If production examination patterns deviate substantially due to UI layout changes (e.g., carousels, image badges), propensity estimation must be recalibrated via randomized interleaving.
- **Propensity variance:** High inverse propensity weights (w_max = 15.0) on rare lower-position clicks increase variance in the gradient updates.

## What a Wrong Call Costs
- **Shipping the Naive Ranker:** Would reinforce a feedback loop where already-popular items monopolize top ranks, starving novel products and destroying search relevance on long-tail niche queries.
- **Prematurely Rejecting the IPS Ranker:** Would leave a +4.2% True NDCG relevance improvement on the table, continuing to subject users to a sub-optimal BM25 lexical ranking experience.

## Not Done
- No real-time contextual bandit exploration.
- No session-level multi-click query reformulation modeling.
- No cross-device user personalization graph.
