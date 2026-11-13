# Hostile Review of Projects 01 to 14 (v0.1.0 Release)

This review examines each project from the perspective of an expert reviewer one level up (a Principal ML Scientist, Chief Risk Officer, or Head of Decision Science). For each of the 14 projects, it identifies the single most damaging question a sceptical reviewer could ask, evaluates whether the committed evidence answers it, and classifies the required remedy as either a wording change or new work.

Per release instructions, no new work is performed here. Every item is labelled strictly "wording" or "new work".

---

## Project 01: House Price Regression

- **Question**: The decision memo claims the model passes as a first-pass valuation screen, but 1990 census prices are capped at $500,000 where the model fails systematically, the typical test error is $67,987 (over 30% of median house value), and temporal validation is absent. Can this model be defended as an automated valuation screen for real financial decisions?
- **Committed evidence**: Partly. The evidence proves that spatial splitting prevents severe leakage (a random split understates RMSE by 0.1919) and identifies the $500,000 cap as the weakest segment. However, the evidence cannot show temporal stability on a single cross-section, nor does it provide a cost-weighted utility metric.
- **Classification**: **wording**
- **Remedy**: Clarify in the decision memo that the PASS verdict only certifies statistical improvement over a spatial constant baseline in an academic census snapshot, and explicitly state that the model cannot be deployed as an automated screen for financial decisions.

---

## Project 02: Fraud Detection

- **Question**: The 79.4% cost reduction and optimal threshold rule assume an unconstrained review capacity and a static $5 review cost per flagged transaction. If investigation queues have hard daily caps or review costs scale with transaction complexity, does the per-transaction rule still beat fixed-threshold policies?
- **Committed evidence**: Partly. The model card includes a sensitivity table varying review cost from $1 to $20, confirming the expected-cost rule remains superior across that range. However, the committed evidence assumes an infinite analyst queue with zero backlog, ignoring queue exhaustion and capacity bounds.
- **Classification**: **wording**
- **Remedy**: State explicitly in the decision memo that the cost advantage is demonstrated only under unconstrained capacity and static unit costs, and note that operational implementation requires capacity-constrained thresholding.

---

## Project 03: Demand Forecasting

- **Question**: Why was an autoregressive linear Ridge model deployed for a 7-day multi-step daily forecast without exogenous weather features or weather forecasts, when bike rental demand is overwhelmingly driven by rain and temperature?
- **Committed evidence**: Yes. The committed evidence answers this question by reporting a failed decision gate. The headline Ridge model achieves a test MAE of 984.6 rentals/day, losing to a simple 28-day historical moving average (961.0 rentals/day). The decision memo explicitly names missing weather data and calls out weekday 6 as the worst segment.
- **Classification**: **wording**
- **Remedy**: State in the decision memo that historical autoregression alone is fundamentally insufficient for outdoor demand planning, and that weather forecast integration is mandatory for future iterations.

---

## Project 04: Text Classification

- **Question**: The decision memo declares PASS because Naive Bayes beats a degenerate majority baseline (macro-F1 0.6761 vs 0.0050 across 20 classes), but no transformer or neural baseline was benchmarked. Does clearing a near-zero baseline hurdle demonstrate sufficient accuracy for operational text routing?
- **Committed evidence**: No. The committed evidence proves that text cleaning prevents metadata leakage (retaining headers/footers inflates F1 to 0.8252), but deliberately leaves out transformer models and does not provide an operational cost model for misrouted posts.
- **Classification**: **wording**
- **Remedy**: Update the decision memo to clarify that the gate was designed as an entry-level sanity test against metadata leakage and class imbalance, not as a sufficient threshold for production routing without benchmark against modern language models.

---

## Project 05: RAG with Evals

- **Question**: The pipeline passes its minimum bar gate, but has a confidence ranking AUC of 0.379 (worse than random guessing) and correctly refuses only 1 out of 16 unanswerable questions (6.25%). Does clearing this gate demonstrate a viable RAG system or merely a retriever that reliably hallucinates on unanswerable inputs?
- **Committed evidence**: No. While the decision memo honestly acknowledges the failure (reporting answered_unanswerable on 15 of 16 questions and an inverted confidence AUC of 0.379), the committed evidence demonstrates that the current pipeline cannot reliably distinguish answerable from unanswerable queries.
- **Classification**: **new work**
- **Remedy**: Building an effective refusal mechanism (such as semantic entailment checks, calibrated confidence scoring, or an upstream query routing classifier) and validating it on unanswerable queries is new engineering work.

---

## Project 06: Experiment Analysis

- **Question**: The analysis validates CUPED and Holm correction under standard simulation settings, but how do variance reduction and false positive rates behave under severe heavy-tailed distributions (e.g., whale spenders) or sample ratio mismatches (SRM)?
- **Committed evidence**: Partly. The error grid tests sample size and effect size combinations, disclosing where standard errors become anti-conservative. However, the simulation does not evaluate heavy-tailed Pareto distributions or sample ratio mismatches.
- **Classification**: **wording**
- **Remedy**: Add a prominent disclaimer in the decision memo stating that CUPED standard errors require sandwich or bootstrap estimation when metrics exhibit heavy tails, and that SRM tests must precede variance reduction in production.

---

## Project 07: Model Monitoring and Drift

- **Question**: Since the label-free PSI monitor fails its false alarm rate limit (upper bound 0.105 vs 0.10 target) due to calibration noise and catches concept drift in only 3.3% of streams, why should any production team rely on input drift monitoring for model governance?
- **Committed evidence**: Yes. The committed evidence answers this question by failing the gate. It shows that label-free monitors are fundamentally blind to concept drift (which is only caught weeks later when ground truth labels arrive) and proves that rule-of-thumb cutoffs (PSI > 0.2) either produce 100% false alarms or miss real shifts.
- **Classification**: **wording**
- **Remedy**: Emphasize in the decision memo that input drift monitoring must never be treated as a proxy for performance monitoring, and that calibrating drift thresholds requires an extensive reference history rarely available in practice.

---

## Project 08: Fairness and Model Risk Audit

- **Question**: The decision memo shows that selecting a blind policy on a 3,000-applicant validation set appeared fair (gap 0.087) but failed on test (gap 0.149). Does post hoc threshold adjustment solve group disparity when small subgroup positive counts make statistical certification impossible?
- **Committed evidence**: Yes. The decision memo directly documents the failure (test TPR gap 0.149). It explains that picking the lowest-gap model on small dev sets causes selection bias, and shows that even when group thresholds equalize rates in expectation, small minority positive counts leave the confidence interval wider than the regulatory tolerance.
- **Classification**: **wording**
- **Remedy**: Clarify in the decision memo that sample size requirements for regulatory fairness certification are strictly bounded by minority positive counts, making certification impossible without adequate subgroup representation.

---

## Project 09: LLM Judge Audit

- **Question**: Because all LLM judge outputs and human annotators were synthetically simulated with fixed planted bias parameters rather than generated by actual frontier LLMs, does this project measure actual LLM judge reliability or merely validate the simulator's built-in formulas?
- **Committed evidence**: No. The committed evidence demonstrates the mathematical validity of swap-order filtering and length bias vulnerability in the synthetic system, but no live LLM API was called.
- **Classification**: **new work**
- **Remedy**: Executing the designed audit against real commercial or open-weight LLMs with human benchmark annotations requires new API integration, compute, and evaluation runs.

---

## Project 10: Structured Extraction

- **Question**: On unseen invoice templates, the headline extractor produced 715 errors, and 100% of them were valid dates, amounts, or currencies that pass schema validation. Why was an extraction model that fails silently on new layouts evaluated for unattended automation?
- **Committed evidence**: Yes. The decision memo answers this by failing the decision gate (unseen template accuracy 0.313 vs 0.70 bar). It specifically highlights the danger of silent valid errors over explicit abstention, noting that rule-based extractors are operationally safer because they abstain when uncertain.
- **Classification**: **wording**
- **Remedy**: Update the decision memo to explicitly state that unattended processing of unseen invoice formats presents unacceptable financial ledger risk, and mandatory human review is required for unindexed templates.

---

## Project 11: Recommender Evaluation

- **Question**: If offline NDCG@10 on logged clicks exhibits a negative correlation (-0.68) with true user satisfaction and ranks the oracle model near the bottom, does logged offline evaluation provide any reliable signal for choosing recommender algorithms?
- **Committed evidence**: No. While the committed evidence diagnoses the failure mechanism by failing the gate, offline unweighted ranking provides zero reliable signal for production deployment decisions.
- **Classification**: **new work**
- **Remedy**: Building an unbiased evaluation pipeline (deploying randomized exploration traffic to log propensity scores, fitting an inverse propensity score estimator, or running an online A/B test) is new engineering work.

---

## Project 12: Uplift Modelling

- **Question**: The model passes its decision gate on a large trial of 20,000 customers with an uncommonly strong negative treatment effect (35% sleeping dogs who buy less when discounted). What happens to model selection and net profit in typical retail settings with small sample sizes and subtle treatment heterogeneity?
- **Committed evidence**: Yes. The committed evidence directly answers this in the fragility section: with 2,000 training customers, causal trees collapse and simple linear models dominate. Furthermore, dev set AUUC differences between top models are within noise, making model selection akin to a coin flip.
- **Classification**: **wording**
- **Remedy**: Explicitly state in the decision memo that uplift modeling requires large-scale randomized trials (>5,000 customers) and substantial treatment heterogeneity, and should not be attempted on small pilot campaigns.

---

## Project 13: Promotion Lift

- **Question**: In 22% of simulated 200-store campaigns, the difference-in-differences estimator failed to exclude zero and gave an inconclusive recommendation. Does an evaluation protocol that happened to yield a decisive result on one simulated campaign provide a robust decision framework?
- **Committed evidence**: Yes. The committed evidence explicitly documents this power limitation: the decision memo states that at 200 stores, the method gives a definite answer in only 78% of campaigns, and includes a sample-size curve showing that 400+ stores are necessary for 90%+ statistical power.
- **Classification**: **wording**
- **Remedy**: Note in the decision memo that single-campaign evaluation at 200 stores is under-powered, and decision gates for promotion policies should require minimum power across multi-campaign portfolios rather than relying on a single campaign realization.

---

## Project 14: Out-of-Stock Detection

- **Question**: If 86.4% of store visits dispatched by the model find fully stocked shelves (precision 0.136), how long before store managers and stock clerks lose all trust in the system and stop performing shelf audits?
- **Committed evidence**: No. The committed evidence measures the low precision (0.136) and notes that visits are wasted, but nothing in the repository measures manager behavior, worker compliance over time, or the operational tipping point where store clerks abandon the audit list.
- **Classification**: **new work**
- **Remedy**: Developing a calibrated probability model from audited visits, designing a dynamically tuned visit policy that enforces an operational precision floor, and validating frontline adoption in pilot stores is new work.
