# Decision Memo: Transitional Care Outreach Risk Stratification

**To:** Chief Medical Officer, VP of Population Health, and Care Management Leadership
**From:** Applied Biostatistics and Clinical Machine Learning Team
**Date:** 2026-11-14
**Decision Gate Status:** **PASSED**
**Action:** Approve the Calibrated Clinical Risk Model for transitional care coordinator triage. Explicitly ban uncalibrated balanced classification models from clinical workflow integration.

> [!NOTE]
> No patient data is used in this repository. All clinical observations are synthetically generated. Nothing herein constitutes medical advice, clinical guidance, or regulatory determination.

## The Question
Should our hospital network triage post-discharge transitional care nurse coordinator outreach using standard balanced classification algorithms (optimized solely for AUC) or a well-calibrated logistic risk model validated via Decision Curve Analysis?

## The Rule (Pre-registered Before Results)
Pass only if all four hold for the headline Calibrated Clinical Risk Model, fixed before results: (1) Brier score on holdout admissions is below 0.150 (and strictly lower than uncalibrated baseline by at least 0.030); (2) Expected Calibration Error (ECE across 10 bins) is below 0.035; (3) Net Benefit at clinical intervention threshold pt = 0.15 is at least 0.085 and strictly exceeds both Treat All and the uncalibrated model; (4) Discrimination AUC is at least 0.720 with 95% bootstrap confidence interval lower bound above 0.700.

## The Call
The candidate Calibrated Clinical Risk Model **PASSED** all four pre-registered decision gate criteria.
- **brier_score_below_threshold**: PASS
- **expected_calibration_error_below_threshold**: PASS
- **net_benefit_exceeds_defaults**: PASS
- **discrimination_auc_acceptable**: PASS

- **Brier Score:** 0.1485 (uncalibrated: 0.2086, improvement: 0.0600).
- **Expected Calibration Error:** 0.0113 (uncalibrated: 0.2327, improvement: 0.2214).
- **Clinical Net Benefit (pt = 0.15):** 0.0990 (95% CI: [0.0865, 0.1151]).
- **Discrimination AUC:** 0.7231 (95% CI: [0.7006, 0.7453]).

## What This Means
1. **AUC alone is an unsafe metric for clinical deployment:** The uncalibrated balanced model achieves an AUC of 0.7227, virtually indistinguishable from the calibrated model's 0.7231. However, its predicted probabilities are systematically distorted, generating an unacceptable Expected Calibration Error of 0.2327.
2. **Decision Curve Analysis prevents nurse alert fatigue:** At the clinical action threshold pt = 0.15, transitional care nurse contact costs 150 USD while preventing a readmission valued at 1,000 USD. The calibrated model achieves a Net Benefit of 0.0990, avoiding dozens of unwarranted outreach calls per 100 patients compared to indiscriminate outreach. In contrast, the uncalibrated model destroys clinical utility through false positives, cutting Net Benefit by 14.0%.
3. **Site adjustment is mandatory across diverse hospital networks:** In rural critical access facilities where baseline risk is 10.96%, an unadjusted model predicts 19.72% risk, nearly double the true event rate. Incorporating facility-level controls prevents alert fatigue in community and rural hospitals.

## Where This Is Fragile
- **Fixed Cost-to-Benefit Ratio:** The pt = 0.15 threshold assumes nurse coordinator outreach costs 150 USD and prevented adverse events save 1,000 USD net. If nurse staffing shortages increase contact costs, the optimal decision threshold shifts upward.
- **Binary Outcome Window:** The analysis models a single 30-day post-discharge window without accounting for in-hospital mortality as a competing risk.
- **Synthetic Patient Generation:** Clinical records were generated from parameterized demographic and clinical distributions. Prospective clinical validation on real electronic health records is required prior to bed-side deployment.

## What a Wrong Call Costs
- **Deploying the Uncalibrated Model:** Overwhelms transitional care nurse coordinators with false-positive alerts, causing operational burnout, alert fatigue, and missed outreach to genuinely high-risk patients.
- **Treating All Patients Without Triage:** Incurs unsustainable nurse coordinator payroll expenses with diminishing clinical returns.

## Not Done
- Competing risk survival models (e.g., Fine-Gray subdistribution hazards for mortality).
- Natural language processing on unstructured physician discharge summaries.
- Prospective cluster-randomized stepped-wedge clinical trial validation.
