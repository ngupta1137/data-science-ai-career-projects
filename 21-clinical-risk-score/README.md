# Project 21: Clinical risk score and Decision Curve Analysis

**Status: built on simulated clinical discharge records. The decision gate PASSED, and that is reported as is.** Every number in the reports comes from `results/metrics.json`.

Part of the [Healthcare and biostatistics](../domains/healthcare.md) domain guide.

> [!NOTE]
> No patient data is used in this repository. All clinical observations are synthetically generated. Nothing herein constitutes medical advice, clinical guidance, or regulatory determination.

## Decision Context
A regional hospital health system care management team must decide whether to deploy machine learning risk scores to triage post-discharge transitional care nurse coordinator outreach calls. The person acting on the recommendation is the Chief Medical Officer, VP of Population Health, or Care Management Director. A wrong decision causes severe operational failure: deploying a standard high-AUC classification model trained with class reweighting floods nurse coordinators with false-positive alerts on low-risk patients, inducing clinician alert fatigue, burning clinical staff capacity, and diluting outreach to truly high-risk patients. The decision gate was pre-registered in `clinical_risk/config.py` before any results were generated.

## What You Learn
- Why discrimination (AUC / c-statistic) is blind to clinical actionability: two models can have identical 0.72 AUC while one causes severe operational harm.
- Why probability calibration (Brier score and Expected Calibration Error) is mandatory for clinical decision support.
- How to apply Decision Curve Analysis (Vickers & Elkin 2006) to quantify clinical Net Benefit across decision thresholds without requiring arbitrary utility weights.
- How clinical economics dictate the decision threshold: when nurse outreach costs 150 USD and an avoided readmission is valued at 1,000 USD, the clinical action threshold is pt = 150 / 1000 = 0.15 (15% 30-day risk).
- Why cross-facility hospital case-mix shifts require stratified baseline calibration to avoid overwhelming rural and community hospitals with false alarms.
- How to construct a pre-registered clinical evaluation protocol with bootstrap confidence intervals.

## Data
Simulated inpatient hospital discharge cohort across three facility tiers (tertiary academic center, community regional hospital, rural critical access hospital) with parameterized baseline risk, clinical acuity, comorbidities, and 30-day post-discharge outcomes.

**Data storage best practice:** Following industry best practices for algorithmic evaluation, no raw synthetic CSV patient records are committed to git to prevent repository bloat and maintain clean version control. Instead, the reproducible data generator in `clinical_risk/data.py` synthesizes admission records deterministically in memory from fixed random seeds. This ensures zero external download dependencies, prevents repository bloat, and guarantees an exact known ground-truth risk benchmark. See `data/DATA_CARD.md` for generation details and parameter specifications.

## Models Compared
| Model | Specification | Operational Role |
|---|---|---|
| Oracle Benchmark | True latent data-generating risk probability | Theoretical ceiling for clinical Net Benefit |
| Calibrated Clinical Risk (Headline) | Logistic regression with facility fixed intercepts and clinical covariates | Calibrated probability estimation for Decision Curve Analysis |
| Uncalibrated Balanced Model (Trap) | Logistic regression with class rebalancing (`class_weight='balanced'`) | High-AUC classifier with severe probability distortion |
| Unadjusted Site Risk | Logistic regression omitting hospital facility indicators | Illustrates cross-facility case-mix calibration drift |
| Simple Clinical Triage | Rule-based heuristic scoring age, prior admissions, and comorbidities | Standard clinical rule baseline |

## The Traps, Measured
1. **High AUC with poor calibration trap:** Standard clinical ML workflows often apply positive-class reweighting to handle low prevalence. Both models achieve an AUC of 0.723. However, the uncalibrated model inflates predicted probabilities, spiking Expected Calibration Error from 0.0113 to 0.2327 and worsening Brier score from 0.1485 to 0.2086.
2. **Net benefit degradation under Decision Curve Analysis:** At the clinical threshold pt = 0.15, transitional care coordinator outreach costs 150 USD against 1,000 USD prevented readmission value. The calibrated model achieves a Net Benefit of 0.0990 (close to the Oracle ceiling of 0.0992), avoiding 12.3 unnecessary outreach calls per 100 patients compared to treating all patients. The uncalibrated model suffers a 14.0% loss in Net Benefit (0.0851), generating excessive false-positive outreach.
3. **Hospital case-mix shift trap:** In rural critical access hospitals with 10.96% actual 30-day adverse event prevalence, omitting facility indicators causes the unadjusted model to predict 19.72% risk (an over-prediction of +8.76 percentage points). Incorporating facility controls aligns predictions with observed reality (9.61%), eliminating alert fatigue.

## Process Note (Read This)
- The decision gate rule was set in `clinical_risk/config.py` before results were generated: (1) Brier score on holdout admissions is below 0.150 and strictly lower than uncalibrated baseline by at least 0.030; (2) Expected Calibration Error is below 0.035; (3) Net Benefit at pt = 0.15 is at least 0.085 and strictly exceeds both Treat All and the uncalibrated model; (4) Discrimination AUC is at least 0.720 with 95% bootstrap CI lower bound above 0.700.
- The decision gate PASSED. The headline Calibrated Clinical Risk Model achieved Brier score 0.1485, ECE 0.0113, Net Benefit 0.0990 (95% CI: [0.0865, 0.1151]), and AUC 0.7231 (95% CI: [0.7006, 0.7453]).
- Weakest claims:
  1. Fixed cost-to-benefit ratio: The pt = 0.15 threshold assumes nurse coordinator contact costs 150 USD and prevented adverse events save 1,000 USD net. In regions with acute nurse staffing shortages, higher wage rates shift the operational threshold upward.
  2. Single binary outcome window: The evaluation models a single 30-day post-discharge window without accounting for in-hospital mortality as a competing risk.
  3. Synthetic patient population: Observations were generated from parameterized clinical distributions. Prospective validation on real institutional electronic health records is required prior to bed-side deployment.

## Not Done
- Competing risk survival analysis (Fine-Gray subdistribution hazards for mortality).
- Natural language processing on unstructured physician discharge summaries.
- Prospective cluster-randomized stepped-wedge clinical trial validation.

## Run

```bash
pip install -r ../requirements.txt                     # Windows PowerShell: pip install -r ..\requirements.txt
python -m clinical_risk.train --write-checksum          # writes data/CHECKSUMS.txt and results/
python -m clinical_risk.report                          # renders reports/ from results/
python -m pytest                                       # tests; no network needed
```

`python -m clinical_risk.train --fast` runs a rapid smoke test. Its output is marked `fast_mode: true` and the release gate rejects it.

## Files

| Path | Purpose |
|---|---|
| `clinical_risk/config.py` | Constants, clinical decision parameters, pre-set gate rule |
| `clinical_risk/data.py` | Clinical cohort generator, facility mix, validation rules |
| `clinical_risk/models.py` | Calibrated Logistic, Uncalibrated Balanced, Unadjusted Site, Baselines |
| `clinical_risk/metrics.py` | Brier score, AUC, ECE, Decision Curve Analysis (Net Benefit), traps |
| `clinical_risk/train.py` | Full pipeline, gate evaluation, Decision Curve diagnostic plot |
| `clinical_risk/report.py` | Renders model card and clinical decision memo |
| `tests/` | Unit tests for data validation, metric formulas, traps, reproduction |
| `results/` | `metrics.json`, `error_analysis.json`, `decision_curve_and_calibration.png` |
| `reports/` | `model_card.md`, `decision_memo.md` |
