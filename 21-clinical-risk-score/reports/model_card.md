# Model Card: 30-Day Post-Discharge Clinical Adverse Event Risk

**Project:** `21-clinical-risk-score`
**Domain:** Healthcare and Biostatistics
**Decision Gate Status:** **PASSED** (reported as is)

> [!NOTE]
> No patient data is used in this repository. All clinical observations are synthetically generated. Nothing herein constitutes medical advice, clinical guidance, or regulatory determination.

## Model Overview
- **Intended Use:** Stratifying post-discharge 30-day adverse event and readmission risk to triage transitional care nurse coordinator outreach calls.
- **Headline Architecture:** Calibrated Logistic Regression with facility-level fixed intercepts and demographic, comorbidity, lab, and admission duration features.
- **Decision Threshold:** pt = 0.15 (derived from 150 USD nurse contact cost versus 1,000 USD prevented readmission value).
- **Core Methodology:** Decision Curve Analysis (Vickers & Elkin 2006) quantifying clinical Net Benefit against Treat All and Treat None default policies.

## Benchmark Model Performance
| Specification | Brier Score | AUC (C-Statistic) | Expected Calibration Error | Net Benefit (pt = 0.15) |
| :--- | :--- | :--- | :--- | :--- |
| Oracle (True Data-Generating Risk) | 0.1484 | 0.7251 | 0.0138 | 0.0992 |
| Calibrated Clinical Risk (Headline) | 0.1485 | 0.7231 | 0.0113 | 0.0990 |
| Uncalibrated Balanced Model (High AUC Trap) | 0.2086 | 0.7227 | 0.2327 | 0.0851 |
| Unadjusted Site Risk (No Site Controls) | 0.1527 | 0.6995 | 0.0120 | 0.0991 |
| Simple Clinical Triage (Heuristic) | 0.1585 | 0.6593 | 0.0221 | 0.0909 |

- **Headline Metric (Net Benefit at pt = 0.15):** 0.0990.
- **95% Bootstrap Confidence Interval:** [0.0865, 0.1151] (empirical_bootstrap_percentile).
- **Discrimination AUC:** 0.7231 (95% CI: [0.7006, 0.7453]).
- **Expected Calibration Error:** 0.0113 across 10 uniform risk bins.

## Decision Curve Analysis (Clinical Utility Spectrum)
| Threshold (pt) | Calibrated Model Net Benefit | Uncalibrated Model Net Benefit | Treat All Net Benefit | Avoided Outreach per 100 Patients |
| :--- | :--- | :--- | :--- | :--- |
| 0.05 | 0.1747 | 0.1745 | 0.1744 | +0.6 |
| 0.10 | 0.1377 | 0.1299 | 0.1285 | +8.3 |
| 0.15 | 0.0990 | 0.0851 | 0.0773 | +12.3 |
| 0.20 | 0.0763 | 0.0404 | 0.0196 | +22.7 |
| 0.25 | 0.0560 | 0.0006 | -0.0458 | +30.5 |
| 0.30 | 0.0420 | -0.0352 | -0.1205 | +37.9 |
| 0.35 | 0.0341 | -0.0690 | -0.2067 | +44.7 |
| 0.40 | 0.0256 | -0.0947 | -0.3072 | +49.9 |

## Cross-Facility Calibration and Case-Mix Shift
| Hospital Facility | Observed Prevalence | Calibrated Predicted Risk | Unadjusted Predicted Risk | Uncalibrated Predicted Risk |
| :--- | :--- | :--- | :--- | :--- |
| Tertiary Academic | 27.86% | 27.27% | 21.13% | 54.25% |
| Community Regional | 18.53% | 18.24% | 21.51% | 41.41% |
| Rural Access | 10.96% | 9.61% | 19.72% | 25.94% |

## Evaluation Traps Measured
1. **High AUC with Poor Calibration Trap:**
   - Both models achieve nearly identical discrimination (Calibrated AUC = 0.7231 vs Uncalibrated AUC = 0.7227).
   - The uncalibrated balanced model inflates predicted probabilities, spiking Expected Calibration Error by +0.2214 (ECE 0.2327 vs 0.0113) and Brier score (0.2086 vs 0.1485).
2. **Net Benefit Degradation Trap:**
   - In Decision Curve Analysis at pt = 0.15, uncalibrated probabilities cause a 14.01% collapse in clinical Net Benefit (0.0851 vs 0.0990).
   - This degradation stems from severe false-positive inflation, flooding transitional care nurses with low-yield outreach visits.
3. **Hospital Case-Mix Shift Trap:**
   - In rural critical access facilities with 10.96% actual adverse event prevalence, omitting hospital site controls causes the unadjusted model to over-predict risk as 19.72% (over-prediction error: +8.76 percentage points).
   - Facility-level calibration corrects this prediction to 9.61%, preventing alert fatigue in community settings.

## Visual Diagnostics
Visual diagnostic curves displaying Decision Curve Analysis and Calibration Reliability curves are stored at [results/decision_curve_and_calibration.png](../results/decision_curve_and_calibration.png).

## Data & Provenance
- **Training Cohort:** 6,000 patient discharge records.
- **Holdout Test Cohort:** 3,000 patient discharge records.
- **Adverse Event Prevalence:** 21.15% (train), 21.57% (test).
- **Dataset Fingerprint (SHA-256):** `054a5ff9340308029d6480bf4b115879c7508715a9beac7e565483fbe4bf1e2e`.
- **Python / Scikit-learn:** Python 3.13.7 / Scikit-learn 1.9.1.
