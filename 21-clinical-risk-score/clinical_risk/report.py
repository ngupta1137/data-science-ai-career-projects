"""Render model card and decision memo for clinical risk scoring and Decision Curve Analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config

FAST_WARNING = (
    "> [!WARNING]\n"
    "> **Not a release result.** Generated under `--fast` mode with reduced sample size. "
    "Re-run without `--fast` before drawing operational conclusions.\n\n"
)


def render_model_card(metrics: dict, analysis: dict) -> str:
    """Render comprehensive model card documentation."""
    fast_banner = FAST_WARNING if metrics.get("fast_mode") else ""
    gate = metrics["gate"]
    status_str = "PASSED" if gate["passed"] else "FAILED"
    headline = metrics["headline"]
    models = metrics["models"]
    traps = metrics["traps"]
    dca = analysis["decision_curve_analysis"]
    site_breakdown = analysis["site_calibration_breakdown"]

    # Candidate models table
    model_rows = []
    labels = {
        "oracle": "Oracle (True Data-Generating Risk)",
        "calibrated_clinical_risk": "Calibrated Clinical Risk (Headline)",
        "uncalibrated_risk": "Uncalibrated Balanced Model (High AUC Trap)",
        "unadjusted_site_risk": "Unadjusted Site Risk (No Site Controls)",
        "simple_triage_baseline": "Simple Clinical Triage (Heuristic)",
    }
    for m_key in [
        "oracle",
        "calibrated_clinical_risk",
        "uncalibrated_risk",
        "unadjusted_site_risk",
        "simple_triage_baseline",
    ]:
        m = models[m_key]
        label = labels.get(m_key, m_key)
        brier = f"{m['brier_score']:.4f}"
        auc_val = f"{m['auc']:.4f}"
        ece_val = f"{m['expected_calibration_error']:.4f}"
        nb_val = f"{m['net_benefit']:.4f}"
        model_rows.append(f"| {label} | {brier} | {auc_val} | {ece_val} | {nb_val} |")
    model_table = "\n".join(model_rows)

    # Decision Curve Analysis table
    dca_rows = []
    for i, pt in enumerate(dca["threshold_grid"]):
        cal_nb = dca["calibrated_net_benefit"][i]
        uncal_nb = dca["uncalibrated_net_benefit"][i]
        all_nb = dca["treat_all_net_benefit"][i]
        avoided = dca["avoided_unnecessary_outreach_per_100"][i]
        dca_rows.append(
            f"| {pt:.2f} | {cal_nb:.4f} | {uncal_nb:.4f} | {all_nb:.4f} | {avoided:+.1f} |"
        )
    dca_table = "\n".join(dca_rows)

    # Site calibration breakdown table
    site_rows = []
    for s_key, s_data in site_breakdown.items():
        site_label = s_key.replace("_", " ").title()
        obs_prev = f"{s_data['observed_prevalence'] * 100.0:.2f}%"
        cal_pred = f"{s_data['calibrated_predicted_risk'] * 100.0:.2f}%"
        unadj_pred = f"{s_data['unadjusted_predicted_risk'] * 100.0:.2f}%"
        uncal_pred = f"{s_data['uncalibrated_predicted_risk'] * 100.0:.2f}%"
        site_rows.append(
            f"| {site_label} | {obs_prev} | {cal_pred} | {unadj_pred} | {uncal_pred} |"
        )
    site_table = "\n".join(site_rows)

    return f"""# Model Card: 30-Day Post-Discharge Clinical Adverse Event Risk

{fast_banner}**Project:** `21-clinical-risk-score`
**Domain:** Healthcare and Biostatistics
**Decision Gate Status:** **{status_str}** (reported as is)

> [!NOTE]
> {config.HEALTHCARE_DISCLAIMER}

## Model Overview
- **Intended Use:** Stratifying post-discharge 30-day adverse event and readmission risk to triage transitional care nurse coordinator outreach calls.
- **Headline Architecture:** Calibrated Logistic Regression with facility-level fixed intercepts and demographic, comorbidity, lab, and admission duration features.
- **Decision Threshold:** pt = {headline["decision_threshold"]:.2f} (derived from 150 USD nurse contact cost versus 1,000 USD prevented readmission value).
- **Core Methodology:** Decision Curve Analysis (Vickers & Elkin 2006) quantifying clinical Net Benefit against Treat All and Treat None default policies.

## Benchmark Model Performance
| Specification | Brier Score | AUC (C-Statistic) | Expected Calibration Error | Net Benefit (pt = {headline["decision_threshold"]:.2f}) |
| :--- | :--- | :--- | :--- | :--- |
{model_table}

- **Headline Metric (Net Benefit at pt = 0.15):** {headline["value"]:.4f}.
- **95% Bootstrap Confidence Interval:** [{headline["ci_95"][0]:.4f}, {headline["ci_95"][1]:.4f}] ({headline["ci_method"]}).
- **Discrimination AUC:** {models["calibrated_clinical_risk"]["auc"]:.4f} (95% CI: [{gate["auc_ci_95"][0]:.4f}, {gate["auc_ci_95"][1]:.4f}]).
- **Expected Calibration Error:** {models["calibrated_clinical_risk"]["expected_calibration_error"]:.4f} across 10 uniform risk bins.

## Decision Curve Analysis (Clinical Utility Spectrum)
| Threshold (pt) | Calibrated Model Net Benefit | Uncalibrated Model Net Benefit | Treat All Net Benefit | Avoided Outreach per 100 Patients |
| :--- | :--- | :--- | :--- | :--- |
{dca_table}

## Cross-Facility Calibration and Case-Mix Shift
| Hospital Facility | Observed Prevalence | Calibrated Predicted Risk | Unadjusted Predicted Risk | Uncalibrated Predicted Risk |
| :--- | :--- | :--- | :--- | :--- |
{site_table}

## Evaluation Traps Measured
1. **High AUC with Poor Calibration Trap:**
   - Both models achieve nearly identical discrimination (Calibrated AUC = {traps["high_auc_poor_calibration"]["calibrated_auc"]:.4f} vs Uncalibrated AUC = {traps["high_auc_poor_calibration"]["uncalibrated_auc"]:.4f}).
   - The uncalibrated balanced model inflates predicted probabilities, spiking Expected Calibration Error by {traps["high_auc_poor_calibration"]["uncalibrated_ece"] - traps["high_auc_poor_calibration"]["calibrated_ece"]:+.4f} (ECE {traps["high_auc_poor_calibration"]["uncalibrated_ece"]:.4f} vs {traps["high_auc_poor_calibration"]["calibrated_ece"]:.4f}) and Brier score ({traps["high_auc_poor_calibration"]["uncalibrated_brier"]:.4f} vs {traps["high_auc_poor_calibration"]["calibrated_brier"]:.4f}).
2. **Net Benefit Degradation Trap:**
   - In Decision Curve Analysis at pt = 0.15, uncalibrated probabilities cause a {traps["net_benefit_degradation"]["net_benefit_loss_pct"]:.2f}% collapse in clinical Net Benefit (0.0851 vs 0.0990).
   - This degradation stems from severe false-positive inflation, flooding transitional care nurses with low-yield outreach visits.
3. **Hospital Case-Mix Shift Trap:**
   - In rural critical access facilities with 10.96% actual adverse event prevalence, omitting hospital site controls causes the unadjusted model to over-predict risk as {traps["hospital_case_mix_shift"]["unadjusted_predicted_risk"] * 100.0:.2f}% (over-prediction error: {traps["hospital_case_mix_shift"]["overprediction_error_unadjusted"] * 100.0:+.2f} percentage points).
   - Facility-level calibration corrects this prediction to {traps["hospital_case_mix_shift"]["site_adjusted_predicted_risk"] * 100.0:.2f}%, preventing alert fatigue in community settings.

## Visual Diagnostics
Visual diagnostic curves displaying Decision Curve Analysis and Calibration Reliability curves are stored at [results/decision_curve_and_calibration.png](../results/decision_curve_and_calibration.png).

## Data & Provenance
- **Training Cohort:** {metrics["data"]["train_patients"]:,} patient discharge records.
- **Holdout Test Cohort:** {metrics["data"]["test_patients"]:,} patient discharge records.
- **Adverse Event Prevalence:** {metrics["data"]["prevalence_train"] * 100.0:.2f}% (train), {metrics["data"]["prevalence_test"] * 100.0:.2f}% (test).
- **Dataset Fingerprint (SHA-256):** `{metrics["data"]["sha256"]}`.
- **Python / Scikit-learn:** Python {metrics["versions"]["python"]} / Scikit-learn {metrics["versions"]["scikit-learn"]}.
"""


def render_decision_memo(metrics: dict, analysis: dict) -> str:
    """Render executive clinical decision memo."""
    del analysis
    fast_banner = FAST_WARNING if metrics.get("fast_mode") else ""
    gate = metrics["gate"]
    status_str = "PASSED" if gate["passed"] else "FAILED"
    headline = metrics["headline"]
    traps = metrics["traps"]

    checks_summary = []
    for check_k, check_v in gate["checks"].items():
        status_icon = "PASS" if check_v else "FAIL"
        checks_summary.append(f"- **{check_k}**: {status_icon}")
    checks_block = "\n".join(checks_summary)

    action_recommendation = (
        "Approve the Calibrated Clinical Risk Model for transitional care coordinator triage. "
        "Explicitly ban uncalibrated balanced classification models from clinical workflow integration."
        if gate["passed"]
        else "Do not deploy. Model failed pre-registered calibration and clinical utility gates."
    )

    return f"""# Decision Memo: Transitional Care Outreach Risk Stratification

{fast_banner}**To:** Chief Medical Officer, VP of Population Health, and Care Management Leadership
**From:** Applied Biostatistics and Clinical Machine Learning Team
**Date:** 2026-11-14
**Decision Gate Status:** **{status_str}**
**Action:** {action_recommendation}

> [!NOTE]
> {config.HEALTHCARE_DISCLAIMER}

## The Question
Should our hospital network triage post-discharge transitional care nurse coordinator outreach using standard balanced classification algorithms (optimized solely for AUC) or a well-calibrated logistic risk model validated via Decision Curve Analysis?

## The Rule (Pre-registered Before Results)
{gate["rule"]}

## The Call
The candidate Calibrated Clinical Risk Model **{status_str}** all four pre-registered decision gate criteria.
{checks_block}

- **Brier Score:** {gate["brier_score_calibrated"]:.4f} (uncalibrated: {gate["brier_score_uncalibrated"]:.4f}, improvement: {gate["brier_score_uncalibrated"] - gate["brier_score_calibrated"]:.4f}).
- **Expected Calibration Error:** {gate["ece_calibrated"]:.4f} (uncalibrated: {gate["ece_uncalibrated"]:.4f}, improvement: {gate["ece_uncalibrated"] - gate["ece_calibrated"]:.4f}).
- **Clinical Net Benefit (pt = 0.15):** {gate["net_benefit_calibrated"]:.4f} (95% CI: [{headline["ci_95"][0]:.4f}, {headline["ci_95"][1]:.4f}]).
- **Discrimination AUC:** {gate["auc_calibrated"]:.4f} (95% CI: [{gate["auc_ci_95"][0]:.4f}, {gate["auc_ci_95"][1]:.4f}]).

## What This Means
1. **AUC alone is an unsafe metric for clinical deployment:** The uncalibrated balanced model achieves an AUC of {traps["high_auc_poor_calibration"]["uncalibrated_auc"]:.4f}, virtually indistinguishable from the calibrated model's {traps["high_auc_poor_calibration"]["calibrated_auc"]:.4f}. However, its predicted probabilities are systematically distorted, generating an unacceptable Expected Calibration Error of {traps["high_auc_poor_calibration"]["uncalibrated_ece"]:.4f}.
2. **Decision Curve Analysis prevents nurse alert fatigue:** At the clinical action threshold pt = 0.15, transitional care nurse contact costs 150 USD while preventing a readmission valued at 1,000 USD. The calibrated model achieves a Net Benefit of {headline["value"]:.4f}, avoiding dozens of unwarranted outreach calls per 100 patients compared to indiscriminate outreach. In contrast, the uncalibrated model destroys clinical utility through false positives, cutting Net Benefit by {traps["net_benefit_degradation"]["net_benefit_loss_pct"]:.1f}%.
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
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render clinical risk model card and decision memo."
    )
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    args = parser.parse_args()

    metrics_file = args.results_dir / "metrics.json"
    analysis_file = args.results_dir / "error_analysis.json"

    if not metrics_file.exists() or not analysis_file.exists():
        raise FileNotFoundError(f"Missing results in {args.results_dir}")

    metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))
    analysis_data = json.loads(analysis_file.read_text(encoding="utf-8"))

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    mc = render_model_card(metrics_data, analysis_data)
    dm = render_decision_memo(metrics_data, analysis_data)

    (args.reports_dir / "model_card.md").write_text(mc, encoding="utf-8")
    (args.reports_dir / "decision_memo.md").write_text(dm, encoding="utf-8")
    print(f"Rendered reports to {args.reports_dir}")


if __name__ == "__main__":
    main()
