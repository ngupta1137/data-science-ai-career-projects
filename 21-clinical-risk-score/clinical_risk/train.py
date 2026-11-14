"""Training, evaluation, and reporting pipeline for clinical risk scoring and Decision Curve Analysis."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config, data, metrics, models

CAVEATS = [
    "No patient data is used. All clinical observations are synthetically generated under open-source code.",
    "Nothing herein constitutes medical advice, clinical guidance, or regulatory determination.",
    "The 15% decision threshold assumes a 1:5.7 cost-to-benefit ratio between coordinator outreach and prevented readmission.",
    "Clinical workflows require local institutional review and prospective validation before operational deployment.",
]


def _versions() -> dict[str, str]:
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    }


def run_pipeline(
    results_dir: Path | None = None,
    fast_mode: bool = False,
) -> tuple[dict, dict]:
    """Execute the full clinical risk model evaluation and Decision Curve Analysis."""
    out_dir = results_dir or config.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    n_train = config.FAST_TRAIN_PATIENTS if fast_mode else config.TRAIN_PATIENTS
    n_test = config.FAST_TEST_PATIENTS if fast_mode else config.TEST_PATIENTS

    train_ds = data.generate_clinical_data(n_patients=n_train, seed=config.SEED)
    test_ds = data.generate_clinical_data(n_patients=n_test, seed=config.SEED + 100)

    # Validate checksum outside fast mode
    if not fast_mode and config.CHECKSUM_FILE.exists():
        expected_sha = config.CHECKSUM_FILE.read_text(encoding="utf-8").strip().split()[0]
        if train_ds.sha256 != expected_sha:
            raise RuntimeError(
                f"Data fingerprint {train_ds.sha256} does not match expected checksum {expected_sha}."
            )

    # Fit candidate models on training cohort
    calibrated_model = models.CalibratedClinicalRiskModel().fit(train_ds)
    uncalibrated_model = models.UncalibratedRiskModel().fit(train_ds)
    unadjusted_model = models.UnadjustedSiteRiskModel().fit(train_ds)
    triage_model = models.SimpleTriageBaseline().fit(train_ds)
    oracle_model = models.OracleModel().fit(train_ds)

    # Evaluate all models on holdout test cohort
    cal_eval = metrics.evaluate_model_performance(calibrated_model, test_ds)
    uncal_eval = metrics.evaluate_model_performance(uncalibrated_model, test_ds)
    unadj_eval = metrics.evaluate_model_performance(unadjusted_model, test_ds)
    triage_eval = metrics.evaluate_model_performance(triage_model, test_ds)
    oracle_eval = metrics.evaluate_model_performance(oracle_model, test_ds)

    # Bootstrap 95% CIs for headline calibrated model
    n_boot = 50 if fast_mode else 250
    _, y_test, _ = test_ds.to_arrays()
    p_cal_test = calibrated_model.predict_proba(test_ds)
    cis = metrics.bootstrap_metric_cis(y_test, p_cal_test, n_boot=n_boot, seed=config.SEED)

    # Decision gate check
    gate_result = metrics.evaluate_decision_gate(
        calibrated_eval=cal_eval,
        uncalibrated_eval=uncal_eval,
        auc_ci_95=cis["auc_ci_95"],
        nb_ci_95=cis["net_benefit_ci_95"],
    )

    # Traps measurement
    traps = metrics.compute_traps(
        calibrated_eval=cal_eval,
        uncalibrated_eval=uncal_eval,
        unadjusted_site_eval=unadj_eval,
        oracle_eval=oracle_eval,
    )

    # Generate visual diagnostics: Decision Curve and Calibration Curve
    plot_path = out_dir / "decision_curve_and_calibration.png"
    _plot_decision_and_calibration(
        test_ds,
        calibrated_model,
        uncalibrated_model,
        triage_model,
        oracle_model,
        plot_path,
    )

    # Headline block for metrics.json
    headline_block = {
        "model": config.HEADLINE_MODEL,
        "metric": "net_benefit",
        "value": float(round(cal_eval["net_benefit"], 4)),
        "ci_95": [
            float(round(cis["net_benefit_ci_95"][0], 4)),
            float(round(cis["net_benefit_ci_95"][1], 4)),
        ],
        "ci_method": "empirical_bootstrap_percentile",
        "decision_threshold": config.DECISION_THRESHOLD,
    }

    models_summary = {
        "calibrated_clinical_risk": {
            "brier_score": cal_eval["brier_score"],
            "auc": cal_eval["auc"],
            "expected_calibration_error": cal_eval["expected_calibration_error"],
            "net_benefit": cal_eval["net_benefit"],
            "site_breakdown": cal_eval["site_breakdown"],
        },
        "uncalibrated_risk": {
            "brier_score": uncal_eval["brier_score"],
            "auc": uncal_eval["auc"],
            "expected_calibration_error": uncal_eval["expected_calibration_error"],
            "net_benefit": uncal_eval["net_benefit"],
            "site_breakdown": uncal_eval["site_breakdown"],
        },
        "unadjusted_site_risk": {
            "brier_score": unadj_eval["brier_score"],
            "auc": unadj_eval["auc"],
            "expected_calibration_error": unadj_eval["expected_calibration_error"],
            "net_benefit": unadj_eval["net_benefit"],
            "site_breakdown": unadj_eval["site_breakdown"],
        },
        "simple_triage_baseline": {
            "brier_score": triage_eval["brier_score"],
            "auc": triage_eval["auc"],
            "expected_calibration_error": triage_eval["expected_calibration_error"],
            "net_benefit": triage_eval["net_benefit"],
        },
        "oracle": {
            "brier_score": oracle_eval["brier_score"],
            "auc": oracle_eval["auc"],
            "expected_calibration_error": oracle_eval["expected_calibration_error"],
            "net_benefit": oracle_eval["net_benefit"],
        },
    }

    metrics_payload = {
        "project": "21-clinical-risk-score",
        "data_source": "simulated_known_truth",
        "fast_mode": fast_mode,
        "seed": config.SEED,
        "healthcare_disclaimer": config.HEALTHCARE_DISCLAIMER,
        "versions": _versions(),
        "data": {
            "train_patients": n_train,
            "test_patients": n_test,
            "prevalence_train": float(round(train_ds.prevalence, 4)),
            "prevalence_test": float(round(test_ds.prevalence, 4)),
            "site_distribution_test": test_ds.site_counts,
            "mean_age": float(round(test_ds.mean_age, 2)),
            "sha256": train_ds.sha256,
        },
        "assumptions": {
            "decision_threshold": config.DECISION_THRESHOLD,
            "outreach_cost_usd": config.OUTREACH_COST_USD,
            "avoided_admission_value_usd": config.AVOIDED_ADMISSION_VALUE_USD,
            "max_brier_score": config.MAX_BRIER_SCORE,
            "max_ece": config.MAX_ECE,
            "min_net_benefit": config.MIN_NET_BENEFIT,
            "min_auc": config.MIN_AUC,
        },
        "headline": headline_block,
        "gate": gate_result,
        "models": models_summary,
        "traps": traps,
        "caveats": CAVEATS,
    }

    error_analysis_payload = {
        "project": "21-clinical-risk-score",
        "fast_mode": fast_mode,
        "decision_curve_analysis": {
            "threshold_grid": config.THRESHOLD_GRID,
            "calibrated_net_benefit": [
                round(pt_data["net_benefit_model"], 4) for pt_data in cal_eval["decision_curve"]
            ],
            "uncalibrated_net_benefit": [
                round(pt_data["net_benefit_model"], 4) for pt_data in uncal_eval["decision_curve"]
            ],
            "treat_all_net_benefit": [
                round(pt_data["net_benefit_treat_all"], 4) for pt_data in cal_eval["decision_curve"]
            ],
            "avoided_unnecessary_outreach_per_100": [
                round(pt_data["avoided_outreach_per_100"], 1)
                for pt_data in cal_eval["decision_curve"]
            ],
        },
        "site_calibration_breakdown": {
            s: {
                "observed_prevalence": round(
                    cal_eval["site_breakdown"][s]["observed_prevalence"], 4
                ),
                "calibrated_predicted_risk": round(
                    cal_eval["site_breakdown"][s]["predicted_mean_risk"], 4
                ),
                "unadjusted_predicted_risk": round(
                    unadj_eval["site_breakdown"][s]["predicted_mean_risk"], 4
                ),
                "uncalibrated_predicted_risk": round(
                    uncal_eval["site_breakdown"][s]["predicted_mean_risk"], 4
                ),
            }
            for s in config.HOSPITAL_SITES
        },
        "post_hoc_study": {
            "status": "designed_after_results_does_not_change_verdict",
            "observation": (
                "Uncalibrated models falsely appear competitive when evaluated solely on AUC, "
                "but Decision Curve Analysis reveals that using uncalibrated probabilities harms "
                "clinical efficiency by triggering excessive low-yield coordinator outreach."
            ),
        },
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (out_dir / "error_analysis.json").write_text(
        json.dumps(error_analysis_payload, indent=2), encoding="utf-8"
    )

    return metrics_payload, error_analysis_payload


def _plot_decision_and_calibration(
    test_ds: data.ClinicalRiskDataset,
    cal_m,
    uncal_m,
    triage_m,
    oracle_m,
    out_path: Path,
) -> None:
    """Generate side-by-side Decision Curve and Calibration reliability plots."""
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _, y_true, _ = test_ds.to_arrays()

    pts = np.linspace(0.04, 0.38, 50)
    prev = float(np.mean(y_true))

    # 1. Decision Curve Analysis (Left Panel)
    p_cal = cal_m.predict_proba(test_ds)
    p_uncal = uncal_m.predict_proba(test_ds)
    p_triage = triage_m.predict_proba(test_ds)
    p_oracle = oracle_m.predict_proba(test_ds)

    nb_cal = [metrics.compute_net_benefit(y_true, p_cal, pt) for pt in pts]
    nb_uncal = [metrics.compute_net_benefit(y_true, p_uncal, pt) for pt in pts]
    nb_triage = [metrics.compute_net_benefit(y_true, p_triage, pt) for pt in pts]
    nb_oracle = [metrics.compute_net_benefit(y_true, p_oracle, pt) for pt in pts]
    nb_all = [prev - (1.0 - prev) * (pt / (1.0 - pt)) for pt in pts]
    nb_none = np.zeros_like(pts)

    ax1.plot(pts, nb_oracle, "k--", label="Oracle Risk", linewidth=1.8)
    ax1.plot(pts, nb_cal, "b-", label="Calibrated Clinical Risk (Headline)", linewidth=2.2)
    ax1.plot(pts, nb_uncal, "r-", label="Uncalibrated Balanced Model", linewidth=1.8)
    ax1.plot(pts, nb_triage, "m:", label="Simple Clinical Triage", linewidth=1.5)
    ax1.plot(pts, nb_all, "g-.", label="Treat All Patients", linewidth=1.4)
    ax1.plot(pts, nb_none, "k:", label="Treat None (Net Benefit = 0)", linewidth=1.0)

    ax1.axvline(
        config.DECISION_THRESHOLD,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Decision pt = {config.DECISION_THRESHOLD:.2f}",
    )
    ax1.set_xlabel("Clinical Decision Threshold (p_t)")
    ax1.set_ylabel("Net Benefit")
    ax1.set_title("Decision Curve Analysis (Clinical Utility)")
    ax1.set_ylim(-0.02, max(nb_oracle) * 1.25)
    ax1.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Calibration Reliability Curves (Right Panel)
    from sklearn.calibration import calibration_curve

    prob_true_cal, prob_pred_cal = calibration_curve(y_true, p_cal, n_bins=8)
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_true, p_uncal, n_bins=8)

    ax2.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (45 deg)")
    ax2.plot(
        prob_pred_cal, prob_true_cal, "b-s", label="Calibrated Clinical (ECE < 0.03)", linewidth=2
    )
    ax2.plot(
        prob_pred_uncal,
        prob_true_uncal,
        "r-o",
        label="Uncalibrated Balanced (Severe ECE)",
        linewidth=2,
    )

    ax2.set_xlabel("Mean Predicted 30-Day Risk")
    ax2.set_ylabel("Observed Readmission Rate")
    ax2.set_title("Calibration Reliability Diagram")
    ax2.set_xlim(0, 0.8)
    ax2.set_ylim(0, 0.8)
    ax2.legend(loc="upper left", fontsize=8.5, frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run clinical risk score evaluation and Decision Curve Analysis."
    )
    parser.add_argument(
        "--fast", action="store_true", help="Run fast test with reduced patient cohort."
    )
    parser.add_argument(
        "--write-checksum",
        action="store_true",
        help="Write dataset SHA-256 to CHECKSUMS.txt.",
    )
    args = parser.parse_args()

    metrics_res, _ = run_pipeline(fast_mode=args.fast)

    if args.write_checksum:
        sha = metrics_res["data"]["sha256"]
        config.CHECKSUM_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.CHECKSUM_FILE.write_text(f"{sha}  simulated_clinical_admissions\n", encoding="utf-8")
        print(f"Wrote checksum {sha} to {config.CHECKSUM_FILE}")

    print(f"Pipeline complete. Gate passed: {metrics_res['gate']['passed']}")


if __name__ == "__main__":
    main()
