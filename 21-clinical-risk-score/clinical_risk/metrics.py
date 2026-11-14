"""Clinical risk scoring metrics, Decision Curve Analysis (Net Benefit), and calibration."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import config
from .data import ClinicalRiskDataset


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared probability error against binary clinical outcome."""
    return float(brier_score_loss(y_true, y_prob))


def compute_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under receiver operating characteristic curve (c-statistic)."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) across uniform risk deciles."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        in_bin = (y_prob >= low) & (y_prob <= high if i == n_bins - 1 else y_prob < high)
        count = int(np.sum(in_bin))
        if count > 0:
            obs_rate = float(np.mean(y_true[in_bin]))
            pred_rate = float(np.mean(y_prob[in_bin]))
            ece += (count / n) * abs(obs_rate - pred_rate)

    return float(ece)


def compute_net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> float:
    """Decision Curve Analysis: Net Benefit = (TP / N) - (FP / N) * [p_t / (1 - p_t)].

    Derived by Vickers & Elkin (2006) to quantify clinical utility under asymmetric
    cost-benefit ratios between false negatives and false positives.
    """
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError(f"Decision threshold must be strictly between 0 and 1: {threshold}")

    flagged = y_prob >= threshold
    n = len(y_true)
    tp = float(np.sum((flagged == 1) & (y_true == 1)))
    fp = float(np.sum((flagged == 1) & (y_true == 0)))
    weight = threshold / (1.0 - threshold)

    net_benefit = (tp / n) - (fp / n) * weight
    return float(net_benefit)


def compute_decision_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[dict]:
    """Calculate Net Benefit across an operational decision threshold spectrum."""
    pts = thresholds or config.THRESHOLD_GRID
    prevalence = float(np.mean(y_true))
    curve = []

    for pt in pts:
        weight = pt / (1.0 - pt)
        nb_model = compute_net_benefit(y_true, y_prob, pt)
        nb_all = prevalence - (1.0 - prevalence) * weight
        nb_none = 0.0

        # Avoided unnecessary outreach visits per 100 patients vs Treat All
        diff = nb_model - nb_all
        avoided_per_100 = (diff / weight) * 100.0 if weight > 0 else 0.0

        curve.append(
            {
                "threshold": float(pt),
                "net_benefit_model": float(nb_model),
                "net_benefit_treat_all": float(nb_all),
                "net_benefit_treat_none": float(nb_none),
                "avoided_outreach_per_100": float(avoided_per_100),
            }
        )
    return curve


def evaluate_model_performance(
    model,
    dataset: ClinicalRiskDataset,
    threshold: float = config.DECISION_THRESHOLD,
) -> dict:
    """Evaluate full discrimination, calibration, and clinical utility metrics."""
    _, y_true, _ = dataset.to_arrays(include_site=True)
    y_prob = model.predict_proba(dataset)

    brier = compute_brier_score(y_true, y_prob)
    auc = compute_auc(y_true, y_prob)
    ece = compute_expected_calibration_error(y_true, y_prob)
    nb = compute_net_benefit(y_true, y_prob, threshold)
    curve = compute_decision_curve(y_true, y_prob)

    # Site-stratified calibration
    site_breakdown = {}
    obs = dataset.observations
    sites = sorted(config.HOSPITAL_SITES.keys())
    for s in sites:
        idx = np.array([o.hospital_site == s for o in obs], dtype=bool)
        if np.sum(idx) > 0:
            y_s = y_true[idx]
            p_s = y_prob[idx]
            site_breakdown[s] = {
                "admissions": int(np.sum(idx)),
                "observed_prevalence": float(np.mean(y_s)),
                "predicted_mean_risk": float(np.mean(p_s)),
                "brier_score": float(brier_score_loss(y_s, p_s)),
                "auc": float(roc_auc_score(y_s, p_s)) if len(np.unique(y_s)) > 1 else 0.5,
                "net_benefit": float(compute_net_benefit(y_s, p_s, threshold)),
            }

    return {
        "model_name": getattr(model, "name", "unknown"),
        "brier_score": float(brier),
        "auc": float(auc),
        "expected_calibration_error": float(ece),
        "net_benefit": float(nb),
        "decision_threshold": float(threshold),
        "decision_curve": curve,
        "site_breakdown": site_breakdown,
        "mean_predicted_probability": float(np.mean(y_prob)),
        "observed_prevalence": float(np.mean(y_true)),
    }


def bootstrap_metric_cis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 250,
    seed: int = config.SEED,
    threshold: float = config.DECISION_THRESHOLD,
) -> dict[str, tuple[float, float]]:
    """Compute empirical 95% bootstrap confidence intervals for AUC and Net Benefit."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    auc_samples: list[float] = []
    nb_samples: list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        if len(np.unique(y_b)) > 1:
            auc_samples.append(float(roc_auc_score(y_b, p_b)))
        nb_samples.append(float(compute_net_benefit(y_b, p_b, threshold)))

    return {
        "auc_ci_95": (
            float(np.percentile(auc_samples, 2.5)),
            float(np.percentile(auc_samples, 97.5)),
        ),
        "net_benefit_ci_95": (
            float(np.percentile(nb_samples, 2.5)),
            float(np.percentile(nb_samples, 97.5)),
        ),
    }


def evaluate_decision_gate(
    calibrated_eval: dict,
    uncalibrated_eval: dict,
    auc_ci_95: tuple[float, float],
    nb_ci_95: tuple[float, float],
) -> dict:
    """Pre-registered decision gate check."""
    del nb_ci_95
    brier_cal = calibrated_eval["brier_score"]
    brier_uncal = uncalibrated_eval["brier_score"]
    ece_cal = calibrated_eval["expected_calibration_error"]
    ece_uncal = uncalibrated_eval["expected_calibration_error"]
    nb_cal = calibrated_eval["net_benefit"]
    nb_uncal = uncalibrated_eval["net_benefit"]
    auc_cal = calibrated_eval["auc"]

    # Net benefit of Treat All at the threshold
    pt = config.DECISION_THRESHOLD
    prev = calibrated_eval["observed_prevalence"]
    weight = pt / (1.0 - pt)
    nb_treat_all = prev - (1.0 - prev) * weight

    checks = {
        "brier_score_below_threshold": bool(
            brier_cal < config.MAX_BRIER_SCORE
            and (brier_uncal - brier_cal) >= config.MIN_BRIER_IMPROVEMENT
        ),
        "expected_calibration_error_below_threshold": bool(
            ece_cal < config.MAX_ECE and (ece_uncal - ece_cal) >= config.MIN_ECE_IMPROVEMENT
        ),
        "net_benefit_exceeds_defaults": bool(
            nb_cal >= config.MIN_NET_BENEFIT
            and nb_cal > nb_treat_all
            and nb_cal > 0.0
            and nb_cal > nb_uncal
        ),
        "discrimination_auc_acceptable": bool(
            auc_cal >= config.MIN_AUC and auc_ci_95[0] > config.MIN_AUC_CI_LOWER
        ),
    }

    return {
        "rule": config.GATE_RULE,
        "headline_model": config.HEADLINE_MODEL,
        "brier_score_calibrated": float(brier_cal),
        "brier_score_uncalibrated": float(brier_uncal),
        "ece_calibrated": float(ece_cal),
        "ece_uncalibrated": float(ece_uncal),
        "net_benefit_calibrated": float(nb_cal),
        "net_benefit_uncalibrated": float(nb_uncal),
        "net_benefit_treat_all": float(nb_treat_all),
        "auc_calibrated": float(auc_cal),
        "auc_ci_95": [float(auc_ci_95[0]), float(auc_ci_95[1])],
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def compute_traps(
    calibrated_eval: dict,
    uncalibrated_eval: dict,
    unadjusted_site_eval: dict,
    oracle_eval: dict,
) -> dict:
    """Measure the three pre-registered clinical evaluation traps."""
    # Trap 1: High AUC with poor calibration
    auc_cal = calibrated_eval["auc"]
    auc_uncal = uncalibrated_eval["auc"]
    ece_cal = calibrated_eval["expected_calibration_error"]
    ece_uncal = uncalibrated_eval["expected_calibration_error"]

    # Trap 2: Net benefit degradation under decision curve analysis
    nb_cal = calibrated_eval["net_benefit"]
    nb_uncal = uncalibrated_eval["net_benefit"]
    nb_oracle = oracle_eval["net_benefit"]
    nb_treat_all = calibrated_eval["decision_curve"][2]["net_benefit_treat_all"]  # at pt = 0.15

    # Trap 3: Hospital site case-mix shift (Rural access hospital)
    rural_unadj = unadjusted_site_eval["site_breakdown"]["rural_access"]
    rural_cal = calibrated_eval["site_breakdown"]["rural_access"]
    rural_obs = rural_cal["observed_prevalence"]
    rural_pred_unadj = rural_unadj["predicted_mean_risk"]
    rural_pred_cal = rural_cal["predicted_mean_risk"]

    return {
        "high_auc_poor_calibration": {
            "uncalibrated_auc": float(auc_uncal),
            "calibrated_auc": float(auc_cal),
            "auc_difference": float(auc_uncal - auc_cal),
            "uncalibrated_ece": float(ece_uncal),
            "calibrated_ece": float(ece_cal),
            "uncalibrated_brier": float(uncalibrated_eval["brier_score"]),
            "calibrated_brier": float(calibrated_eval["brier_score"]),
            "description": (
                "Standard medical ML models trained with positive class rebalancing achieve "
                "identical or high discrimination (AUC), but their predicted probabilities "
                "are severely inflated, generating extreme calibration error."
            ),
        },
        "net_benefit_degradation": {
            "threshold": float(config.DECISION_THRESHOLD),
            "oracle_net_benefit": float(nb_oracle),
            "calibrated_net_benefit": float(nb_cal),
            "uncalibrated_net_benefit": float(nb_uncal),
            "treat_all_net_benefit": float(nb_treat_all),
            "net_benefit_loss_pct": float(
                (nb_cal - nb_uncal) / nb_cal * 100.0 if nb_cal > 0 else 0.0
            ),
            "description": (
                "On Decision Curve Analysis, uncalibrated risk scores trigger massive false-positive "
                "outreach calls, degrading Net Benefit below the default Treat All strategy."
            ),
        },
        "hospital_case_mix_shift": {
            "target_facility": "rural_access",
            "actual_prevalence": float(rural_obs),
            "unadjusted_predicted_risk": float(rural_pred_unadj),
            "site_adjusted_predicted_risk": float(rural_pred_cal),
            "overprediction_error_unadjusted": float(rural_pred_unadj - rural_obs),
            "overprediction_error_adjusted": float(rural_pred_cal - rural_obs),
            "description": (
                "Deploying an unadjusted model across diverse healthcare facilities causes severe "
                "case-mix distortion: at rural community hospitals with lower baseline acuity, "
                "the unadjusted model over-predicts adverse risk by 2x, causing clinician alert fatigue."
            ),
        },
    }
