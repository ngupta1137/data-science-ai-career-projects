import numpy as np
import pytest

from clinical_risk import metrics


def test_brier_score_known_cases():
    y_true = np.array([0, 1, 0, 1], dtype=int)

    # Perfect predictions
    perfect = np.array([0.0, 1.0, 0.0, 1.0])
    assert metrics.compute_brier_score(y_true, perfect) == pytest.approx(0.0)

    # Inverted predictions
    inverted = np.array([1.0, 0.0, 1.0, 0.0])
    assert metrics.compute_brier_score(y_true, inverted) == pytest.approx(1.0)

    # Uninformative 0.5
    half = np.array([0.5, 0.5, 0.5, 0.5])
    assert metrics.compute_brier_score(y_true, half) == pytest.approx(0.25)


def test_auc_known_cases():
    y_true = np.array([0, 0, 1, 1], dtype=int)
    perfect_rank = np.array([0.1, 0.2, 0.8, 0.9])
    assert metrics.compute_auc(y_true, perfect_rank) == pytest.approx(1.0)

    inverted_rank = np.array([0.9, 0.8, 0.2, 0.1])
    assert metrics.compute_auc(y_true, inverted_rank) == pytest.approx(0.0)


def test_expected_calibration_error():
    # Perfectly calibrated case: in each bin, predicted prob matches empirical mean
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    # Bin [0, 0.5): 4 items, mean 0.0. Pred: 0.0
    # Bin [0.5, 1.0]: 4 items, mean 1.0. Pred: 1.0
    y_prob = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    ece = metrics.compute_expected_calibration_error(y_true, y_prob, n_bins=2)
    assert ece == pytest.approx(0.0)


def test_net_benefit_hand_calculation():
    # 10 patients: 2 true positive, 8 negative
    # Flagged: 2 true positives, 2 false positives -> TP=2, FP=2, N=10
    # pt = 0.20 -> weight = 0.20 / 0.80 = 0.25
    # NB = (2/10) - (2/10) * 0.25 = 0.20 - 0.05 = 0.15
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.5, 0.4, 0.3, 0.25, 0.1, 0.1, 0.05, 0.05, 0.02, 0.01])
    threshold = 0.20
    nb = metrics.compute_net_benefit(y_true, y_prob, threshold)
    assert nb == pytest.approx(0.15)


def test_decision_curve_properties():
    y_true = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.6, 0.2, 0.3, 0.05, 0.1, 0.7, 0.15])
    curve = metrics.compute_decision_curve(y_true, y_prob, thresholds=[0.10, 0.20])

    for pt_info in curve:
        pt = pt_info["threshold"]
        prev = np.mean(y_true)
        w = pt / (1.0 - pt)
        expected_treat_all = prev - (1.0 - prev) * w
        assert pt_info["net_benefit_treat_none"] == pytest.approx(0.0)
        assert pt_info["net_benefit_treat_all"] == pytest.approx(expected_treat_all)


def test_gate_evaluation_logic():
    cal_eval = {
        "brier_score": 0.140,
        "auc": 0.730,
        "expected_calibration_error": 0.020,
        "net_benefit": 0.095,
        "observed_prevalence": 0.20,
    }
    uncal_eval = {
        "brier_score": 0.200,
        "auc": 0.730,
        "expected_calibration_error": 0.200,
        "net_benefit": 0.080,
    }
    auc_ci = (0.710, 0.750)
    nb_ci = (0.088, 0.105)

    gate = metrics.evaluate_decision_gate(
        calibrated_eval=cal_eval,
        uncalibrated_eval=uncal_eval,
        auc_ci_95=auc_ci,
        nb_ci_95=nb_ci,
    )
    assert gate["passed"] is True
    assert all(gate["checks"].values())

    # Failing Brier score check
    cal_eval_bad = dict(cal_eval)
    cal_eval_bad["brier_score"] = 0.190
    gate_bad = metrics.evaluate_decision_gate(
        calibrated_eval=cal_eval_bad,
        uncalibrated_eval=uncal_eval,
        auc_ci_95=auc_ci,
        nb_ci_95=nb_ci,
    )
    assert gate_bad["passed"] is False
    assert gate_bad["checks"]["brier_score_below_threshold"] is False
