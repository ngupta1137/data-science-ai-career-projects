import numpy as np
import pytest

from fairness_audit import metrics


def test_auc_hand_values():
    y = np.array([0, 0, 1, 1])
    assert metrics.auc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert metrics.auc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    # one of four positive-negative pairs is misordered
    assert metrics.auc(np.array([0.1, 0.5, 0.4, 0.9]), y) == pytest.approx(0.75)
    assert np.isnan(metrics.auc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_group_report_hand_example():
    # group A: y=[1,1,0,0], approve=[1,0,1,0]; group B: y=[1,0,0,0], approve=[0,0,0,1]
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    approve = np.array([1, 0, 1, 0, 0, 0, 0, 1], dtype=bool)
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    rep = metrics.group_report(y, approve.astype(float), approve, g)
    assert rep["A"]["tpr"] == 0.5 and rep["A"]["fpr"] == 0.5 and rep["A"]["ppv"] == 0.5
    assert rep["A"]["selection_rate"] == 0.5 and rep["A"]["base_rate"] == 0.5
    assert rep["B"]["tpr"] == 0.0 and rep["B"]["fpr"] == pytest.approx(1 / 3)
    assert rep["B"]["ppv"] == 0.0 and rep["B"]["base_rate"] == 0.25
    d = metrics.disparities(rep)
    assert d["tpr_gap"] == pytest.approx(-0.5)
    assert d["fpr_gap"] == pytest.approx(1 / 3 - 0.5)
    assert d["selection_ratio_min_over_max"] == 0.5


def test_disparities_handle_nobody_approved():
    y = np.array([1, 0, 1, 0])
    approve = np.zeros(4, dtype=bool)
    rep = metrics.group_report(y, approve.astype(float), approve, np.array([0, 0, 1, 1]))
    d = metrics.disparities(rep)
    assert d["selection_ratio_min_over_max"] is None and d["ppv_gap"] is None


def test_stratified_bootstrap_keeps_group_sizes_and_is_deterministic():
    g = np.array([0] * 70 + [1] * 30)
    seen = []

    def stat(idx):
        seen.append(((g[idx] == 0).sum(), (g[idx] == 1).sum()))
        return float(g[idx].mean())

    a = metrics.stratified_bootstrap(stat, g, n_boot=20, seed=1)
    assert set(seen) == {(70, 30)}
    assert np.array_equal(a, metrics.stratified_bootstrap(stat, g, n_boot=20, seed=1))
    assert np.allclose(a, 0.3)


def test_ci95_ignores_nan_and_handles_all_nan():
    low, high = metrics.ci95(np.array([1.0, 2.0, 3.0, np.nan]))
    assert 1.0 <= low <= high <= 3.0
    assert all(np.isnan(v) for v in metrics.ci95(np.array([np.nan, np.nan])))
