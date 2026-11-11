import numpy as np
import pytest

from fairness_audit import config, data, models


def test_utility_hand_value():
    y = np.array([1, 1, 0, 0])
    approve = np.array([True, False, True, False])
    # one good approved (+1), one bad approved (-2), over four applicants
    assert models.utility(y, approve, cost_bad=2.0) == pytest.approx(-0.25)
    assert models.utility(y, np.zeros(4, dtype=bool)) == 0.0


def test_best_threshold_separates_a_clean_signal():
    score = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    y = np.array([0] * 50 + [1] * 50)
    t = models.best_threshold(score, y)
    assert 0.4 <= t <= 0.6


def test_group_thresholds_reach_the_target_tpr():
    rng = np.random.default_rng(0)
    g = np.repeat([0, 1], 4000)
    y = np.tile([1, 0], 4000)
    score = np.where(y == 1, rng.normal(1.0, 1, 8000), rng.normal(0, 1, 8000))
    score[g == 1] -= 0.5  # group B scores lower
    th = models.group_thresholds(score, y, g, target_tpr=0.6)
    approve = models.apply_thresholds(score, g, th)
    for gval in (0, 1):
        tpr = approve[(g == gval) & (y == 1)].mean()
        assert tpr >= 0.6 and tpr == pytest.approx(0.6, abs=0.01)
    assert th[1] < th[0]


def test_apply_thresholds_uses_the_right_group():
    score = np.array([0.5, 0.5])
    g = np.array([0, 1])
    assert models.apply_thresholds(score, g, {0: 0.6, 1: 0.4}).tolist() == [False, True]


def test_scorer_uses_only_requested_columns():
    splits = data.make_splits(3)
    f = models.fit_scorer(splits["fit"], ["x1", "x2"])
    shuffled = splits["dev"].copy()
    shuffled["x3"] = 100.0
    assert np.allclose(f(splits["dev"]), f(shuffled))
    assert config.FEATURES_NO_PROXY == ("x1", "x2", "x4")
