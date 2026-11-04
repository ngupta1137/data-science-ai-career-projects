import numpy as np
import pytest

from fraud_detection import metrics


def test_average_precision_matches_hand_value():
    # Sorted by score: 0.8 (fraud), 0.4 (legit), 0.35 (fraud), 0.1 (legit).
    # Precision 1.0 at recall 0.5, then 2/3 at recall 1.0. AP = 0.5 * 1 + 0.5 * 2/3.
    y = [0, 0, 1, 1]
    s = [0.1, 0.4, 0.35, 0.8]
    assert metrics.average_precision(y, s) == pytest.approx(0.5 + 0.5 * 2 / 3)


def test_constant_scorer_gets_the_fraud_rate():
    y = np.array([0] * 97 + [1] * 3)
    assert metrics.average_precision(y, np.zeros(100)) == pytest.approx(0.03)


def test_recall_at_top_fraction_matches_hand_value():
    y = [1, 0, 1, 0, 0]
    s = [0.9, 0.8, 0.7, 0.1, 0.2]
    # Top 40% of 5 rows is 2 rows: scores 0.9 (fraud) and 0.8 (legit). Caught 1 of 2.
    assert metrics.recall_at_top_fraction(y, s, 0.4) == pytest.approx(0.5)


def test_policy_cost_matches_hand_value():
    y = [1, 0, 1, 0]
    amount = [100.0, 50.0, 20.0, 10.0]
    flag = [True, False, False, True]
    # Missed fraud: row 2 with amount 20. Two flags at 5 each = 10. Total 30.
    assert metrics.policy_cost(y, amount, flag, 5.0) == pytest.approx(30.0)


def test_policy_cost_flag_nothing_is_total_fraud_amount():
    y = [1, 0, 1]
    amount = [100.0, 5.0, 20.0]
    assert metrics.policy_cost(y, amount, [False] * 3, 5.0) == pytest.approx(120.0)


def test_best_global_threshold_flags_everything_when_review_is_cheap():
    scores = [0.9, 0.8, 0.1]
    y = [1, 0, 1]
    amount = [100.0, 10.0, 50.0]
    # k=0: 150. k=1: 150-100+5=55. k=2: 50+10=60. k=3: 0+15=15.
    threshold, cost = metrics.best_global_threshold(scores, y, amount, 5.0)
    assert cost == pytest.approx(15.0)
    assert threshold == pytest.approx(0.1)


def test_best_global_threshold_flags_only_the_top_row_at_medium_cost():
    # Review cost 60. k=0: 150. k=1: 150-100+60=110. k=2: 50+120=170. k=3: 180.
    threshold, cost = metrics.best_global_threshold(
        [0.9, 0.8, 0.1], [1, 0, 1], [100.0, 10.0, 50.0], 60.0
    )
    assert threshold == pytest.approx(0.9)
    assert cost == pytest.approx(110.0)


def test_best_global_threshold_flags_nothing_when_review_is_expensive():
    # Review cost 200. Flagging even the top row costs 150-100+200=250, above 150.
    threshold, cost = metrics.best_global_threshold(
        [0.9, 0.8, 0.1], [1, 0, 1], [100.0, 10.0, 50.0], 200.0
    )
    assert threshold == float("inf")
    assert cost == pytest.approx(150.0)


def test_best_global_threshold_never_splits_tied_scores():
    scores = [0.5, 0.5, 0.5, 0.1]
    y = [1, 0, 0, 0]
    amount = [100.0, 1.0, 1.0, 1.0]
    threshold, _ = metrics.best_global_threshold(scores, y, amount, 1.0)
    flagged = np.array(scores) >= threshold
    assert flagged[:3].all() or not flagged[:3].any()


def test_best_global_threshold_matches_brute_force():
    rng = np.random.default_rng(1)
    scores = rng.random(60)
    y = (rng.random(60) < 0.2).astype(int)
    amount = rng.uniform(1, 100, 60)
    _, cost = metrics.best_global_threshold(scores, y, amount, 7.0)
    candidates = [np.inf, *np.unique(scores)]
    brute = min(metrics.policy_cost(y, amount, scores >= t, 7.0) for t in candidates)
    assert cost == pytest.approx(brute)


def test_expected_cost_flag_rule():
    flag = metrics.expected_cost_flag([0.5, 0.01, 0.2], [100.0, 100.0, 10.0], 5.0)
    # 50 >= 5 flag, 1 < 5 no, 2 < 5 no.
    assert flag.tolist() == [True, False, False]


def test_platt_is_monotone_and_in_unit_interval():
    rng = np.random.default_rng(2)
    scores = rng.random(500)
    y = (rng.random(500) < scores**2).astype(int)
    calibrate = metrics.fit_platt(scores, y)
    grid = np.linspace(0.01, 0.99, 50)
    out = calibrate(grid)
    assert np.all(np.diff(out) >= -1e-12)
    assert out.min() >= 0 and out.max() <= 1


def test_calibration_ratio_hand_value():
    assert metrics.calibration_ratio([1, 0, 0, 1], [0.5, 0.25, 0.25, 0.5]) == pytest.approx(0.75)


def test_reliability_table_bins_cover_all_rows():
    rng = np.random.default_rng(3)
    probs = rng.random(100)
    y = (rng.random(100) < probs).astype(int)
    table = metrics.reliability_table(y, probs, n_bins=10)
    assert sum(r["n"] for r in table) == 100
    assert sum(r["fraud_count"] for r in table) == int(y.sum())


def test_bootstrap_is_deterministic_for_a_seed():
    rng = np.random.default_rng(4)
    y = (rng.random(300) < 0.1).astype(int)
    s = y * 0.5 + rng.random(300)

    def stat(idx):
        return metrics.average_precision(y[idx], s[idx])

    a = metrics.bootstrap_ci(stat, 300, n_boot=100, seed=7)
    b = metrics.bootstrap_ci(stat, 300, n_boot=100, seed=7)
    assert a == b
    assert a[0] <= a[1]
