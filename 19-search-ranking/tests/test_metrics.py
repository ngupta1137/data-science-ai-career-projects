import numpy as np
import pytest

from search_ranking import metrics


def test_dcg_hand_calculation():
    gains = [3, 2, 1, 0]
    # (2^3 - 1) / log2(2) = 7.0 / 1.0 = 7.0
    # (2^2 - 1) / log2(3) = 3.0 / 1.5849625 = 1.892789
    # (2^1 - 1) / log2(4) = 1.0 / 2.0 = 0.5
    # (2^0 - 1) / log2(5) = 0.0
    # Total = 7.0 + 1.892789 + 0.5 = 9.392789
    dcg = metrics.compute_dcg_at_k(gains, k=4)
    assert dcg == pytest.approx(9.392789, rel=1e-4)


def test_ndcg_properties():
    # Perfect ranking has NDCG = 1.0
    ideal = [3, 2, 1, 0]
    assert metrics.compute_ndcg_at_k(ideal, ideal, k=4) == pytest.approx(1.0)

    # Sub-optimal ranking has NDCG < 1.0
    sub = [0, 1, 2, 3]
    ndcg_sub = metrics.compute_ndcg_at_k(sub, ideal, k=4)
    assert 0.0 < ndcg_sub < 1.0

    # All zero gains returns 0.0
    zeros = [0, 0, 0, 0]
    assert metrics.compute_ndcg_at_k(zeros, zeros, k=4) == 0.0

    # Empty array returns 0.0
    assert metrics.compute_ndcg_at_k([], [], k=4) == 0.0


def test_bootstrap_paired_ci():
    rng = np.random.default_rng(42)
    # Generate synthetic candidate and baseline scores with true mean lift ~ 0.04
    base = rng.uniform(0.7, 0.9, size=100).tolist()
    cand = [b + float(rng.normal(0.04, 0.01)) for b in base]

    ci_low, ci_high = metrics.bootstrap_paired_lift_ci(cand, base, n_boot=200, seed=42)
    mean_diff = float(np.mean(cand) - np.mean(base))

    assert ci_low < mean_diff < ci_high
    assert ci_low > 0.0  # Confirms positive lower bound


def test_gate_conjunction():
    headline_eval = {
        "overall": {"true_ndcg": 0.88, "click_ndcg": 0.45, "ips_ndcg": 0.44},
        "query_true_ndcg": [0.88] * 50,
        "segments": {"tail": {"true_ndcg": 0.82}},
    }
    baseline_eval = {
        "overall": {"true_ndcg": 0.83, "click_ndcg": 0.45, "ips_ndcg": 0.43},
        "query_true_ndcg": [0.83] * 50,
        "segments": {"tail": {"true_ndcg": 0.80}},
    }

    res = metrics.evaluate_decision_gate(headline_eval, baseline_eval, seed=42)
    assert res["passed"] == all(res["checks"].values())
    assert res["checks"]["unbiased_ndcg_lift_meets_threshold"] is True
    assert res["checks"]["no_tail_query_degradation"] is True
