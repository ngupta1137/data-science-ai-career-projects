import json

import pytest

from fairness_audit import config, report, train


@pytest.fixture(scope="module")
def fast(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    return train.run(out, fast=True), out


def test_fast_run_is_marked_and_writes_files(fast):
    metrics, out = fast
    assert metrics["fast_mode"] is True
    for name in ("metrics.json", "error_analysis.json", "tpr_by_group.png"):
        assert (out / name).exists()
    analysis = json.loads((out / "error_analysis.json").read_text())
    assert "Not a release result" in report.render_model_card(metrics, analysis)


def test_run_is_deterministic(fast, tmp_path):
    again = train.run(tmp_path, fast=True)
    # NaN (undefined precision when nobody is approved) never equals NaN, so compare cleaned.
    assert train._clean(again["policies"]) == train._clean(fast[0]["policies"])
    assert train._clean(again["headline"]) == train._clean(fast[0]["headline"])


def test_policy_stats_hand_example():
    import numpy as np

    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    approve = np.array([1, 0, 1, 0, 0, 0, 0, 1], dtype=bool)
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    util, tpr_gap, fpr_gap, ppv_gap, ratio = train.policy_stats(y, approve, g, cost_bad=2.0)
    assert util == pytest.approx((1 - 2 - 2) / 8)  # one good approved, two bad approved
    assert tpr_gap == pytest.approx(-0.5) and ratio == 0.5
    assert fpr_gap == pytest.approx(1 / 3 - 0.5) and ppv_gap == pytest.approx(-0.5)


def test_reject_all_has_zero_gap_and_zero_utility(fast):
    p = fast[0]["policies"]["reject_all"]
    assert p["tpr_gap"]["value"] == 0.0 and p["utility"]["value"] == 0.0


def test_unawareness_does_not_close_the_gap(fast):
    t = fast[0]["traps"]["fairness_through_unawareness"]
    assert abs(t["tpr_gap_unaware"]["value"]) > 0.05
    assert abs(t["tpr_gap_blind_no_proxy"]["value"]) > 0.05


def test_test_tuned_thresholds_look_better_than_dev_tuned(fast):
    t = fast[0]["traps"]["group_thresholds_tuned_on_test"]
    assert abs(t["tpr_gap_with_leak"]) <= abs(t["tpr_gap_honest_dev_tuned"]) + 1e-9


def test_headline_is_picked_on_dev_not_test(fast):
    analysis = json.loads((fast[1] / "error_analysis.json").read_text())
    head = analysis["headline_policy"]
    dev = analysis["dev_rows"]
    eligible = [
        n for n, r in dev.items() if r["utility"] > 0 and abs(r["tpr_gap"]) <= config.TPR_GAP_TOL
    ]
    if eligible:
        assert head in eligible
        assert dev[head]["utility"] == max(dev[n]["utility"] for n in eligible)


def test_gate_has_the_two_documented_checks(fast):
    assert set(fast[0]["gate"]["checks"]) == {
        "tpr_gap_within_tolerance",
        "beats_rejecting_everyone",
    }


def test_dev_size_study_spread_shrinks_with_more_dev_data(fast):
    rows = json.loads((fast[1] / "error_analysis.json").read_text())["post_hoc_dev_size_study"][
        "rows"
    ]
    assert rows[0]["sd_large_eval_tpr_gap"] > rows[-1]["sd_large_eval_tpr_gap"]


def test_committed_results_match_a_fresh_full_run(tmp_path):
    committed = config.RESULTS_DIR / "metrics.json"
    if not committed.exists():
        pytest.skip("no committed results")
    stored = json.loads(committed.read_text(encoding="utf-8"))
    if stored["fast_mode"]:
        pytest.skip("committed results are from a fast run")
    fresh = train.run(tmp_path, data_sha256=stored["data"]["sha256"])
    assert fresh["headline"]["policy"] == stored["headline"]["policy"]
    assert fresh["headline"]["value"] == pytest.approx(stored["headline"]["value"], abs=1e-12)
    assert fresh["gate"]["passed"] == stored["gate"]["passed"]
