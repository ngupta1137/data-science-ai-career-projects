import json

import numpy as np
import pytest

from drift_monitor import config, report, train


@pytest.fixture(scope="module")
def fast(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    return train.run(out, fast=True), out


def test_fast_run_is_marked_and_writes_files(fast):
    metrics, out = fast
    assert metrics["fast_mode"] is True
    for name in ("metrics.json", "error_analysis.json", "detection_by_shift_size.png"):
        assert (out / name).exists()
    analysis = json.loads((out / "error_analysis.json").read_text())
    assert "Not a release result" in report.render_model_card(metrics, analysis)


def test_run_is_deterministic(fast, tmp_path):
    again = train.run(tmp_path, fast=True)
    assert again["results"] == fast[0]["results"]
    assert again["thresholds"] == fast[0]["thresholds"]


def test_calibrate_hits_the_budget_in_sample():
    rng = np.random.default_rng(0)
    stats = {m: rng.random((200, 52)) for m in config.MONITORS}
    thr = train.calibrate(stats, budget=0.05)
    for m in config.MONITORS:
        assert (stats[m].max(axis=1) > thr[m]).mean() <= 0.05


def test_evaluate_hand_example():
    # three streams, 6 weeks, onset 3. Stream 0 alarms at week 4, stream 1 never, stream 2 at week 1.
    stat = np.zeros((3, 6))
    stat[0, 4] = 1.0
    stat[2, 1] = 1.0
    out = train.evaluate(stat, thr=0.5, lag=2, onset=3)
    assert out["detection"]["rate"] == pytest.approx(1 / 3)
    assert out["median_delay_weeks"] == 1 + 2  # first alarm one week after onset, plus lag
    assert out["pre_onset_false_alarm"]["rate"] == pytest.approx(1 / 3)
    assert train.evaluate(stat, 0.5, 0, None)["false_alarm"]["rate"] == pytest.approx(2 / 3)


def test_evaluate_treats_nan_as_no_alarm():
    stat = np.full((2, 5), np.nan)
    assert train.evaluate(stat, 0.1, 0, None)["false_alarm"]["rate"] == 0.0


def test_concept_drift_is_only_seen_with_labels(fast):
    res = fast[0]["results"]
    assert res["auc_drop"]["concept_drift"]["detection"]["rate"] > 0.8
    assert res["psi_max"]["concept_drift"]["detection"]["rate"] < 0.5


def test_fixed_psi_false_alarms_at_small_batches(fast):
    fixed = fast[0]["traps"]["fixed_psi_rule_of_thumb"]["by_threshold"]["0.2"]
    assert fixed["false_alarm_batch_100"]["rate"] > fixed["false_alarm_batch_500"]["rate"]


def test_gate_has_the_two_documented_checks(fast):
    assert set(fast[0]["gate"]["checks"]) == {"false_alarm_within_limit", "abrupt_shift_detected"}


def test_committed_results_match_a_fresh_full_run(tmp_path):
    committed = config.RESULTS_DIR / "metrics.json"
    if not committed.exists():
        pytest.skip("no committed results")
    stored = json.loads(committed.read_text(encoding="utf-8"))
    if stored["fast_mode"]:
        pytest.skip("committed results are from a fast run")
    fresh = train.run(tmp_path, data_sha256=stored["data"]["sha256"])
    assert fresh["thresholds"] == pytest.approx(stored["thresholds"], abs=1e-9)
    assert fresh["headline"]["monitor"] == stored["headline"]["monitor"]
    assert fresh["headline"]["false_alarm_rate"]["rate"] == pytest.approx(
        stored["headline"]["false_alarm_rate"]["rate"], abs=1e-12
    )
    assert fresh["gate"]["passed"] == stored["gate"]["passed"]
