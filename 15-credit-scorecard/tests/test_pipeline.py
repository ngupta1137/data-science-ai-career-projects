import json

import pytest

from credit_score import config, data, report, train


@pytest.fixture(scope="module")
def fast(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    return train.run(out, fast=True), out


def _a(out):
    return json.loads((out / "error_analysis.json").read_text())


def test_fast_run_is_marked_and_writes_files(fast):
    m, out = fast
    assert m["fast_mode"] is True
    for name in ("metrics.json", "error_analysis.json", "calibration.png"):
        assert (out / name).exists()
    assert "Not a release result" in report.render_model_card(m, _a(out))
    assert "Not a release result" in report.render_decision_memo(m, _a(out))


def test_accepted_only_under_predicts_on_declined(fast):
    t = fast[0]["traps"]["evaluate_on_approved_only"]
    assert t["calibration_old_declined"] < 0.85


def test_recalibration_moves_toward_truth(fast):
    mm = fast[0]["models"]
    assert abs(mm["recalibrated"]["calibration_all"] - 1) < abs(mm["accepted_only"]["calibration_all"] - 1)


def test_oracle_is_best(fast):
    mm = fast[0]["models"]
    assert all(mm["oracle"]["profit_per_applicant"]["value"] >= r["profit_per_applicant"]["value"] for r in mm.values())


def test_gate_is_conjunction(fast):
    g = fast[0]["gate"]
    assert g["passed"] == all(g["checks"].values()) and len(g["checks"]) == 3


def test_full_run_reproduces_committed_results(tmp_path):
    committed = config.RESULTS_DIR / "metrics.json"
    if not committed.exists() or json.loads(committed.read_text())["fast_mode"]:
        pytest.skip("no committed full-run results yet")
    again = train._clean(train.run(tmp_path, fast=False, data_sha256=data.fingerprint()))
    saved = json.loads(committed.read_text())
    assert again["models"] == saved["models"]
    assert again["gate"] == saved["gate"]
