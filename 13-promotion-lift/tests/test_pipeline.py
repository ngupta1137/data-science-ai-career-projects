import json

import pytest

from promo_lift import config, data, report, train


@pytest.fixture(scope="module")
def fast(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    return train.run(out, fast=True), out


def _analysis(out):
    return json.loads((out / "error_analysis.json").read_text())


def test_fast_run_is_marked_and_writes_files(fast):
    m, out = fast
    assert m["fast_mode"] is True
    for name in ("metrics.json", "error_analysis.json", "weekly_sales.png"):
        assert (out / name).exists()
    assert "Not a release result" in report.render_model_card(m, _analysis(out))
    assert "Not a release result" in report.render_decision_memo(m, _analysis(out))


def test_deterministic(fast, tmp_path):
    again = train.run(tmp_path, fast=True)
    assert train._clean(again["estimators"]) == train._clean(fast[0]["estimators"])


def test_shortcuts_are_biased_upward_and_full_did_is_not(fast):
    s = _analysis(fast[1])["method_study"]["estimators"]
    assert s["pre_post"]["bias"] > 50 and s["window_did"]["bias"] > 30
    assert abs(s["full_did"]["bias"]) < 10


def test_manager_picked_stores_bias_the_right_method(fast):
    t = fast[0]["traps"]["manager_picked_stores"]
    assert t["bias_full_did_picked"] > t["bias_full_did_random"] + 30


def test_more_stores_give_more_definite_calls(fast):
    rows = _analysis(fast[1])["stores_needed"]
    assert rows[-1]["share_unclear"] < rows[0]["share_unclear"]


def test_gate_is_conjunction(fast):
    g = fast[0]["gate"]
    assert g["passed"] == all(g["checks"].values()) and len(g["checks"]) == 3


def test_full_run_reproduces_committed_results(tmp_path):
    committed = config.RESULTS_DIR / "metrics.json"
    if not committed.exists() or json.loads(committed.read_text())["fast_mode"]:
        pytest.skip("no committed full-run results yet")
    again = train._clean(train.run(tmp_path, fast=False, data_sha256=data.fingerprint()))
    saved = json.loads(committed.read_text())
    assert again["estimators"] == saved["estimators"]
    assert again["gate"] == saved["gate"]
