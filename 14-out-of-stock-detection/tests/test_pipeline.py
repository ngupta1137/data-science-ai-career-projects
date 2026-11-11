import json

import pytest

from oos_detect import config, data, report, train


@pytest.fixture(scope="module")
def fast(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    return train.run(out, fast=True), out


def _a(out):
    return json.loads((out / "error_analysis.json").read_text())


def test_fast_run_is_marked_and_writes_files(fast):
    m, out = fast
    assert m["fast_mode"] is True
    for name in ("metrics.json", "error_analysis.json", "net_value.png"):
        assert (out / name).exists()
    assert "Not a release result" in report.render_model_card(m, _a(out))
    assert "Not a release result" in report.render_decision_memo(m, _a(out))


def test_deterministic(fast, tmp_path):
    assert train._clean(train.run(tmp_path, fast=True)["policies"]) == train._clean(
        fast[0]["policies"]
    )


def test_oracle_is_best_and_no_visits_is_zero(fast):
    p = fast[0]["policies"]
    assert p["no_visits"]["net_value_per_store_day"]["value"] == 0
    assert all(
        p["oracle"]["net_value_per_store_day"]["value"] >= v["net_value_per_store_day"]["value"]
        for v in p.values()
    )


def test_zero_run_rule_floods_reps_with_slow_sellers(fast):
    slow = _a(fast[1])["demand_bands"][0]
    assert slow["zero_run_rule"]["share_of_visits"] > 0.5
    assert (
        fast[0]["traps"]["same_rule_for_every_shelf"]["zero_run_rule_flagged_per_day"]
        > fast[0]["traps"]["same_rule_for_every_shelf"]["capacity_per_day"]
    )


def test_gate_is_conjunction(fast):
    g = fast[0]["gate"]
    assert g["passed"] == all(g["checks"].values()) and len(g["checks"]) == 3


def test_full_run_reproduces_committed_results(tmp_path):
    committed = config.RESULTS_DIR / "metrics.json"
    if not committed.exists() or json.loads(committed.read_text())["fast_mode"]:
        pytest.skip("no committed full-run results yet")
    again = train._clean(train.run(tmp_path, fast=False, data_sha256=data.fingerprint()))
    saved = json.loads(committed.read_text())
    assert again["policies"] == saved["policies"]
    assert again["gate"] == saved["gate"]
