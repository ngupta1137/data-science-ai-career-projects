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


def _assert_close(a, b, path="result"):
    """Equal up to floating-point noise. Booleans, strings and counts must match exactly.

    Numerical libraries can differ in the last digits between machines, so floats get a
    relative tolerance of 1e-6. Gate checks are compared exactly, so no verdict can move.
    """
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-6, abs=1e-9), path
    else:
        assert a == b, path


def test_full_run_reproduces_committed_results(tmp_path):
    committed = config.RESULTS_DIR / "metrics.json"
    if not committed.exists() or json.loads(committed.read_text())["fast_mode"]:
        pytest.skip("no committed full-run results yet")
    again = train._clean(train.run(tmp_path, fast=False, data_sha256=data.fingerprint()))
    saved = json.loads(committed.read_text())
    _assert_close(again["estimators"], saved["estimators"])
    assert again["gate"] == saved["gate"]
