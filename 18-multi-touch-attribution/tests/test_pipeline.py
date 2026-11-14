import json

import pytest

from touch_attribution import config, report, train


@pytest.fixture(scope="module")
def fast_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("results_fast")
    metrics, analysis = train.run_pipeline(results_dir=out, fast_mode=True)
    return metrics, analysis, out


def test_fast_run_is_marked_and_writes_files(fast_run):
    metrics, analysis, out = fast_run
    assert metrics["fast_mode"] is True
    for fname in ("metrics.json", "error_analysis.json", "attribution_comparison.png"):
        assert (out / fname).exists()

    mc = report.render_model_card(metrics, analysis)
    dm = report.render_decision_memo(metrics, analysis)
    assert "Not a release result" in mc
    assert "Not a release result" in dm


def test_gate_is_conjunction(fast_run):
    metrics, _, _ = fast_run
    gate = metrics["gate"]
    assert len(gate["checks"]) == 3
    assert gate["passed"] == all(gate["checks"].values())


def test_traps_show_claimed_directions(fast_run):
    metrics, _, _ = fast_run
    traps = metrics["traps"]

    # Trap 1: Last touch starves top-of-funnel Display and over-credits bottom-of-funnel Email
    t1 = traps["last_touch_bias"]
    assert t1["display_last_touch_share"] < t1["display_true_credit_share"]
    assert t1["email_last_touch_share"] > t1["email_true_credit_share"]

    # Trap 2: Substantial share of conversions are organic baseline
    t2 = traps["ignoring_organic_baseline"]
    assert t2["organic_conversion_share"] > 0.20
    assert t2["expected_organic_conversions"] > 0

    # Trap 3: Linear attribution suffers positive share error from equal dilution
    t3 = traps["linear_dilution"]
    assert t3["linear_share_mae"] > 0.01


def test_deterministic(tmp_path):
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    m1, _ = train.run_pipeline(results_dir=out1, fast_mode=True)
    m2, _ = train.run_pipeline(results_dir=out2, fast_mode=True)
    assert m1["headline"]["value"] == m2["headline"]["value"]
    assert m1["gate"]["passed"] == m2["gate"]["passed"]


def _assert_close(a, b, path="result"):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        assert a == pytest.approx(b, rel=0.15, abs=0.10), path
    else:
        assert a == b, path


def test_full_run_reproduces_committed_results(tmp_path):
    committed_path = config.RESULTS_DIR / "metrics.json"
    if not committed_path.exists():
        pytest.skip("No committed results yet")
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    if committed.get("fast_mode", False):
        pytest.skip("Committed results are marked fast_mode")

    out = tmp_path / "full_reproduce"
    again, _ = train.run_pipeline(results_dir=out, fast_mode=False)
    _assert_close(again["models"], committed["models"])
    _assert_close(again["headline"], committed["headline"])
    _assert_close(again["budget_allocation"], committed["budget_allocation"])
    _assert_close(again["traps"], committed["traps"])
    assert again["gate"] == committed["gate"]
