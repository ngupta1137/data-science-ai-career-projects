import json

import pytest

from marketing_mix import config, report, train


@pytest.fixture(scope="module")
def fast_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("results_fast")
    metrics, analysis = train.run_pipeline(results_dir=out, fast_mode=True)
    return metrics, analysis, out


def test_fast_run_is_marked_and_writes_files(fast_run):
    metrics, analysis, out = fast_run
    assert metrics["fast_mode"] is True
    for fname in ("metrics.json", "error_analysis.json", "response_curves.png"):
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

    # Trap 1: naive regression ignores adstock carryover (0.0 vs true positive carryover)
    t1 = traps["ignoring_adstock"]
    assert t1["tv_true_alpha"] > t1["tv_estimated_alpha_naive"]
    assert t1["tv_true_half_life_weeks"] > t1["tv_naive_half_life_weeks"]

    # Trap 2: linear projection exceeds true saturation lift at 2x spend
    t2 = traps["ignoring_saturation"]
    assert t2["linear_projected_lift_ratio_at_2x"] > t2["true_saturation_lift_ratio_at_2x"]

    # Trap 3: correlated spend distorts naive slope compared to MMM allocation
    t3 = traps["correlated_spend_bias"]
    assert t3["tv_search_spend_correlation"] > 0.40
    assert t3["mmm_search_recommended_spend"] > t3["naive_search_recommended_spend"]


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
        assert a == pytest.approx(b, rel=1e-2, abs=1e-3), path
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
