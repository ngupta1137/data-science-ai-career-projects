import json

import pytest

from price_elasticity import config, report, train


@pytest.fixture(scope="module")
def fast_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("results_fast")
    metrics, analysis = train.run_pipeline(results_dir=out, fast_mode=True)
    return metrics, analysis, out


def test_fast_run_is_marked_and_writes_files(fast_run):
    metrics, analysis, out = fast_run
    assert metrics["fast_mode"] is True
    for fname in ("metrics.json", "error_analysis.json", "elasticity_curve.png"):
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

    # Trap 1: Simultaneity / Endogeneity bias
    t1 = traps["simultaneity_bias"]
    assert t1["price_demand_shock_correlation"] > 0.30
    assert t1["first_stage_f_statistic"] > 50.0
    assert abs(t1["two_sls_elasticity"] - t1["true_elasticity"]) < abs(
        t1["naive_ols_elasticity"] - t1["true_elasticity"]
    )

    # Trap 2: Profit destruction from naive pricing
    t2 = traps["profit_destruction"]
    assert t2["naive_profit_loss_pct"] > 15.0
    assert t2["two_sls_profit_loss_pct"] < 5.0
    assert t2["two_sls_recovered_profit"] > t2["naive_recovered_profit"]

    # Trap 3: Omitted variable bias without seasonality
    t3 = traps["omitted_variable_bias"]
    # Omitting seasonality adds positive bias (attenuates elasticity closer to 0)
    assert t3["additional_bias_from_omission"] > 0.0


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
    _assert_close(again["traps"], committed["traps"])
    assert again["gate"] == committed["gate"]
