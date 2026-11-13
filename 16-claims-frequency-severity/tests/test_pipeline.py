import json

import pytest

from claims_pricing import config, report, train


@pytest.fixture(scope="module")
def fast_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("results_fast")
    metrics, analysis = train.run_pipeline(results_dir=out, fast_mode=True)
    return metrics, analysis, out


def test_fast_run_is_marked_and_writes_files(fast_run):
    metrics, analysis, out = fast_run
    assert metrics["fast_mode"] is True
    for fname in ("metrics.json", "error_analysis.json", "loss_ratio_by_decile.png"):
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

    # Trap 1: naive count model has inflated loss ratio on short exposure
    t1 = traps["ignoring_exposure_time"]
    assert t1["no_exposure_lr_short"] > t1["two_part_lr_short"]

    # Trap 2: flat pricing has higher loss ratio in decile 10 than decile 1
    t2 = traps["loss_ratio_by_decile"]
    assert t2["flat_decile_10_loss_ratio"] > t2["flat_decile_1_loss_ratio"]

    # Trap 3: two-part model has lower pure premium RMSE than single Tweedie
    t3 = traps["one_model_for_frequency_and_size"]
    assert t3["two_part_rmse_vs_truth"] < t3["single_tweedie_rmse_vs_truth"]


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
        assert a == pytest.approx(b, rel=1e-5, abs=1e-8), path
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
    _assert_close(again["policies"], committed["policies"])
    _assert_close(again["headline"], committed["headline"])
    _assert_close(again["traps"], committed["traps"])
    assert again["gate"] == committed["gate"]
