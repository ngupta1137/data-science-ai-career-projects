import json

import pytest

from llm_judge_eval import config, data, report, train


@pytest.fixture(scope="module")
def fast(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    return train.run(out, fast=True), out


def test_fast_run_is_marked_and_writes_files(fast):
    metrics, out = fast
    assert metrics["fast_mode"] is True
    for name in ("metrics.json", "error_analysis.json", "agreement.png"):
        assert (out / name).exists()
    analysis = json.loads((out / "error_analysis.json").read_text())
    assert "Not a release result" in report.render_model_card(metrics, analysis)
    assert "Not a release result" in report.render_decision_memo(metrics, analysis)


def test_run_is_deterministic(fast, tmp_path):
    again = train.run(tmp_path, fast=True)
    assert train._clean(again["candidates"]) == train._clean(fast[0]["candidates"])
    assert again["headline"] == fast[0]["headline"]


def test_oracle_agreement_equals_consensus_accuracy(fast):
    t = fast[0]["traps"]["human_label_is_not_truth"]
    assert (
        t["oracle_agreement_with_consensus"]["value"]
        == t["consensus_agreement_with_truth"]["value"]
    )
    assert t["oracle_agreement_with_consensus"]["value"] < 1.0


def test_coin_swap_matches_single_order_in_expectation(fast):
    c = fast[0]["candidates"]
    gap = (
        c["swap_coin"]["agreement_consensus"]["value"]
        - c["single_order"]["agreement_consensus"]["value"]
    )
    assert abs(gap) < 0.04


def test_triage_trades_coverage_for_agreement(fast):
    m = fast[0]
    tri = m["triage"]
    assert 0.5 < tri["coverage"] < 0.95
    assert (
        tri["agreement_consensus_on_decided"]["value"]
        > m["candidates"]["swap_coin"]["agreement_consensus"]["value"] + 0.05
    )
    assert abs(tri["agreement_consensus_on_split"]["value"] - 0.5) < 0.12


def test_fixed_slot_evaluation_inflates_agreement(fast):
    t = fast[0]["traps"]["fixed_slot_evaluation"]
    assert (
        t["agreement_when_consensus_winner_always_first"]["value"]
        > t["agreement_when_slots_balanced"]["value"] + 0.02
    )


def test_length_confound_moves_a_length_biased_judge(fast):
    c = fast[0]["traps"]["length_confounded_eval_set"]["by_length_quality_correlation"]
    assert c["rho=+0.3"]["swap_coin"]["value"] > c["rho=-0.3"]["swap_coin"]["value"] + 0.02
    assert c["rho=+0.3"]["length_only"]["value"] > 0.5 > c["rho=-0.3"]["length_only"]["value"]


def test_slot_bias_is_recovered(fast):
    b = fast[0]["bias"]
    assert b["recovered_with_true_quality"]["position"]["value"] == pytest.approx(
        b["planted"]["position"], abs=0.15
    )
    assert b["recovered_with_true_quality"]["quality_slope"]["value"] == pytest.approx(
        b["planted"]["quality_slope"], abs=0.25
    )
    assert b["recovered_with_true_quality"]["length_slope"]["value"] == pytest.approx(
        b["planted"]["length_slope"], abs=0.15
    )


def test_length_only_ranks_the_verbose_system_first(fast):
    t = fast[0]["traps"]["verbose_system_over_ranked"]
    assert t["length_only"]["order"][0] == "V"
    assert t["oracle"]["order"] == ["S5", "S4", "V", "S3", "S2", "S1"]
    assert max(t["oracle"]["share_of_resamples_v_beats"].values()) < config.OUTRANK_LIMIT


def test_gate_verdict_is_the_conjunction_of_its_checks(fast):
    g = fast[0]["gate"]
    assert g["passed"] == all(g["checks"].values())
    assert len(g["checks"]) == 3


def test_headline_is_picked_on_dev(fast):
    analysis = json.loads((fast[1] / "error_analysis.json").read_text())
    dev = analysis["dev_agreement"]
    head = analysis["headline_judge"]
    assert dev[head] == max(dev.values())


def test_eval_set_size_lower_bound_grows_with_n(fast):
    lb = fast[0]["traps"]["eval_set_size"]["wilson_lower_bound_by_n"]
    vals = [lb[str(n)] for n in config.EVAL_SIZES]
    assert vals == sorted(vals)


def test_n_needed():
    assert train.n_needed(0.75, 0.80) is None
    n = train.n_needed(0.85, 0.80)
    assert n is not None and n > 100


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
    again = train.run(tmp_path, fast=False, data_sha256=data.fingerprint())
    saved = json.loads(committed.read_text())
    cleaned = train._clean(again)
    _assert_close(cleaned["candidates"], saved["candidates"])
    _assert_close(cleaned["headline"], saved["headline"])
    assert cleaned["gate"]["checks"] == saved["gate"]["checks"]
