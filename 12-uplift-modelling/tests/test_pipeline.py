import json

import pytest

from uplift_eval import config, data, report, train


@pytest.fixture(scope="module")
def fast(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    return train.run(out, fast=True), out


def test_fast_run_is_marked_and_writes_files(fast):
    metrics, out = fast
    assert metrics["fast_mode"] is True
    for name in ("metrics.json", "error_analysis.json", "qini.png"):
        assert (out / name).exists()
    analysis = json.loads((out / "error_analysis.json").read_text())
    assert "Not a release result" in report.render_model_card(metrics, analysis)
    assert "Not a release result" in report.render_decision_memo(metrics, analysis)


def test_run_is_deterministic(fast, tmp_path):
    again = train.run(tmp_path, fast=True)
    assert train._clean(again["policies"]) == train._clean(fast[0]["policies"])


def test_reference_policies(fast):
    p = fast[0]["policies"]
    assert (
        p["treat_none"]["true_profit_population"] == 0.0 and p["treat_none"]["treat_share"] == 0.0
    )
    assert p["treat_all"]["treat_share"] == 1.0
    assert (
        p["oracle"]["true_profit_population"]
        >= max(p[n]["true_profit_population"] for n in config.CANDIDATES) - 1e-9
    )


def test_uplift_models_beat_blanket_offers_on_true_profit(fast):
    p = fast[0]["policies"]
    for n in config.CANDIDATES:
        assert p[n]["true_profit_population"] > p["treat_all"]["true_profit_population"] + 0.5


def test_response_model_loses_to_uplift_model_at_the_same_budget(fast):
    t = fast[0]["traps"]["response_model_instead_of_uplift"]
    assert t["true_profit_uplift_model"] > t["true_profit_response_model"] + 0.3
    assert t["true_profit_response_model"] > t["true_profit_random"]


def test_blanket_offers_harm_a_large_segment(fast, tmp_path):
    analysis = json.loads((fast[1] / "error_analysis.json").read_text())
    harmed = analysis["segments"][0]
    assert harmed["share_of_customers"] > 0.2 and harmed["profit_per_customer_if_treat_all"] < -1
    assert harmed["headline_treats"] < 0.15


def test_training_on_biased_offers_costs_profit(fast):
    rows = fast[0]["traps"]["offers_that_followed_buying_signals"]
    assert rows["naive_effect_in_observational_data"] > 2 * rows["true_average_effect"]
    worse = [
        v["trained_on_observational"] < v["trained_on_trial"]
        for v in rows["true_profit_population"].values()
    ]
    assert sum(worse) >= 3


def test_simple_learner_is_competitive_with_little_data(fast):
    small = fast[0]["traps"]["data_needed"][0]
    assert small["train_customers"] == 2000
    assert small["mean_true_profit"]["s_linear"] > small["mean_true_profit"]["ct_gbm"]


def test_higher_offer_cost_shrinks_the_treated_share(fast):
    rows = json.loads((fast[1] / "error_analysis.json").read_text())["cost_sensitivity"]
    shares = [r["treat_share"] for r in rows]
    assert shares == sorted(shares, reverse=True)


def test_gate_verdict_is_the_conjunction_of_its_checks(fast):
    g = fast[0]["gate"]
    assert g["passed"] == all(g["checks"].values()) and len(g["checks"]) == 3


def test_headline_is_picked_on_dev(fast):
    analysis = json.loads((fast[1] / "error_analysis.json").read_text())
    dev = analysis["dev_auuc"]
    assert dev[analysis["headline_learner"]] == max(dev.values())


def _assert_close(a, b, path="policies"):
    """Equal up to floating-point noise. Booleans, strings and counts must match exactly.

    Gradient boosting can differ in the last digits between machines, so floats get a
    relative tolerance of 1e-6. This does not loosen any gate: the gate checks are compared exactly.
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
    _assert_close(cleaned["policies"], saved["policies"])
    assert cleaned["gate"]["checks"] == saved["gate"]["checks"]
