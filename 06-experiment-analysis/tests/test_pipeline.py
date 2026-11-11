import json

import pytest

from experiment_analysis import config, report, train


@pytest.fixture(scope="module")
def fast(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    return train.run(out, fast=True), out


def test_fast_run_is_marked_and_writes_files(fast):
    metrics, out = fast
    assert metrics["fast_mode"] is True
    for name in ("metrics.json", "error_analysis.json", "peeking_false_positives.png"):
        assert (out / name).exists()
    assert "Not a release result" in report.render_model_card(
        metrics, json.loads((out / "error_analysis.json").read_text())
    )


def test_run_is_deterministic(fast, tmp_path):
    again = train.run(tmp_path, fast=True)
    assert again["fixed_horizon_grid"] == fast[0]["fixed_horizon_grid"]
    assert again["peeking"] == fast[0]["peeking"]


def test_peeking_is_worse_than_bonferroni_and_grows_with_looks(fast):
    rows = fast[0]["peeking"]
    assert rows[0]["naive_false_positive"]["rate"] == rows[0]["bonferroni_false_positive"]["rate"]
    assert rows[-1]["naive_false_positive"]["rate"] > rows[-1]["bonferroni_false_positive"]["rate"]
    assert rows[-1]["naive_false_positive"]["rate"] > rows[0]["naive_false_positive"]["rate"]


def test_contaminated_covariate_biases_cuped(fast):
    cc = fast[0]["traps"]["contaminated_covariate"]
    assert cc["cuped_mean_error_contaminated_covariate"] < -0.1
    assert abs(cc["cuped_mean_error_clean_covariate"]) < 0.05


def test_uncorrected_metrics_inflate_family_wise_error(fast):
    t = fast[0]["traps"]["uncorrected_metrics"]
    assert t["uncorrected_family_wise_error"]["rate"] > 3 * t["holm_family_wise_error"]["rate"]


def test_gate_has_the_four_documented_checks(fast):
    assert set(fast[0]["gate"]["checks"]) == {
        "aa_false_positive_contains_nominal",
        "cuped_unbiased",
        "cuped_variance_ratio_below_one",
        "holm_family_wise_error_controlled",
    }


def test_committed_results_match_a_fresh_full_run(tmp_path):
    committed = config.RESULTS_DIR / "metrics.json"
    if not committed.exists():
        pytest.skip("no committed results")
    stored = json.loads(committed.read_text(encoding="utf-8"))
    if stored["fast_mode"]:
        pytest.skip("committed results are from a fast run")
    fresh = train.run(tmp_path, data_sha256=stored["data"]["sha256"])
    assert fresh["headline"]["value"] == pytest.approx(stored["headline"]["value"], abs=1e-9)
    assert fresh["gate"]["passed"] == stored["gate"]["passed"]
    assert fresh["peeking"][2]["naive_false_positive"]["rate"] == pytest.approx(
        stored["peeking"][2]["naive_false_positive"]["rate"], abs=1e-9
    )
