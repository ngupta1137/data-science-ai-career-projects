import json

import numpy as np
import pytest

from recsys_eval import config, data, report, train


@pytest.fixture(scope="module")
def fast(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    return train.run(out, fast=True), out


def test_fast_run_is_marked_and_writes_files(fast):
    metrics, out = fast
    assert metrics["fast_mode"] is True
    for name in ("metrics.json", "error_analysis.json", "observed_vs_true.png"):
        assert (out / name).exists()
    analysis = json.loads((out / "error_analysis.json").read_text())
    assert "Not a release result" in report.render_model_card(metrics, analysis)
    assert "Not a release result" in report.render_decision_memo(metrics, analysis)


def test_run_is_deterministic(fast, tmp_path):
    again = train.run(tmp_path, fast=True)
    assert train._clean(again["models"]) == train._clean(fast[0]["models"])


def test_reference_points_of_true_value(fast):
    m = fast[0]["models"]
    assert m["random"]["value_captured"] == pytest.approx(0.0)
    assert m["oracle"]["value_captured"] == pytest.approx(1.0)
    assert m["oracle"]["true_like10"]["value"] > m["svd_ips"]["true_like10"]["value"]


def test_oracle_looks_bad_on_the_observed_metric(fast):
    m = fast[0]["models"]
    assert m["oracle"]["ndcg10"]["value"] < m["popularity"]["ndcg10"]["value"]
    assert m["oracle"]["true_like10"]["value"] > 3 * m["popularity"]["true_like10"]["value"]


def test_popularity_recommends_a_few_items_to_everyone(fast):
    m = fast[0]["models"]
    assert m["popularity"]["coverage10"] < 0.2 < m["random"]["coverage10"]
    assert m["popularity"]["gini10"] > 0.9


def test_exposure_bias_breaks_the_offline_metric(fast):
    sweep = {
        r["exposure_strength"]: r["spearman_observed_vs_true"]
        for r in fast[0]["traps"]["exposure_sweep"]
    }
    assert sweep[0.0] > 0.7 > 0 > sweep[1.5]


def test_random_split_scores_higher_than_temporal(fast):
    t = fast[0]["traps"]["random_split_inflates_scores"]
    names = ("popularity", "item_knn", "svd", "svd_ips")
    assert np.mean([t["ndcg10_random_split"][n] for n in names]) > np.mean(
        [t["ndcg10_temporal_test"][n] for n in names]
    )


def test_sampled_hit_rate_is_inflated_for_poor_rankers(fast):
    by = fast[0]["traps"]["sampled_metrics"]["by_model"]
    assert by["random"]["sampled_hit10"] > by["random"]["full_hit10"] + 0.02
    assert by["oracle"]["sampled_hit10"] > by["oracle"]["full_hit10"] + 0.02


def test_gate_verdict_is_the_conjunction_of_its_checks(fast):
    g = fast[0]["gate"]
    assert g["passed"] == all(g["checks"].values()) and len(g["checks"]) == 3


def test_headline_is_picked_on_validation(fast):
    analysis = json.loads((fast[1] / "error_analysis.json").read_text())
    val = analysis["validation_ndcg10"]
    assert val[analysis["headline_model"]] == max(val.values())
    assert set(val) == set(config.CANDIDATES)


def test_eligible_users_have_history_and_holdout(fast):
    analysis = json.loads((fast[1] / "error_analysis.json").read_text())
    assert 500 < analysis["eligible_users"] <= config.N_USERS


def test_full_run_reproduces_committed_results(tmp_path):
    committed = config.RESULTS_DIR / "metrics.json"
    if not committed.exists() or json.loads(committed.read_text())["fast_mode"]:
        pytest.skip("no committed full-run results yet")
    again = train.run(tmp_path, fast=False, data_sha256=data.fingerprint())
    saved = json.loads(committed.read_text())
    cleaned = train._clean(again)
    assert cleaned["models"] == saved["models"]
    assert cleaned["gate"]["checks"] == saved["gate"]["checks"]
