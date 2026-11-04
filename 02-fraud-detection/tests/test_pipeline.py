"""End-to-end smoke test on synthetic data. It proves the pipeline runs and the output
files have the right shape. It says nothing about real-data performance."""

import json

import pytest

from fraud_detection import config, report, train
from fraud_detection.synthetic import make_synthetic_fraud

RUN = {"fast": True, "data_source": "synthetic_test", "expected_rows": None, "expected_fraud": None}


def test_pipeline_runs_and_writes_valid_results(tmp_path):
    df = make_synthetic_fraud(n=12000, seed=11)
    metrics = train.run(df, tmp_path, **RUN)

    for name in ("metrics.json", "error_analysis.json", "precision_recall.png", "calibration.png"):
        assert (tmp_path / name).exists(), name

    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["fast_mode"] is True
    assert saved["data_source"] == "synthetic_test"
    assert saved["headline"]["model"] in ("logistic", "hist_gradient_boosting")
    low, high = saved["headline"]["ci95"]
    assert low <= high
    split = saved["split"]
    assert split["fit_rows"] + split["val_rows"] + split["test_rows"] == 12000
    assert set(saved["models"]) == {"baseline_prior", "logistic", "hist_gradient_boosting"}
    assert set(saved["cost_by_review_cost"]) == {str(c) for c in config.REVIEW_COSTS}
    assert set(saved["traps"]) == {"accuracy", "threshold_on_test", "random_split"}
    assert isinstance(saved["gate"]["passed"], bool)
    assert metrics["headline"]["model"] == saved["headline"]["model"]


def test_baseline_pr_auc_equals_test_fraud_rate(tmp_path):
    df = make_synthetic_fraud(n=12000, seed=12)
    metrics = train.run(df, tmp_path, **RUN)
    assert metrics["baseline"]["test_pr_auc"] == pytest.approx(metrics["split"]["test_fraud_rate"])


def test_model_beats_baseline_on_learnable_synthetic_data(tmp_path):
    df = make_synthetic_fraud(n=12000, seed=13)
    metrics = train.run(df, tmp_path, **RUN)
    assert metrics["headline"]["value"] > metrics["baseline"]["test_pr_auc"]
    assert metrics["gate"]["pr_auc_interval_above_baseline"] is True


def test_oracle_threshold_is_never_worse_than_validation_threshold(tmp_path):
    df = make_synthetic_fraud(n=12000, seed=14)
    metrics = train.run(df, tmp_path, **RUN)
    for entry in metrics["cost_by_review_cost"].values():
        assert (
            entry["oracle_threshold_tuned_on_test"] <= entry["global_threshold_tuned_on_val"] + 1e-9
        )
        assert entry["oracle_threshold_tuned_on_test"] <= entry["best_trivial"] + 1e-9


def test_same_seed_gives_same_metrics(tmp_path):
    df = make_synthetic_fraud(n=10000, seed=15)
    a = train.run(df, tmp_path / "a", **RUN)
    b = train.run(df, tmp_path / "b", **RUN)
    assert a["headline"] == b["headline"]
    assert a["models"] == b["models"]
    assert a["cost_by_review_cost"] == b["cost_by_review_cost"]


def test_too_few_fraud_rows_in_a_split_is_an_error(tmp_path):
    df = make_synthetic_fraud(n=3000, fraud_rate=0.004, seed=16)
    with pytest.raises(ValueError, match="fewer than"):
        train.run(df, tmp_path, **RUN)


def test_reports_render_and_flag_non_release_runs(tmp_path):
    df = make_synthetic_fraud(n=10000, seed=17)
    train.run(df, tmp_path, **RUN)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    analysis = json.loads((tmp_path / "error_analysis.json").read_text(encoding="utf-8"))
    card = report.render_model_card(metrics, analysis)
    memo = report.render_decision_memo(metrics, analysis)
    assert "Not a release result" in card and "Not a release result" in memo
    assert chr(0x2014) not in card and chr(0x2014) not in memo
