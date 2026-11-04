"""End-to-end smoke test on synthetic data. It proves the pipeline runs and the
output files have the right shape. It says nothing about real-data performance."""

import json

from conftest import make_synthetic_housing

from src import report, train


def test_pipeline_runs_and_writes_valid_results(tmp_path):
    df = make_synthetic_housing(n=2500, seed=11)
    metrics = train.run(
        df,
        tmp_path,
        fast=True,
        data_source="synthetic_test",
        expected_rows=None,
    )

    for name in ("metrics.json", "error_analysis.json", "test_predictions.csv",
                 "pred_vs_actual.png", "residuals_vs_predicted.png"):
        assert (tmp_path / name).exists(), name

    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["fast_mode"] is True
    assert saved["data_source"] == "synthetic_test"
    assert saved["headline"]["model"] in ("ridge", "hist_gradient_boosting")
    low, high = saved["headline"]["ci95"]
    assert low <= high
    assert saved["split"]["train_rows"] + saved["split"]["test_rows"] == 2500
    assert set(saved["models"]) == {
        "baseline_mean", "baseline_median", "ridge", "hist_gradient_boosting",
    }
    assert "random_split_test_rmse" in saved["leakage_trap"]
    assert isinstance(saved["gate"]["passed"], bool)
    assert metrics["headline"]["model"] == saved["headline"]["model"]


def test_model_beats_constant_baseline_on_learnable_synthetic_data(tmp_path):
    df = make_synthetic_housing(n=2500, seed=12)
    metrics = train.run(df, tmp_path, fast=True, data_source="synthetic_test", expected_rows=None)
    assert metrics["headline"]["value"] < metrics["best_baseline"]["test_rmse"]


def test_same_seed_gives_same_metrics(tmp_path):
    df = make_synthetic_housing(n=2000, seed=13)
    a = train.run(df, tmp_path / "a", fast=True, data_source="synthetic_test", expected_rows=None)
    b = train.run(df, tmp_path / "b", fast=True, data_source="synthetic_test", expected_rows=None)
    assert a["headline"] == b["headline"]
    assert a["models"] == b["models"]


def test_reports_render_and_flag_non_release_runs(tmp_path):
    df = make_synthetic_housing(n=2000, seed=14)
    train.run(df, tmp_path, fast=True, data_source="synthetic_test", expected_rows=None)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    analysis = json.loads((tmp_path / "error_analysis.json").read_text(encoding="utf-8"))
    card = report.render_model_card(metrics, analysis)
    memo = report.render_decision_memo(metrics, analysis)
    assert "Not a release result" in card
    assert "Not a release result" in memo
    assert chr(0x2014) not in card and chr(0x2014) not in memo
