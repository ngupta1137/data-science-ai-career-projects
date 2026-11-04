"""End-to-end smoke test on synthetic data. It proves the pipeline runs and the output
files have the right shape. It says nothing about real-data performance."""

import json

from demand_forecast import report, train
from demand_forecast.synthetic import make_synthetic_demand

RUN = {"fast": True, "data_source": "synthetic_test", "expected_rows": None}


def test_pipeline_runs_and_writes_valid_results(tmp_path):
    metrics = train.run(make_synthetic_demand(seed=1), tmp_path, **RUN)
    for name in ("metrics.json", "error_analysis.json", "forecast_vs_actual.png"):
        assert (tmp_path / name).exists(), name
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["fast_mode"] is True and saved["data_source"] == "synthetic_test"
    assert saved["headline"]["model"] in ("ridge", "hist_gradient_boosting")
    low, high = saved["headline"]["ci95"]
    assert low <= high
    bt = saved["backtest"]
    assert bt["tuning_origins"] + bt["test_origins"] == bt["origins"]
    assert bt["test_days"] == bt["test_origins"] * bt["horizon_days"]
    assert set(saved["traps"]) == {"same_day_components", "short_lag", "random_kfold"}
    assert isinstance(saved["gate"]["passed"], bool)
    assert metrics["headline"]["model"] == saved["headline"]["model"]


def test_headline_and_baseline_are_chosen_on_tuning_folds_only(tmp_path):
    metrics = train.run(make_synthetic_demand(seed=2), tmp_path, **RUN)
    tuned = {n: r["tuning_mae"] for n, r in metrics["models"].items()}
    head = metrics["headline"]["model"]
    base = metrics["best_baseline"]["model"]
    assert tuned[head] == min(tuned[n] for n in ("ridge", "hist_gradient_boosting"))
    assert tuned[base] == min(v for n, v in tuned.items() if n.startswith("baseline_"))


def test_same_day_leak_makes_the_score_implausibly_good(tmp_path):
    metrics = train.run(make_synthetic_demand(seed=3), tmp_path, **RUN)
    trap = metrics["traps"]["same_day_components"]
    assert trap["test_mae_with_leak"] < 0.5 * trap["test_mae_honest"]


def test_model_beats_naive_last_on_a_seasonal_series(tmp_path):
    metrics = train.run(make_synthetic_demand(seed=4), tmp_path, **RUN)
    assert (
        metrics["models"][metrics["headline"]["model"]]["test_mae"]
        < (metrics["models"]["baseline_naive_last"]["test_mae"])
    )


def test_same_seed_gives_same_metrics(tmp_path):
    df = make_synthetic_demand(seed=5)
    a = train.run(df, tmp_path / "a", **RUN)
    b = train.run(df, tmp_path / "b", **RUN)
    assert a["headline"] == b["headline"] and a["models"] == b["models"]


def test_too_short_a_series_is_an_error(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="origins"):
        train.run(make_synthetic_demand(n=380, seed=6), tmp_path, **RUN)


def test_reports_render_and_flag_non_release_runs(tmp_path):
    train.run(make_synthetic_demand(seed=7), tmp_path, **RUN)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    analysis = json.loads((tmp_path / "error_analysis.json").read_text(encoding="utf-8"))
    card = report.render_model_card(metrics, analysis)
    memo = report.render_decision_memo(metrics, analysis)
    assert "Not a release result" in card and "Not a release result" in memo
    assert chr(0x2014) not in card and chr(0x2014) not in memo
