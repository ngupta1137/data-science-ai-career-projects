"""Checks the committed results/metrics.json once it exists. Skipped before the first real run."""

import json

import pytest

from demand_forecast import config

METRICS = config.RESULTS_DIR / "metrics.json"


@pytest.mark.skipif(not METRICS.exists(), reason="results not generated yet")
def test_committed_metrics_have_required_shape():
    saved = json.loads(METRICS.read_text(encoding="utf-8"))
    for key in (
        "project",
        "seed",
        "versions",
        "data",
        "backtest",
        "models",
        "headline",
        "gate",
        "traps",
    ):
        assert key in saved
    assert saved["project"] == config.PROJECT_NAME and saved["seed"] == config.SEED


@pytest.mark.skipif(not METRICS.exists(), reason="results not generated yet")
def test_committed_metrics_are_a_real_full_run():
    saved = json.loads(METRICS.read_text(encoding="utf-8"))
    assert saved["fast_mode"] is False
    assert saved["data_source"] == "bike_sharing_uci"
    assert saved["data"]["rows"] == config.EXPECTED_ROWS
    assert saved["data"]["sha256"]
