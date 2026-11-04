"""Checks the committed results/metrics.json once it exists. Skipped before the first real run."""

import json

import pytest

from src import config

METRICS = config.RESULTS_DIR / "metrics.json"


@pytest.mark.skipif(not METRICS.exists(), reason="results not generated yet")
def test_committed_metrics_have_required_shape():
    saved = json.loads(METRICS.read_text(encoding="utf-8"))
    for key in ("project", "seed", "versions", "data", "split", "models", "headline", "gate", "leakage_trap"):
        assert key in saved
    assert saved["project"] == config.PROJECT_NAME
    assert saved["seed"] == config.SEED


@pytest.mark.skipif(not METRICS.exists(), reason="results not generated yet")
def test_committed_metrics_are_a_real_full_run():
    saved = json.loads(METRICS.read_text(encoding="utf-8"))
    assert saved["fast_mode"] is False
    assert saved["data_source"] == "california_housing"
    assert saved["data"]["rows"] == config.EXPECTED_ROWS
    assert saved["data"]["sha256"]
