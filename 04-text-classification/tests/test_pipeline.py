"""End-to-end smoke test on synthetic data. It proves the pipeline runs and the output
files have the right shape. It says nothing about real-data performance."""

import json

import pytest

from text_classification import report, train
from text_classification.synthetic import make_synthetic_text

RUN = {
    "fast": True,
    "data_source": "synthetic_test",
    "expected_rows": None,
    "expected_classes": None,
}


def test_pipeline_runs_and_writes_valid_results(tmp_path):
    metrics = train.run(make_synthetic_text(seed=1), tmp_path, **RUN)
    for name in ("metrics.json", "error_analysis.json", "confusion_matrix.png"):
        assert (tmp_path / name).exists(), name
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["fast_mode"] is True and saved["data_source"] == "synthetic_test"
    assert set(saved["models"]) == {
        "baseline_majority",
        "tfidf_nb",
        "tfidf_logreg",
        "tfidf_linear_svc",
    }
    assert set(saved["traps"]) == {"metadata_kept", "vectorizer_fit_on_test"}
    low, high = saved["headline"]["ci95"]
    assert low <= high
    assert isinstance(saved["gate"]["passed"], bool)
    assert metrics["headline"]["model"] == saved["headline"]["model"]


def test_headline_is_chosen_on_validation_not_test(tmp_path):
    metrics = train.run(make_synthetic_text(seed=2), tmp_path, **RUN)
    tuned = {n: r["val_macro_f1"] for n, r in metrics["models"].items() if n != "baseline_majority"}
    assert tuned[metrics["headline"]["model"]] == max(tuned.values())


def test_models_beat_the_majority_baseline_on_learnable_text(tmp_path):
    metrics = train.run(make_synthetic_text(seed=3), tmp_path, **RUN)
    assert metrics["headline"]["value"] > 10 * metrics["baseline"]["test_macro_f1"]
    assert metrics["gate"]["interval_above_majority_baseline"] is True


def test_metadata_trap_surfaces_header_tokens(tmp_path):
    metrics = train.run(make_synthetic_text(seed=4), tmp_path, **RUN)
    trap = metrics["traps"]["metadata_kept"]
    assert trap["share_of_top_features_absent_from_clean_vocabulary"] > 0
    tokens = [t for toks in trap["top_features_metadata_kept"].values() for t in toks]
    assert any(t.startswith(("org", "user")) for t in tokens)
    assert trap["test_macro_f1_metadata_kept"] >= trap["test_macro_f1_metadata_removed"] - 1e-9


def test_empty_documents_are_reported_not_dropped(tmp_path):
    train.run(make_synthetic_text(seed=5, empty_rate=0.1), tmp_path, **RUN)
    analysis = json.loads((tmp_path / "error_analysis.json").read_text(encoding="utf-8"))
    assert analysis["empty_documents"]["n"] > 0


def test_same_seed_gives_same_metrics(tmp_path):
    df = make_synthetic_text(seed=6)
    a = train.run(df, tmp_path / "a", **RUN)
    b = train.run(df, tmp_path / "b", **RUN)
    assert a["headline"] == b["headline"] and a["models"] == b["models"]


def test_invalid_frame_is_rejected(tmp_path):
    df = make_synthetic_text(seed=7).drop(columns=["text_raw"])
    with pytest.raises(ValueError, match="missing columns"):
        train.run(df, tmp_path, **RUN)


def test_reports_render_and_flag_non_release_runs(tmp_path):
    train.run(make_synthetic_text(seed=8), tmp_path, **RUN)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    analysis = json.loads((tmp_path / "error_analysis.json").read_text(encoding="utf-8"))
    card = report.render_model_card(metrics, analysis)
    memo = report.render_decision_memo(metrics, analysis)
    assert "Not a release result" in card and "Not a release result" in memo
    assert chr(0x2014) not in card and chr(0x2014) not in memo
