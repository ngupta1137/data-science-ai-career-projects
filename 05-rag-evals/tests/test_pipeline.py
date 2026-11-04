import json

import pytest

from rag_eval import config, report, train
from rag_eval.generate import build_documents, build_questions


@pytest.fixture(scope="module")
def fast_metrics(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    metrics = train.run(build_documents(), build_questions(), out, fast=True)
    return metrics, out


def test_fast_run_writes_marked_results(fast_metrics):
    metrics, out = fast_metrics
    assert metrics["fast_mode"] is True
    for name in ("metrics.json", "error_analysis.json", "correctness_by_type.png"):
        assert (out / name).exists()
    assert "Not a release result" in report.render_model_card(metrics, {"by_type": {}})


def test_pipeline_is_deterministic(fast_metrics, tmp_path):
    first, _ = fast_metrics
    again = train.run(build_documents(), build_questions(), tmp_path, fast=True)
    assert again["retrieval"] == first["retrieval"]
    assert again["end_to_end"] == first["end_to_end"]


def test_random_baseline_is_far_below_real_retrievers(fast_metrics):
    r = fast_metrics[0]["retrieval"]
    assert r["bm25"]["test_mrr"] > r["random"]["test_mrr"] + 0.3


def test_taxonomy_covers_every_test_question(fast_metrics):
    m = fast_metrics[0]
    assert sum(m["end_to_end"]["taxonomy"].values()) == m["data"]["test_questions"]


def test_always_abstain_equals_share_of_unanswerable(fast_metrics):
    m = fast_metrics[0]
    n_unans = sum(
        v
        for k, v in m["end_to_end"]["taxonomy"].items()
        if k in ("correct_refusal", "answered_unanswerable")
    )
    assert m["end_to_end"]["always_abstain_correct"] == pytest.approx(
        n_unans / m["data"]["test_questions"]
    )


def test_failure_category_rules():
    q_un = {"type": "unanswerable_attribute", "gold_docs": [], "answer": None}
    base = {"retrieved": ["a"], "source": "a", "confidence": 0.5}
    assert train.failure_category({**base, "question": q_un, "answer": "5 days"}, 0.3) == (
        "answered_unanswerable"
    )
    assert train.failure_category({**base, "question": q_un, "answer": None}, 0.3) == (
        "correct_refusal"
    )
    q = {"type": "stale_version", "gold_docs": ["new"], "answer": "16 weeks"}
    old = {
        "question": q,
        "retrieved": ["new", "archive_x"],
        "source": "archive_x",
        "confidence": 0.9,
    }
    assert train.failure_category({**old, "answer": "12 weeks"}, 0.3) == "stale_source_chosen"
    assert train.failure_category({**old, "answer": "16 weeks", "source": "new"}, 0.3) == "correct"
    miss = {"question": q, "retrieved": ["z"], "source": "z", "confidence": 0.9, "answer": "1 day"}
    assert train.failure_category(miss, 0.3) == "retrieval_miss"
    low = {
        "question": q,
        "retrieved": ["new"],
        "source": "new",
        "confidence": 0.1,
        "answer": "16 weeks",
    }
    assert train.failure_category(low, 0.3) == "false_abstain"
    span = {**low, "confidence": 0.9, "answer": "2 weeks"}
    assert train.failure_category(span, 0.3) == "wrong_span"


def test_choose_threshold_picks_smallest_best():
    q_ans = {"type": "lexical", "gold_docs": ["d"], "answer": "5 days"}
    q_un = {"type": "unanswerable_offtopic", "gold_docs": [], "answer": None}
    outs = [
        {"question": q_ans, "answer": "5 days", "confidence": 0.8},
        {"question": q_un, "answer": "9 days", "confidence": 0.2},
    ]
    t = train.choose_threshold(outs)
    assert train.is_correct(outs[0], t) and train.is_correct(outs[1], t)
    assert 0.2 < t <= 0.8


def test_committed_results_match_a_fresh_full_run(tmp_path):
    committed = config.RESULTS_DIR / "metrics.json"
    if not committed.exists():
        pytest.skip("no committed results")
    stored = json.loads(committed.read_text(encoding="utf-8"))
    if stored["fast_mode"]:
        pytest.skip("committed results are from a fast run")
    fresh = train.run(
        build_documents(), build_questions(), tmp_path, data_sha256=stored["data"]["sha256"]
    )
    assert fresh["headline"]["retriever"] == stored["headline"]["retriever"]
    assert fresh["end_to_end"]["correct"] == pytest.approx(
        stored["end_to_end"]["correct"], abs=1e-9
    )
    assert fresh["headline"]["value"] == pytest.approx(stored["headline"]["value"], abs=1e-6)
